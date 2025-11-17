import numpy as np
import scanpy as sc
import anndata as ad
import pandas as pd
import scipy.sparse
import torch
from torch_geometric.nn import knn_graph, radius_graph
from torch_geometric.utils import to_undirected
from torch.amp import autocast
from torch.amp import GradScaler

def Cal_Spatial_Net(
    adata,
    mode='Radius', # 'Radius' or 'KNN'
    rad_cutoff=None, # used when mode=='Radius'
    rad_coef=1.1, # used when mode=='Radius' and rad_cutoff is None
    k_cutoff=6, # used when mode=='KNN'
    coor_key='spatial',
    num_workers=1,
    verbose=True,
):
    assert mode in ["Radius", "KNN"]
    if verbose:
        print("Calculating spatial neighbor graph ...")
    if mode == 'KNN':
        edge_index = knn_graph(
            x=torch.tensor(adata.obsm[coor_key]),
            flow="target_to_source",
            k=k_cutoff,
            loop=True,
            num_workers=num_workers,
        )
        edge_index = to_undirected(
            edge_index, num_nodes=adata.shape[0]
        )
    elif mode == 'Radius':
        coords = torch.tensor(adata.obsm[coor_key]).float()
        dist_mat = torch.cdist(coords, coords, p=2)
        eye_mask = torch.eye(coords.size(0), dtype=torch.bool, device=coords.device)
        d_min = dist_mat.masked_fill(eye_mask, float('inf')).min()

        if rad_cutoff is None:
            rad_cutoff = d_min * rad_coef

        edge_index = radius_graph(
            x=coords,
            flow="target_to_source",
            r=rad_cutoff,
            loop=True,  
            num_workers=num_workers, 
        )

    graph_df = pd.DataFrame(edge_index.numpy().T, columns=["Spot1", "Spot2"])
    id_spot_trans = dict(zip(range(adata.n_obs), adata.obs_names))
    graph_df["Spot1"] = graph_df["Spot1"].map(id_spot_trans)
    graph_df["Spot2"] = graph_df["Spot2"].map(id_spot_trans)
    adata.uns["Spatial_Net"] = graph_df

    if verbose:
        print(f"The graph contains {graph_df.shape[0]} edges, {adata.n_obs} spots.")
        print(f"{graph_df.shape[0]/adata.n_obs} neighbors per spot on average.")

def preprocess(
    adata_st_input,
    adata_ref_input,
    celltype_ref_col='celltype',
    sample_col=None,
    celltype_ref=None,
    n_hvg_group=500,
    coor_key="spatial",
    min_genes=1,
    min_cells=1,
):
    adata_st = adata_st_input.copy()

    print("Finding highly variable genes...")
    adata_ref = adata_ref_input.copy()
    adata_ref.var_names_make_unique()
    # Remove mt-genes
    adata_ref = adata_ref[
        :,
        np.array(~adata_ref.var.index.isna())
        & np.array(~adata_ref.var_names.str.startswith("mt-"))
        & np.array(~adata_ref.var_names.str.startswith("MT-")),
    ]
    if celltype_ref is not None:
        if not isinstance(celltype_ref, list):
            raise ValueError("'celltype_ref' must be a list!")
        else:
            adata_ref = adata_ref[
                [
                    (t in celltype_ref)
                    for t in adata_ref.obs[celltype_ref_col].values.astype(str)
                ],
                :,
            ]
    else:
        celltype_counts = adata_ref.obs[celltype_ref_col].value_counts()
        celltype_ref = list(celltype_counts.index[celltype_counts > 1])
        adata_ref = adata_ref[
            [
                (t in celltype_ref)
                for t in adata_ref.obs[celltype_ref_col].values.astype(str)
            ],
            :,
        ]

    # Remove cells and genes with zero counts
    sc.pp.filter_cells(adata_ref, min_genes=min_genes)
    sc.pp.filter_genes(adata_ref, min_cells=min_cells)
    
    # preprocess ST adatas
    adata_st.var_names_make_unique()
    # Remove mt-genes
    adata_st = adata_st[
        :,
        (
            np.array(~adata_st.var.index.str.startswith("mt-"))
            & np.array(~adata_st.var.index.str.startswith("MT-"))
        ),
    ]

    # Take gene intersection
    #genes = list(adata_st.var.index & adata_ref.var.index)
    genes = list(adata_st.var_names.intersection(adata_ref.var_names))
    adata_ref = adata_ref[:, genes]
    adata_st = adata_st[:, genes]

    # Select hvgs
    adata_ref_log = adata_ref.copy()
    sc.pp.log1p(adata_ref_log)
    hvgs = select_hvgs(
        adata_ref_log, celltype_ref_col=celltype_ref_col, num_per_group=n_hvg_group
    )

    print("%d highly variable genes selected." % len(hvgs))
    adata_ref = adata_ref[:, hvgs]

    print("Calculate basis for deconvolution...")
    sc.pp.filter_cells(adata_ref, min_genes=1)
    sc.pp.normalize_total(adata_ref, target_sum=1)
    celltype_list = list(
        sorted(set(adata_ref.obs[celltype_ref_col].values.astype(str)))
    )

    basis = np.zeros((len(celltype_list), len(adata_ref.var.index)))
    if sample_col is not None:
        sample_list = list(sorted(set(adata_ref.obs[sample_col].values.astype(str))))
        for i in range(len(celltype_list)):
            c = celltype_list[i]
            tmp_list = []
            for j in range(len(sample_list)):
                s = sample_list[j]
                tmp = adata_ref[
                    (adata_ref.obs[celltype_ref_col].values.astype(str) == c)
                    & (adata_ref.obs[sample_col].values.astype(str) == s),
                    :,
                ].X
                if scipy.sparse.issparse(tmp):
                    tmp = tmp.toarray()
                if tmp.shape[0] >= 3:
                    tmp_list.append(np.mean(tmp, axis=0).reshape((-1)))
            tmp_mean = np.mean(tmp_list, axis=0)
            if scipy.sparse.issparse(tmp_mean):
                tmp_mean = tmp_mean.toarray()
            print(
                "%d batches are used for computing the basis vector of cell type <%s>."
                % (len(tmp_list), c)
            )
            basis[i, :] = tmp_mean
    else:
        for i in range(len(celltype_list)):
            c = celltype_list[i]
            tmp = adata_ref[
                adata_ref.obs[celltype_ref_col].values.astype(str) == c, :
            ].X
            if scipy.sparse.issparse(tmp):
                tmp = tmp.toarray()
            basis[i, :] = np.mean(tmp, axis=0).reshape((-1))
    
    adata_basis = ad.AnnData(X=basis)
    df_gene = pd.DataFrame({"gene": adata_ref.var.index})
    df_gene = df_gene.set_index("gene")
    df_celltype = pd.DataFrame({"celltype": celltype_list})
    df_celltype = df_celltype.set_index("celltype")
    adata_basis.obs = df_celltype
    adata_basis.var = df_gene
    adata_basis = adata_basis[~np.isnan(adata_basis.X[:, 0])]

    print("Preprocess ST data...")
    # Store counts and library sizes for Poisson modeling
    st_mtx = adata_st[:, hvgs].X.copy()
    if scipy.sparse.issparse(st_mtx):
        st_mtx = st_mtx.toarray()
    adata_st.obsm["count"] = st_mtx
    st_library_size = np.sum(st_mtx, axis=1)
    adata_st.obs["library_size"] = st_library_size

    # Normalize ST data
    sc.pp.normalize_total(adata_st, target_sum=1e4)
    sc.pp.log1p(adata_st)
    adata_st = adata_st[:, hvgs]
    if scipy.sparse.issparse(adata_st.X):
        adata_st.X = adata_st.X.toarray()

    return adata_st, adata_basis


def select_hvgs(adata_ref, celltype_ref_col, num_per_group=200):
    sc.tl.rank_genes_groups(
        adata_ref,
        groupby=celltype_ref_col,
        method="t-test",
        key_added="ttest",
        use_raw=False,
    )
    markers_df = pd.DataFrame(adata_ref.uns["ttest"]["names"]).iloc[0:num_per_group, :]
    genes = sorted(list(np.unique(markers_df.melt().value.values)))
    return genes


class OptimHelper:
    def __init__(self, optimizer, use_amp: bool):
        self.optimizer = optimizer
        self.scaler = GradScaler(enabled=use_amp)
        self.use_amp = use_amp

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def backward(self, loss):
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self):
        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

def patches(total_cols: int, size: int):
    for s in range(0, total_cols, size):
        e = min(s + size, total_cols)
        yield s, e