import torch
import torch.optim as optim
from torch.amp import autocast
from torch.amp import GradScaler
import numpy as np
import pandas as pd
import scipy.sparse
from tqdm import tqdm
from MacSGP.networks import *
from MacSGP.utils import patches, OptimHelper

class Model():

    def __init__(self, adata_st, adata_basis,
                 hidden_dims=[256, 128],
                 n_layers=4,
                 n_SGPs=1,
                 alpha_gcn=None,
                 theta_gcn=None,
                 coef_fe=0.1,
                 coef_reg=1.2,
                 training_steps=3000,
                 lr=0.002,
                 seed=1234,
                 estimate_gamma=False,
                 estimate_gamma_k=True,
                 estimate_alpha=False,
                 ): 
        
        self.training_steps = training_steps

        self.adata_st = adata_st.copy()
        self.celltypes = list(adata_basis.obs.index)

        # add device
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

        self.hidden_dims = [adata_st.shape[1]] + hidden_dims
        self.n_celltype = adata_basis.shape[0]

        G_df = adata_st.uns["Spatial_Net"].copy()
        spots = np.array(adata_st.obs_names)
        spots_id_tran = dict(zip(spots, range(spots.shape[0])))
        G_df["Spot1"] = G_df["Spot1"].map(spots_id_tran)
        G_df["Spot2"] = G_df["Spot2"].map(spots_id_tran)

        G = scipy.sparse.coo_matrix(
            (np.ones(G_df.shape[0]), (G_df["Spot1"], G_df["Spot2"])),
            shape=(adata_st.n_obs, adata_st.n_obs),
        )

        self.edge_index = torch.LongTensor(np.nonzero(G)).to(self.device)

        self.net = SGPNet(hidden_dims=self.hidden_dims,
                            n_layers = n_layers,
                            n_celltypes=self.n_celltype,
                            alpha_gcn = alpha_gcn,
                            theta_gcn = theta_gcn, 
                            estimate_gamma=estimate_gamma,
                            init_gamma=torch.from_numpy(np.array(adata_st.var["gamma"].values)).float().to(self.device),
                            estimate_gamma_k=estimate_gamma_k,
                            estimate_alpha=estimate_alpha,
                            n_SGPs=n_SGPs,
                            coef_fe=coef_fe,
                            coef_reg=coef_reg,
                            ).to(self.device)
        
        self.optimizer = optim.Adamax(list(self.net.parameters()), lr=lr)

        if scipy.sparse.issparse(adata_st.X):
            self.X = torch.from_numpy(adata_st.X.toarray()).float().to(self.device)
        else:
            self.X = torch.from_numpy(adata_st.X).float().to(self.device)
        self.Y = torch.from_numpy(np.array(adata_st.obsm["count"])).float().to(self.device)
        self.lY = torch.from_numpy(np.array(adata_st.obs["library_size"].values.reshape(-1, 1))).float().to(self.device)
        self.basis = torch.from_numpy(np.array(adata_basis.X)).float().to(self.device)
        self.gamma = torch.from_numpy(np.array(adata_st.var["gamma"].values)).float().to(self.device)
        self.alpha = torch.from_numpy(np.array(adata_st.obs["alpha"].values)).float().to(self.device)
        self.proportion = torch.from_numpy(np.array(adata_st.obsm["proportion"])).float().to(self.device)


        self.loss = list()
        self.decon_loss = list()
        self.features_loss = list()
        self.regularization_loss = list()

    def train(
        self,
        report_loss: bool = True,
        step_interval: int = 200,
        test: bool = False,
        gene_patch: bool = False,
        patch_size: int = 200,
        use_amp: bool = False,
    ):
        self.net.train()
        opt = OptimHelper(self.optimizer, use_amp=use_amp)

        for step in tqdm(range(self.training_steps)):
            if gene_patch:
                n_patches = int(np.ceil(self.Y.shape[1] / patch_size))
                opt.zero_grad()
                loss_sum = 0.0
                decon_sum = 0.0
                reg_sum = 0.0

                for start, end in patches(self.Y.shape[1], patch_size):
                    with autocast(enabled=use_amp, device_type = self.device.type):
                        loss_patch = self.net(
                            node_feats=self.X,
                            edge_index=self.edge_index,
                            count_matrix=self.Y[:, start:end],
                            library_size=self.lY,
                            basis=self.basis[:, start:end],
                            alpha=self.alpha,
                            proportion=self.proportion,
                            gene_index=torch.arange(start, end, device=self.device),
                            loss_mode="DECONV",
                            n_patchs=n_patches,  
                            test=test,
                        )
                    loss_sum += float(loss_patch.detach())
                    decon_sum += float(self.net.decon_loss.detach())
                    if hasattr(self.net, "regularization_loss") and self.net.regularization_loss is not None:
                        reg_sum += float(self.net.regularization_loss.detach())
                    opt.backward(loss_patch)

                with autocast(enabled=use_amp, device_type = self.device.type):
                    feature_loss = self.net(
                        node_feats=self.X,
                        edge_index=self.edge_index,
                        count_matrix=None,
                        library_size=None,
                        basis=None,
                        alpha=self.alpha,
                        proportion=self.proportion,
                        gene_index=None,
                        loss_mode="RECON",
                        test=test,
                    )
                loss_sum += float(feature_loss.detach())
                opt.backward(feature_loss)
                opt.step()

                self.loss.append(loss_sum)
                self.decon_loss.append(decon_sum)
                self.features_loss.append(float(self.net.features_loss.detach()))
                self.regularization_loss.append(reg_sum)
            else:
                opt.zero_grad()
                with autocast(enabled=use_amp, device_type = self.device.type):
                    loss = self.net(
                        node_feats=self.X,
                        edge_index=self.edge_index,
                        count_matrix=self.Y,
                        library_size=self.lY,
                        basis=self.basis,
                        alpha=self.alpha,
                        proportion=self.proportion,
                        gene_index=None,
                        test=test,
                    )
                opt.backward(loss)
                opt.step()

                self.loss.append(float(loss.detach()))
                self.decon_loss.append(float(self.net.decon_loss.detach()))
                self.features_loss.append(float(self.net.features_loss.detach()))
                self.regularization_loss.append(
                    float(self.net.regularization_loss.detach()) if hasattr(self.net, "regularization_loss") and self.net.regularization_loss is not None else 0.0
                )

            if report_loss and (step % step_interval == 0):
                tqdm.write(
                    f"Step: {step}, Loss: {self.loss[-1]:.4f}, d_loss: {self.decon_loss[-1]:.4f}, "
                    f"f_loss: {self.features_loss[-1]:.4f}, reg_loss: {self.regularization_loss[-1]:.4f}"
                )

    def eval(self):
        self.net.eval()

        self.Z, self.factor, self.loading, self.gamma, self.alpha_res = self.net.evaluate(self.X, self.edge_index)

        # add learned representations to full ST adata object
        embeddings = self.Z.detach().cpu().numpy()
        cell_reps = pd.DataFrame(embeddings)
        cell_reps.index = self.adata_st.obs.index
        self.adata_st.obsm['latent'] = cell_reps.loc[self.adata_st.obs_names, ].values

        factor = self.factor.detach().cpu().numpy()
        loading = self.loading.detach().cpu().numpy()
        if factor.shape[2] > 1:
            for i in range(factor.shape[0]):
                factor_df = pd.DataFrame(factor[i], index=self.adata_st.obs.index, columns=[f'factor_{j}' for j in range(factor.shape[2])])
                self.adata_st.obsm[f'{self.celltypes[i]}'] = factor_df
            for i in range(loading.shape[0]):
                loading_df = pd.DataFrame(loading[i].T, index=self.adata_st.var.index, columns=[f'loading_{j}' for j in range(loading.shape[1])])
                self.adata_st.varm[f'{self.celltypes[i]}'] = loading_df
        else:
            factor = factor[:, :, 0]
            self.adata_st.obsm['factor'] = pd.DataFrame(factor.T, index=self.adata_st.obs.index, columns=self.celltypes)
            loading = loading[:, 0, :]
            self.adata_st.varm['loading'] = pd.DataFrame(loading.T, index=self.adata_st.var.index, columns=self.celltypes)

        return self.adata_st
    
class Model_deconv():

    def __init__(self, adata_st, adata_basis,
                 hidden_dims=[256, 128],
                 n_layers = 4,
                 alpha_gcn = None,
                 theta_gcn = None, 
                 coef_fe=0.1,
                 training_steps=10000,
                 lr=2e-3,
                 seed=1234,
                 ):
        
        self.training_steps = training_steps

        self.adata_st = adata_st.copy()
        self.celltypes = list(adata_basis.obs.index)

        # add device
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

        self.hidden_dims = [adata_st.shape[1]] + hidden_dims
        self.n_celltype = adata_basis.shape[0]

        G_df = adata_st.uns["Spatial_Net"].copy()
        spots = np.array(adata_st.obs_names)
        spots_id_tran = dict(zip(spots, range(spots.shape[0])))
        G_df["Spot1"] = G_df["Spot1"].map(spots_id_tran)
        G_df["Spot2"] = G_df["Spot2"].map(spots_id_tran)

        G = scipy.sparse.coo_matrix(
            (np.ones(G_df.shape[0]), (G_df["Spot1"], G_df["Spot2"])),
            shape=(adata_st.n_obs, adata_st.n_obs),
        )
        self.edge_index = torch.LongTensor(np.nonzero(G)).to(self.device)

        self.net = DeconvNet(hidden_dims=self.hidden_dims,
                            n_layers = n_layers,
                            n_celltypes=self.n_celltype,
                            alpha_gcn = alpha_gcn,
                            theta_gcn = theta_gcn, 
                            coef_fe=coef_fe,
                            ).to(self.device)
        
        self.optimizer = optim.Adamax(list(self.net.parameters()), lr=lr)

        if scipy.sparse.issparse(adata_st.X):
            self.X = torch.from_numpy(adata_st.X.toarray()).float().to(self.device)
        else:
            self.X = torch.from_numpy(adata_st.X).float().to(self.device)
        self.Y = torch.from_numpy(np.array(adata_st.obsm["count"])).float().to(self.device)
        self.lY = torch.from_numpy(np.array(adata_st.obs["library_size"].values.reshape(-1, 1))).float().to(self.device)
        self.basis = torch.from_numpy(np.array(adata_basis.X)).float().to(self.device)

    def train(
        self,
        report_loss: bool = True,
        step_interval: int = 200,
        test: bool = False,
        gene_patch: bool = False,
        patch_size: int = 200,
        use_amp: bool = False,
    ):
        self.net.train()
        opt = OptimHelper(self.optimizer, use_amp=use_amp)

        for step in tqdm(range(self.training_steps)):
            if gene_patch:
                n_patches = int(np.ceil(self.Y.shape[1] / patch_size))
                opt.zero_grad()
                loss_sum = 0.0
                decon_sum = 0.0

                for start, end in patches(self.Y.shape[1], patch_size):
                    with autocast(enabled=use_amp, device_type = self.device.type):
                        loss_patch = self.net(
                            node_feats=self.X,
                            edge_index=self.edge_index,
                            count_matrix=self.Y[:, start:end],
                            library_size=self.lY,
                            basis=self.basis[:, start:end],
                            gene_index=torch.arange(start, end, device=self.device),
                            loss_mode="DECONV",
                        )
                    loss_sum += float(loss_patch.detach())
                    decon_sum += float(self.net.decon_loss.detach())
                    opt.backward(loss_patch)

                # RECON 一次
                with autocast(enabled=use_amp, device_type = self.device.type):
                    feature_loss = self.net(
                        node_feats=self.X,
                        edge_index=self.edge_index,
                        count_matrix=None,
                        library_size=None,
                        basis=None,
                        gene_index=None,
                        loss_mode="RECON",
                    )
                loss_sum += float(feature_loss.detach())
                opt.backward(feature_loss)
                opt.step()

            else:
                opt.zero_grad()
                with autocast(enabled=use_amp, device_type = self.device.type):
                    loss_sum = self.net(
                        node_feats=self.X,
                        edge_index=self.edge_index,
                        count_matrix=self.Y,
                        library_size=self.lY,
                        basis=self.basis,
                    )
                opt.backward(loss_sum)
                opt.step()
                decon_sum = float(self.net.decon_loss.detach())

            if report_loss and (step % step_interval == 0):
                tqdm.write(
                    f"Step: {step}, Loss: {float(loss_sum):.4f}, d_loss: {float(decon_sum):.4f}, f_loss: {float(self.net.features_loss.detach()):.4f}"
                ) 
    
    def eval(self):
        self.net.eval()
        self.Z, self.beta, self.alpha, self.gamma = self.net.evaluate(self.X, self.edge_index)

        # add learned representations to full ST adata object
        embeddings = self.Z.detach().cpu().numpy()
        cell_reps = pd.DataFrame(embeddings)
        cell_reps.index = self.adata_st.obs.index
        self.adata_st.obsm['latent'] = cell_reps.loc[self.adata_st.obs_names, ].values

        b = self.beta.detach().cpu().numpy()
        alpha = self.alpha.detach().cpu().numpy()
        gamma = self.gamma.detach().cpu().numpy()

        proportion = pd.DataFrame(b, index=self.adata_st.obs.index, columns=self.celltypes)
        self.adata_st.obsm['proportion'] = proportion
        self.adata_st.var['gamma'] = gamma.T
        self.adata_st.obs['alpha'] = alpha

        return self.adata_st
