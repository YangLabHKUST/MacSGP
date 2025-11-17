import numpy as np
import scanpy as sc
import anndata as ad
import pandas as pd
import scipy.sparse
import matplotlib
import matplotlib.pyplot as plt
import os
from matplotlib import cm
from matplotlib import colors

def plot_spatial_ct(adata, celltype_plot=None, title = None, show_title=True,
                index='factor', reverse = False,
                proportion_threshold=0., num_threshold=100, 
                cmap='bwr', vmin=None, vmax=None, crop_coords=None,
                spot_size=5, colorbar_loc='right', alpha_img=0.7, hide_image=False, ncols=5,
                return_ax=False, save_path=None, frameon=False
                ):

    celltype = adata.obsm['proportion'].columns
    if celltype_plot is None:
        celltype_plot = celltype
        celltype_plot = list(celltype_plot)
    
    celltype_drop = []
    celltype_filtered = []
    for i in range(len(celltype_plot)):
        if (adata.obsm['proportion'][celltype_plot[i]]>proportion_threshold).sum() < num_threshold:
            celltype_drop.append(celltype_plot[i])
        else:
            celltype_filtered.append(celltype_plot[i])
    print('Dropping cell types:', celltype_drop)
    celltype_plot = celltype_filtered
    
    adata_tmp = adata.copy()
    factor = adata_tmp.obsm[index].copy()

    if title is None:
        title = index

    if len(celltype_plot) == 0:
        print("No cell types to plot after filtering.")
        return
    elif len(celltype_plot) == 1:
        ct = celltype_plot[0]
        adata_tmp.obs['tmp'] = factor[ct]
        filter = adata_tmp.obsm['proportion'][ct]>=proportion_threshold
        if filter.sum() == 0:
            print(f"No spots meet the proportion threshold for cell type {ct}.")
            return
        adata_tmp.obs['tmp'] = adata_tmp.obs['tmp'].where(filter, np.nan)
        if reverse:
            adata_tmp.obs['tmp'] = -adata_tmp.obs['tmp']
        nrows, ncols = 1, 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
        sc.pl.spatial(
            adata_tmp,
            color='tmp',
            cmap=cmap,
            na_color=None,
            spot_size=spot_size,
            colorbar_loc=colorbar_loc,  
            alpha_img=alpha_img,
            vcenter=0 if index == 'factor' else None,
            vmin=vmin,
            vmax=vmax,
            title=f'{title} of {ct}' if show_title else '',
            frameon=frameon,
            crop_coord=crop_coords,
            ax=axes, 
            show=False,  
            img_key=None if hide_image else 'hires'
        )
        if save_path is not None:
            plt.savefig(save_path)
            plt.close(fig)
            return
        if return_ax:
            return axes
        plt.show()
        return
    
    ncols = ncols
    nrows = (len(celltype_plot) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    total_slots = nrows * ncols
    
    for i, ct in enumerate(celltype_plot):
        ax = axes.flat[i]
        adata_tmp.obs['tmp'] = factor[ct]
        filter = adata_tmp.obsm['proportion'][ct]>=proportion_threshold
        if filter.sum() == 0:
            print(f"No spots meet the proportion threshold for cell type {ct}.")
        adata_tmp.obs['tmp'] = adata_tmp.obs['tmp'].where(filter, np.nan)

        sc.pl.spatial(
            adata_tmp,
            color='tmp',
            cmap=cmap,
            spot_size=spot_size,
            colorbar_loc=colorbar_loc,
            alpha_img=alpha_img,
            vcenter=0 if index == 'factor' else None,
            vmin=vmin,
            vmax=vmax,
            title=f'{title} of {ct}' if show_title else '',
            frameon=frameon,
            crop_coord=crop_coords,
            ax=ax,  
            show=False,  
            img_key=None if hide_image else 'hires'
        )
    if i + 1 < total_slots:
        for j in range(i + 1, total_slots):
            ax = axes.flat[j]
            ax.set_axis_off()
    
    fig.subplots_adjust(left=0.03, right=0.99, top=0.96, bottom=0.05,
                    wspace=0.08, hspace=0.08)
    if save_path is not None:
        plt.savefig(save_path)
        plt.close(fig)
        return
    if return_ax:
        return axes
    plt.show()
