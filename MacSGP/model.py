import torch
import torch.optim as optim
from torch.amp import autocast
from torch.amp import GradScaler
import numpy as np
import pandas as pd
import scipy.sparse
from tqdm import tqdm
from MacSGP.networks import *

class Model():

    def __init__(self, adata_st, adata_basis,
                 hidden_dims=[256, 128],
                 n_layers=4,
                 n_SGPs=1,
                 alpha_gcn=None,
                 theta_gcn=None,
                 coef_fe=0.1,
                 coef_reg=0.1,
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

    def train(self, report_loss=True, step_interval=200, test=False, gene_patch=False, patch_size=200, use_amp=False):
        if gene_patch:
            n_patchs = int(np.ceil(self.Y.shape[1] / patch_size))
            scaler = GradScaler()

        self.net.train()

        for step in tqdm(range(self.training_steps)):
            if gene_patch:
                loss_sum = 0
                decon_loss = 0
                regularization_loss = 0
                self.optimizer.zero_grad()

                for i in range(n_patchs):
                    if i == n_patchs - 1:
                        start = i * patch_size
                        end = self.Y.shape[1]
                    else:
                        start = i * patch_size
                        end = (i + 1) * patch_size
                    if use_amp:
                        with autocast():
                            loss = self.net(node_feats=self.X,
                                            edge_index=self.edge_index,
                                            count_matrix=self.Y[:, start:end],
                                            library_size=self.lY,
                                            basis=self.basis[:, start:end],
                                            alpha = self.alpha,
                                            proportion = self.proportion,
                                            gene_index=torch.arange(start, end).to(self.device),
                                            loss_mode='DECONV',
                                            n_patchs=n_patchs,
                                            test=test
                                            )
                    else:
                        loss = self.net(node_feats=self.X,
                                        edge_index=self.edge_index,
                                        count_matrix=self.Y[:, start:end],
                                        library_size=self.lY,
                                        basis=self.basis[:, start:end],
                                        alpha = self.alpha,
                                        proportion = self.proportion,
                                        gene_index=torch.arange(start, end).to(self.device),
                                        loss_mode='DECONV',
                                        n_patchs=n_patchs,
                                        test=test
                                        )
                    loss_sum += loss.item()
                    decon_loss += self.net.decon_loss.item()
                    regularization_loss += self.net.regularization_loss.item()
                    if use_amp:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                if test:
                    print(torch.cuda.memory_summary())
                torch.cuda.empty_cache()

                if use_amp:
                    with autocast():
                        feature_loss = self.net(node_feats=self.X,
                                                edge_index=self.edge_index,
                                                count_matrix=None,
                                                library_size=None,
                                                basis=None,
                                                alpha = self.alpha,
                                                proportion = self.proportion,
                                                gene_index=None,
                                                loss_mode='RECON',
                                                test=test
                                                )
                else:
                    feature_loss = self.net(node_feats=self.X,
                                            edge_index=self.edge_index,
                                            count_matrix=None,
                                            library_size=None,
                                            basis=None,
                                            alpha = self.alpha,
                                            proportion = self.proportion,
                                            gene_index=None,
                                            loss_mode='RECON',
                                            test=test
                                            )
                loss_sum += feature_loss.item()
                if use_amp:
                    scaler.scale(feature_loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    feature_loss.backward()
                    self.optimizer.step()

                self.features_loss.append(self.net.features_loss.item())
                self.regularization_loss.append(regularization_loss)
                self.loss.append(loss_sum)
                self.decon_loss.append(decon_loss)
            else:
                loss = self.net(node_feats=self.X,
                                edge_index=self.edge_index,
                                count_matrix=self.Y,
                                library_size=self.lY,
                                basis=self.basis,
                                alpha = self.alpha,
                                proportion = self.proportion,
                                gene_index=None,
                                test=test
                                )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.loss.append(loss.item())
                self.decon_loss.append(self.net.decon_loss.item())
                self.features_loss.append(self.net.features_loss.item())
                self.regularization_loss.append(self.net.regularization_loss.item())

            if report_loss:
                if not step % step_interval:
                    print("Step: %s, Loss: %.4f, d_loss: %.4f, f_loss: %.4f, reg_loss: %.4f" % (step, self.loss[-1], 
                    self.decon_loss[-1], self.features_loss[-1], self.regularization_loss[-1]))        

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
        for i in range(factor.shape[0]):
            factor_df = pd.DataFrame(factor[i], index=self.adata_st.obs.index, columns=[f'factor_{j}' for j in range(factor.shape[2])])
            self.adata_st.obsm[f'{self.celltypes[i]}'] = factor_df
        for i in range(loading.shape[0]):
            loading_df = pd.DataFrame(loading[i].T, index=self.adata_st.var.index, columns=[f'loading_{j}' for j in range(loading.shape[1])])
            self.adata_st.varm[f'{self.celltypes[i]}'] = loading_df

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

    def train(self, report_loss=True, step_interval=200, test= False, gene_patch=False, patch_size=200, use_amp=True):
        if gene_patch:
            n_patchs = int(np.ceil(self.Y.shape[1] / patch_size))
            scaler = GradScaler()

        self.net.train()      

        for step in tqdm(range(self.training_steps)):
            # if test:
            #     print(torch.cuda.max_memory_allocated()/1024/1024)
            #     print(torch.cuda.memory_allocated()/1024/1024)
            #     print(torch.cuda.memory_reserved()/1024/1024)
            if gene_patch:
                loss_sum = 0
                decon_loss = 0
                torch.cuda.empty_cache()
                self.optimizer.zero_grad()

                for i in range(n_patchs):
                    if i == n_patchs - 1:
                        start = i * patch_size
                        end = self.Y.shape[1]
                    else:
                        start = i * patch_size
                        end = (i + 1) * patch_size
                    if use_amp:
                        with autocast():
                            loss = self.net(node_feats=self.X,
                                            edge_index=self.edge_index,
                                            count_matrix=self.Y[:, start:end],
                                            library_size=self.lY,
                                            basis=self.basis[:, start:end],
                                            gene_index=torch.arange(start, end).to(self.device),
                                            loss_mode='DECONV',
                                            )
                    else:
                        loss = self.net(node_feats=self.X,
                                        edge_index=self.edge_index,
                                        count_matrix=self.Y[:, start:end],
                                        library_size=self.lY,
                                        basis=self.basis[:, start:end],
                                        gene_index=torch.arange(start, end).to(self.device),
                                        loss_mode='DECONV',
                                        )
                    loss_sum += loss.item()
                    decon_loss += self.net.decon_loss.item()
                    if use_amp:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    #loss.backward()
                
                torch.cuda.empty_cache()
                if use_amp:
                    with autocast():
                        feature_loss = self.net(node_feats=self.X,
                                        edge_index=self.edge_index,
                                        count_matrix=None,
                                        library_size=None,
                                        basis=None,
                                        gene_index=None,
                                        loss_mode='RECON'
                                        )
                else:
                    feature_loss = self.net(node_feats=self.X,
                                        edge_index=self.edge_index,
                                        count_matrix=None,
                                        library_size=None,
                                        basis=None,
                                        gene_index=None,
                                        loss_mode='RECON'
                                        )
                loss_sum += feature_loss.item()
                if test:
                    print(torch.cuda.max_memory_allocated()/1024/1024)
                if use_amp:
                    scaler.scale(feature_loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    feature_loss.backward()
                    self.optimizer.step()
                #self.optimizer.zero_grad()
            else:
                loss_sum = self.net(node_feats=self.X,
                                edge_index=self.edge_index,
                                count_matrix=self.Y,
                                library_size=self.lY,
                                basis=self.basis,
                                )
                self.optimizer.zero_grad()
                loss_sum.backward()
                self.optimizer.step()
                decon_loss = self.net.decon_loss.item()

            if report_loss:
                if not step % step_interval:
                    print("Step: %s, Loss: %.4f, d_loss: %.4f, f_loss: %.4f" % (step, loss_sum, decon_loss, self.net.features_loss.item()))  
    
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
