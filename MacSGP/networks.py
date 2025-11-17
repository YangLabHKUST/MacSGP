import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.nn.conv import GCN2Conv

class DenseLayer(nn.Module):

    def __init__(self, 
                 nfeat, # dimensionality of input features
                 nout, # dimensionality of output features
                 zero_init=False, # initialize weights as zeros; use Xavier uniform init if zero_init=False
                 ):

        super().__init__()

        self.linear = nn.Linear(nfeat, nout)

        # Initialization
        if zero_init:
            nn.init.zeros_(self.linear.weight.data)
        else:
            nn.init.uniform_(self.linear.weight.data, -np.sqrt(6 / (nfeat + nout)), np.sqrt(6 / (nfeat + nout)))
        nn.init.zeros_(self.linear.bias.data)

    def forward(self, 
            node_feats, # input node features
            ):

        node_feats = self.linear(node_feats)

        return node_feats
    

class GCNII(nn.Module):
    def __init__(self, 
                 nfeat,  # dimensionality of input features
                 nhidden, # dimensionality of hidden features
                 nlayers, # number of GCNII layers
                 alpha=None, # initial residual connection weight
                 theta=None # scaling parameter 
                 ):
        super(GCNII, self).__init__()
        
        self.nlayers = nlayers
        if alpha is None:
            self.alpha = [0.1 for _ in range(nlayers)]
        else:
            self.alpha = alpha
        if theta is None:
            self.theta = [1.0 for _ in range(nlayers)]
        else:
            self.theta = theta
        
        self.fcs = nn.ModuleList()
        self.fcs.append(
            nn.Linear(nfeat, nhidden)
        )

        self.convs = nn.ModuleList()
        for layer in range(nlayers):
            self.convs.append(
                GCN2Conv(nhidden, self.alpha[layer], self.theta[layer], layer+1,
                        add_self_loops=False, normalize=True)
            )
        
        self.act_fn = nn.ELU()

    def forward(self, node_feats, edge_index):
        _layers = []
        node_feats = self.act_fn(self.fcs[0](node_feats))
        _layers.append(node_feats)

        for i,con in enumerate(self.convs):
            node_feats = con(node_feats, _layers[0], edge_index)
            node_feats = self.act_fn(node_feats)
        
        return node_feats
    

class SGPNet(nn.Module):
    
    def __init__(self, 
                hidden_dims, # dimensionality of hidden layers
                n_layers, # number of GCNII layers
                n_celltypes, # number of cell types
                n_SGPs, # number of SGPs for each cell type
                alpha_gcn, # initial residual connection weight for GCNII
                theta_gcn, # scaling parameter for GCNII
                estimate_gamma, # whether to estimate platform effects
                init_gamma, # initial value for platform effects
                estimate_gamma_k, # whether to estimate cell-type-specific unwanted effects
                estimate_alpha, # whether to estimate spot effects
                coef_fe, 
                coef_reg, 
                ):
        
        super().__init__()
        # define layers
        # encoder layers
        self.encoder_layer1 = GCNII(hidden_dims[0],n_layers,hidden_dims[1],alpha_gcn,theta_gcn)
        self.encoder_layer2 = DenseLayer(hidden_dims[1], hidden_dims[2])

        # decoder layers
        self.decoder_layer1 = GCNII(hidden_dims[2],n_layers,hidden_dims[1],alpha_gcn,theta_gcn)
        self.decoder_layer2 = DenseLayer(hidden_dims[1], hidden_dims[0])

        # SGP layers
        #self.SGP_layer = DenseLayer(hidden_dims[2], n_celltypes, zero_init=True)
        self.factor_layers = nn.ModuleList(
            [DenseLayer(hidden_dims[2], n_SGPs, zero_init=True) for _ in range(n_celltypes)]
        )
        self.loading = nn.Parameter(torch.randn(n_celltypes, n_SGPs, hidden_dims[0]) * 1e-3)

        self.estimate_alpha = estimate_alpha
        if estimate_alpha:
            self.deconv_alpha_layer = DenseLayer(hidden_dims[2], 1, zero_init=True)

        self.estimate_gamma = estimate_gamma
        if estimate_gamma:
            self.gamma = nn.Parameter(init_gamma)
        else:
            self.gamma = init_gamma

        self.estimate_gamma_k = estimate_gamma_k
        if estimate_gamma_k:
            self.gamma_k = nn.Parameter(torch.zeros(n_celltypes, hidden_dims[0]))

        self.coef_fe = coef_fe
        self.coef_reg = coef_reg
        self.n_SGPs = n_SGPs

    def forward(self, 
            node_feats, # input node features
            edge_index, # edge index
            count_matrix, # gene expression counts
            library_size, # library size
            basis, # basis matrix
            alpha, # spot-specific effects
            proportion, # cell type proportions
            loss_mode="FULL", # loss mode: "FULL" / "RECON" / "DECONV"
            n_patchs=None, # number of gene patchs
            gene_patch=False, # whether to use gene patch
            gene_index=None, # patch gene index
            patch_size=100, # size of gene patch
            test=False, # test mode
            ):
        # encoder
        Z = self.encoder(node_feats, edge_index)

        if loss_mode in ['FULL', 'RECON']:
            # decoder
            node_feats_recon = self.decoder(Z, edge_index)

            # reconstruction loss of node features
            self.features_loss = torch.mean(torch.sqrt(torch.sum(torch.pow(node_feats-node_feats_recon, 2), axis=1)))

        if loss_mode in ['FULL', 'DECONV']:
            if n_patchs is None:
                n_patchs = 1
            
            # factorization 
            #factor = self.deconv_factor_layer(F.elu(Z))
            factors = [layer(F.elu(Z)) for layer in self.deconv_factor_layers]
            factors = torch.stack(factors)

            if self.estimate_alpha:
                alpha_res = self.deconv_alpha_layer(F.elu(Z))
            else:
                alpha_res = 0
         
            if gene_index is None:
                gene_index = torch.arange(count_matrix.shape[1]).to(count_matrix.device)
            
            # main loss
            if self.estimate_gamma_k:
                basis = basis * torch.exp(self.gamma_k[:, gene_index])
            
            u_exp = torch.exp(torch.matmul(factors, self.loading[:, :, gene_index]))
            u_exp = u_exp * torch.matmul(proportion.T.unsqueeze(2), basis.unsqueeze(1))

            log_lam = torch.log(torch.sum(u_exp, axis=0) + 1e-6) + alpha.unsqueeze(1) + alpha_res + self.gamma.unsqueeze(0)[:, gene_index] 

            self.decon_loss = - torch.mean(torch.sum(count_matrix * 
                                        (torch.log(library_size + 1e-6) + log_lam) - library_size * torch.exp(log_lam), axis=1))
            
            self.regularization_loss = torch.mean(torch.pow(factors, 2)) * proportion.shape[1] / n_patchs+ \
                                torch.sum(torch.pow(self.loading[:, :, gene_index], 2)) / self.gamma.shape[0] / self.n_SGPs

        # total loss
        if loss_mode == 'FULL':
            loss = self.decon_loss + self.coef_fe * self.features_loss + self.coef_reg * self.regularization_loss
        elif loss_mode == 'RECON':
            loss = self.coef_fe * self.features_loss
        elif loss_mode == 'DECONV':
            loss = self.decon_loss + self.coef_reg * self.regularization_loss   

        return loss

    def evaluate(self, node_feats, edge_index):
        # encoder
        Z = self.encoder(node_feats, edge_index)
        
        # factorizer
        #factor = self.deconv_factor_layer(F.elu(Z))
        factors = [layer(F.elu(Z)) for layer in self.deconv_factor_layers]
        factors = torch.stack(factors)
        
        if self.estimate_alpha:
            alpha_res = self.deconv_alpha_layer(F.elu(Z))
        else:
            alpha_res = 0

        return Z, factors, self.loading, self.gamma, alpha_res      
    
    def encoder(self, node_feats, edge_index):
        Z = self.encoder_layer1(node_feats, edge_index)
        Z = self.encoder_layer2(Z)
        return Z        
    
    def decoder(self, Z, edge_index):
        x = self.decoder_layer1(Z, edge_index)
        x = self.decoder_layer2(x)
        return x
    

class DeconvNet(nn.Module):

    def __init__(self, 
                 hidden_dims, # dimensionality of hidden layers
                 n_layers, # number of GCNII layers
                 n_celltypes, # number of cell types
                 alpha_gcn,
                 theta_gcn,
                 coef_fe,
                 ):
        
        super().__init__()
        # define layers
        # encoder layers
        self.encoder_layer1 = GCNII(hidden_dims[0],n_layers,hidden_dims[1],alpha_gcn,theta_gcn)
        self.encoder_layer2 = DenseLayer(hidden_dims[1], hidden_dims[2])

        # decoder layers
        self.decoder_layer1 = GCNII(hidden_dims[2],n_layers,hidden_dims[1],alpha_gcn,theta_gcn)
        self.decoder_layer2 = DenseLayer(hidden_dims[1], hidden_dims[0])
        # deconvolution layers
        self.deconv_alpha_layer = DenseLayer(hidden_dims[2], 1, zero_init=True)
        self.deconv_beta_layer = DenseLayer(hidden_dims[2], n_celltypes, zero_init=True)

        self.gamma = nn.Parameter(torch.zeros(1, hidden_dims[0]))

        self.coef_fe = coef_fe

    def forward(self,
            node_feats, # input node features
            edge_index, # edge index
            count_matrix, # gene expression counts
            library_size, # library size
            basis, # basis matrix
            gene_index=None, # gene index
            loss_mode='FULL', # loss mode: "FULL" / "RECON" / "DECONV"
            ):
        # encoder
        Z = self.encoder(node_feats, edge_index)


        if loss_mode in ['FULL', 'RECON']:
            # decoder
            node_feats_recon = self.decoder(Z, edge_index)

            # reconstruction loss of node features
            self.features_loss = torch.mean(torch.sqrt(torch.sum(torch.pow(node_feats-node_feats_recon, 2), axis=1)))
        
        if loss_mode in ['FULL', 'DECONV']:
            # deconvolutioner
            beta, alpha = self.deconvolutioner(Z)
            if gene_index is None:
                gene_index = torch.arange(count_matrix.shape[1]).to(count_matrix.device)
            # main loss
            log_lam = torch.log(torch.matmul(beta, basis) + 1e-6) + alpha + self.gamma[:, gene_index]
            #lam = torch.exp(log_lam)
            self.decon_loss = - torch.mean(torch.sum(count_matrix * 
                                       (torch.log(library_size + 1e-6) + log_lam) - library_size * torch.exp(log_lam), axis=1))
        
        # Total loss
        if loss_mode == 'FULL':
            loss = self.decon_loss + self.coef_fe * self.features_loss
        elif loss_mode == 'RECON':
            loss = self.coef_fe * self.features_loss
        elif loss_mode == 'DECONV':
            loss = self.decon_loss
        
        return loss
    
    def evaluate(self, node_feats, edge_index):
        # encoder
        Z = self.encoder(node_feats, edge_index)
        
        # deconvolutioner
        beta, alpha = self.deconvolutioner(Z)

        return Z, beta, alpha, self.gamma
    
    def encoder(self, node_feats, edge_index):
        x = self.encoder_layer1(node_feats, edge_index)
        x = self.encoder_layer2(x)
        return x

    def decoder(self, Z, edge_index):
        x = self.decoder_layer1(Z, edge_index)
        x = self.decoder_layer2(x)
        return x   

    def deconvolutioner(self, Z):
        beta = self.deconv_beta_layer(F.elu(Z))
        beta = F.softmax(beta, dim=1)
        H = F.elu(Z)
        alpha = self.deconv_alpha_layer(H)
        return beta, alpha     