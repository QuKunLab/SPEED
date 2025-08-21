import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
from sklearn.metrics import roc_auc_score


class Layer(nn.Module):
    """
    A fully connected layer with optional normalization, activation, and dropout.
    """
    def __init__(
        self,
        in_features, 
        out_features, 
        layer_norm = False,
        batch_norm = True,
        bias = True,
        act_fn = nn.LeakyReLU(negative_slope=0.01),
        dropout = True,
        dropout_p = 0.01):
        
        super(Layer, self).__init__()
        self.model = nn.Sequential()
        self.model.add_module("linear",nn.Linear(in_features=in_features, out_features=out_features, bias=bias))
        if layer_norm:
            self.model.add_module("layernorm",nn.LayerNorm(out_features, eps=1e-5, elementwise_affine=False))
        if batch_norm:
            self.model.add_module("batchnorm",nn.BatchNorm1d(out_features, eps=1e-5))
        self.model.add_module("activation",act_fn)
        if dropout:
            self.model.add_module("dropout",nn.Dropout(p=dropout_p))
        
    def forward(self, x):
        return self.model(x)
    
class Encoder(nn.Module):
    """
    A multi-layer perceptron (MLP) encoder for extracting latent representations 
    from various types of input data.

    This encoder can be applied to cells/spots, peaks/features, spatial position 
    encodings, and image embeddings to obtain latent representations.
    """
    def __init__(self, in_features, h_features, z_features, batch_norm, layer_norm, dropout_p=0.1, dropout = True):
        super(Encoder, self).__init__()
        features = [in_features] + h_features
        self.model = nn.Sequential()
        layer_num=len(h_features)
        for l in range(layer_num):
            self.model.add_module(f"layer {l+1}", Layer(features[l], features[l+1],dropout_p = dropout_p, batch_norm=batch_norm, layer_norm=layer_norm,dropout = dropout))
        self.model.add_module(f"layer {l+2}", nn.Linear(features[l+1], z_features))

    def forward(self, x):
        return self.model(x)
        
class SPEED_model(nn.Module):
    """
    The SPEED model for spatial epigenomics data enhancement using deep matrix factorization.

    Parameters:
    - cell_num (int): Number of cells/spots in the dataset.
    - peak_num (int): Number of peaks/features in the dataset.
    - h_peak_features (list of int): List of hidden layer sizes for the peak encoder.
    - h_cell_features (list of int): List of hidden layer sizes for the cell encoder.
    - dim_img_features (int): The dimension of the image features.
    - emb_features (int): Number of embedding features.
    - peak_batch_norm (bool): Whether to apply Batch Normalization in the peak encoder.
    - peak_layer_norm (bool): Whether to apply Layer Normalization in the peak encoder.
    - cell_batch_norm (bool): Whether to apply Batch Normalization in the cell encoder.
    - cell_layer_norm (bool): Whether to apply Layer Normalization in the cell encoder.
    - dropout_p (list of float, optional): Dropout probabilities for peak and cell encoders. Default is ``[0.4, 0.4]``.
    - eps (float, optional): Small constant to avoid division by zero. Default is ``1e-8``.
    - weight1 (float, optional): Weight for positive samples in weighted-BCE loss. Default is ``None``.
    - weight0 (float, optional): Weight for negative samples in weighted-BCE loss. Default is ``None``.
    - cell_depth_mode (int, optional): Whether to include a cell depth term in the encoding. Default is ``1``.
    - is_spatial (bool, optional): Whether to use spatial information. Default is ``False``.
    - peak_train (bool, optional): Whether to train the peak encoder. Default is ``True``.
    """
    def __init__(self, cell_num,peak_num, h_peak_features, h_cell_features, dim_img_features, emb_features, peak_batch_norm, peak_layer_norm, cell_batch_norm, cell_layer_norm,dropout_p=[0.4,0.4],eps=1e-8,weight1=None,weight0=None,cell_depth_mode=1,is_spatial=False,peak_train=True):
        super(SPEED_model, self).__init__()
        self.CellEncoder = Encoder(in_features=peak_num, h_features=h_cell_features, z_features=emb_features+cell_depth_mode, batch_norm=cell_batch_norm, layer_norm=cell_layer_norm,dropout_p=dropout_p[1])
        if peak_train:
            self.peakEncoder = Encoder(in_features=cell_num, h_features=h_peak_features, z_features=emb_features, batch_norm=peak_batch_norm, layer_norm=peak_layer_norm,dropout_p=dropout_p[0])
        if is_spatial:
            self.ImgEncoder = Encoder(in_features=dim_img_features, h_features=[512,128], z_features=emb_features, batch_norm=cell_batch_norm, layer_norm=cell_layer_norm,dropout_p=dropout_p[1])
            self.CooEncoder = Encoder(in_features=cell_num, h_features=h_cell_features, z_features=emb_features, batch_norm=cell_batch_norm, layer_norm=cell_layer_norm,dropout_p=dropout_p[1])
        self.cell_depth_mode = cell_depth_mode
        self.eps = eps
        self.weight1 = weight1
        self.weight0 = weight0
        self.is_spatial = is_spatial
        self.peak_train = peak_train
    
    def forward(self, X, cell, img=None, coo=None):
        if self.peak_train:
            z = self.peakEncoder(X)
            self.peak_emb = z
            z=z.T
        else:
            self.peak_emb = X
            z = X.T
        weight_cell = self.CellEncoder(cell)
        if self.is_spatial:
            weight_img = self.ImgEncoder(img)
            weight_coo = self.CooEncoder(coo)
        if self.cell_depth_mode:
            depth_cell = weight_cell[:,-1].reshape(-1,1)
            weight_cell = weight_cell[:,:-1]
            if self.is_spatial:
                pred_i = torch.mm(weight_cell+weight_img+weight_coo,z)+depth_cell
                self.emb_coo = weight_coo
                self.emb_img = weight_img
            else:
                pred_i = torch.mm(weight_cell,z)+depth_cell
        else:
            if self.is_spatial:
                pred_i = torch.mm(weight_cell+weight_img+weight_coo,z)
                self.emb_coo = weight_coo
                self.emb_img = weight_img
            else:
                pred_i = torch.mm(weight_cell,z)
        pred_i = torch.sigmoid(pred_i)
        self.emb = weight_cell
        return pred_i
    
    def loss(self, x, x_re, peak,reduction='mean',alpha=10,beta=1):
        l_p_1 = F.cosine_embedding_loss(x_re, x,torch.ones([x.shape[0]]).to(x.device),reduction=reduction)
        l_p_2 = F.cosine_embedding_loss(x_re.T, x.T,torch.ones([x.shape[1]]).to(x.device),reduction=reduction)
        loss_rec = (l_p_1+l_p_2)/2
        if self.is_spatial is False:
            return loss_rec
        loss_emb = alpha*F.mse_loss(self.peak_emb,peak, reduction=reduction)
        loss_img = beta*(1/(self.emb_img**2).mean())
        return loss_rec,loss_emb,loss_img