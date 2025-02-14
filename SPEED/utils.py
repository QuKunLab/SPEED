import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import sparse
from scipy.sparse import csr_matrix
from torchvision import transforms
import timm
import scanpy as sc

def process(adata,adata_sc,min_cells_ratio=0):
    """
    Process spatial and single-cell epigenomics data by selecting common features
    and filtering lowly expressed genes.

    Parameters:
    - adata (AnnData): The spatial epigenomics data matrix.
    - adata_sc (AnnData): The reference single-cell epigenomics data matrix.
    - min_cells_ratio (float, optional): Minimum ratio of cells expressed required for a gene to pass filtering.

    Returns:
         An ``AnnData`` of Processed spatial epigenomics data and an ``AnnData`` of Processed single-cell epigenomics data aligned with the spatial dataset.
    """
    varnames = set(adata.var_names) & set(adata_sc.var_names)
    varnames = list(varnames)
    adata = adata[:,varnames].copy()
    sc.pp.filter_genes(adata,min_cells=adata.shape[0]*min_cells_ratio)
    return adata, adata_sc[:,adata.var_names]

def find_pairwise_dis_matrix(coordinates,k):
    distance_matrix = pairwise_distances(coordinates, metric='euclidean')
    
    median_dis = np.median(np.sort(distance_matrix, axis=1)[:, 1])
    dis_matrix = distance_matrix/median_dis
    dis_matrix[dis_matrix>k] = 0
    return dis_matrix

def combined(data, time):
    row_comb = list(np.arange(int(np.ceil(data.shape[0]/time))))*time
    np.random.shuffle(row_comb)
    row_comb = row_comb[:data.shape[0]]
    data_comb = []
    for i in range(int(np.ceil(data.shape[0]/time))):
        data_comb.append(data[np.array(row_comb)==i,:].toarray().sum(0))
    data_comb=np.array(data_comb).T
    return data_comb

def torch_slice(data,idx,mode=None):
    if idx is not None:
        temp = data[idx,:]
    else:
        temp = data
    if mode == 'array':
        temp = temp.toarray()
    if mode == 'dense':
        temp = temp.to_dense()
    if sparse.isspmatrix_csr(temp):
        return torch.sparse_csr_tensor(temp.indptr,temp.indices, temp.data, size=temp.shape)
    else:
        return torch.Tensor(temp)


def Encode_image_resnet(img, spatial):
    """
    Encode spatial transcriptomics images using a pre-trained ResNet model.
    
    Parameters:
    - img (np.ndarray): The input image data in RGB format.
    - spatial (np.ndarray): The spatial coordinates of the spots.
    
    Returns:
        An array of the extracted image features.
    """
    print('segment the image into spots...')
    x = np.array(spatial[:,0], dtype=int)
    y = np.array(spatial[:,1], dtype=int)
    n_bin=np.min([int(np.min(np.diff(np.unique(y))/2)),int(np.min(np.diff(np.unique(x)))/2)])
    img_information = []
    for i in range(len(spatial)):
        y, x = spatial[i,:]
        if x-n_bin <0:
            x=n_bin
        if y-n_bin<0:
            y=n_bin
        if x+n_bin+1>img.shape[0]:
            x=img.shape[0]-n_bin-1
        if y+n_bin+1>img.shape[1]:
            y=img.shape[1]-n_bin-1
        img_partial=img[(x-n_bin):(x+n_bin+1), (y-n_bin):(y+n_bin+1),:]
        img_partial=np.array([img_partial[:,:,0],img_partial[:,:,1],img_partial[:,:,2]])

        img_information.append(img_partial)

    img = np.array(img_information)

    test_tsfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_model = timm.create_model('resnet50',num_classes=0, pretrained=True, pretrained_cfg_overlay={'file':'pytorch_model.bin'})

    repeat = int(np.floor(224/img.shape[2]))
    pad1 = int(np.floor((224-repeat*img.shape[2])/2))
    pad2 = 224-repeat*img.shape[2]-pad1
    x = np.repeat(np.repeat(img,repeat,axis=2),repeat,axis=3)
    x = np.pad(x, ((0,0), (0,0),(pad1,pad2), (pad1,pad2)), 'constant', constant_values=0)
    print('image encoding...')
    x_processed = []
    for i in tqdm(range(len(x))):
        x_processed.append(test_tsfm(torch.Tensor(x[i,:,:,:])).reshape(1,3, 224, 224))
    x_processed = torch.cat(x_processed)
    output = img_model(x_processed).detach().numpy()
    return output

def binary_spotpeak(mtx):
    cut_spot = []
    for i in tqdm(range(mtx.shape[0])):
        data = mtx[i,:]
        cut = np.percentile(data, np.max([0,100-(data.mean())*100]))
        cut_spot.append(cut)
    cut_peak = []
    for i in tqdm(range(mtx.shape[1])):
        data = mtx[:,i]
        cut = np.percentile(data, np.max([0,100-(data.mean())*100]))
        cut_peak.append(cut) 
    cut_spot = np.repeat(np.array(cut_spot).reshape(-1,1),mtx.shape[1],axis=1)
    cut_peak = np.repeat(np.array(cut_peak).reshape(-1,1),mtx.shape[0],axis=1).T
    cut = (cut_peak+cut_spot)/2
    mtx = (mtx>cut).astype(int)
    return mtx