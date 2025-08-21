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
    - min_cells_ratio (float, optional): Minimum ratio of cells expressed required for 
         a gene to pass filtering.

    Returns:
         An ``AnnData`` of Processed spatial epigenomics data and an ``AnnData`` of 
         Processed single-cell epigenomics data aligned with the spatial dataset.
    """
    varnames = np.intersect1d(adata.var_names,adata_sc.var_names)
    varnames = list(varnames)
    adata = adata[:,varnames].copy()
    sc.pp.filter_genes(adata,min_cells=adata.shape[0]*min_cells_ratio)
    return adata, adata_sc[:,adata.var_names]


def find_pairwise_dis_matrix_sparse_cKDTree(coordinates,k,up_num=10000):
    """
    Generates spatial positional encoding based on the coordinates of each cell/spot.
    
    First computing the Euclidean distances between all cells/spots to form a pairwise 
    distance matrix. The matrix is then normalized by dividing each entry by the mean 
    distance between every pair of cells/spots, yielding a pairwise degree matrix. 
    Finally, entries greater than a given threshold ``k`` are set to zero to produce 
    the spatial positional encoding matrix.
    
    To avoid excessively high encoding dimensionality when the cells/spots count is large,
    if the number of cells/spots exceeds ``up_num``, randomly select ``up_num`` 
    cells/spots as anchor points and compute the pairwise degree matrix between all 
    cells/spots and these anchor points as the positional encoding.
    
    Parameters:
    - coordinates (np.ndarray): Spatial coordinates of the cells/spots.
    - k (int): Neighbor degree threshold used for spatial positional encoding of each cell/spot.
    - up_num (int): Maximum encoding dimensionality for the spatial positional encoding.

    Returns:
         Sparse spatial positional encoding matrix.
    """
    from sklearn.neighbors import radius_neighbors_graph
    from scipy.spatial import cKDTree

    tree = cKDTree(coordinates)
    distances, _ = tree.query(coordinates, k=2)
    nearest_distances = distances[:, 1] 
    cut = np.median(nearest_distances)
    
    sparse_dist_matrix = radius_neighbors_graph(coordinates, radius=cut*k+0.1, mode='distance', include_self=False, n_jobs=-1)
    sparse_dist_matrix = sparse_dist_matrix/cut
    if len(coordinates)>up_num:
        print(f'use {up_num} spots for coordinate')
        import random
        random.seed(1)
        idx = random.sample(range(0, len(coordinates)), up_num)
        sparse_dist_matrix = sparse_dist_matrix[:,idx]
    return sparse_dist_matrix.astype(np.float32)


def torch_slice(data,idx,mode=None):
    """
    Slice a tensor, array, or sparse matrix by specified row indices, and automatically 
    convert the result into a PyTorch tensor (supporting both dense and sparse formats).

    Parameters:
    - data (array-like, scipy.sparse matrix, or torch.Tensor): The input 2D data matrix. 
    - idx (array-like or None): Row indices to select from data. If None, no slicing is performed.
    - mode ({'array', 'dense', None}, optional)
        - 'array': Convert the result to a dense NumPy array (via .toarray()).
        - 'dense': Convert a PyTorch sparse tensor to a dense tensor (via .to_dense()).
        - None: Keep the current format before converting to torch.
    
    Returns:
        torch.Tensor or torch.sparse.Tensor. 
    """
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


def Encode_image_resnet(img, spatial, image_model):
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
    spatial = spatial.astype(int) #spatial coordinates to INT
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
    
    from pathlib import Path

    
    if image_model=='Resnet50':
        data_file = Path(__file__).resolve().parent / "resnet50_model.bin"
        img_model = timm.create_model('resnet50',num_classes=0, pretrained=True, pretrained_cfg_overlay={'file':data_file})
    elif image_model=='UNI':
        data_file = Path(__file__).resolve().parent / "uni_model.bin"
        img_model = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True)
        img_model.load_state_dict(torch.load(data_file, map_location='cpu'), strict=True)
    elif image_model=='Prov-Gigapath':
        data_file = Path(__file__).resolve().parent / "Prov-Gigapath_model.bin"
        img_model = timm.create_model("vit_giant_patch14_dinov2.lvd142m", img_size=224, patch_size=16, in_chans=3, embed_dim=1536, depth=40, num_heads=24, mlp_ratio=5.33334, global_pool='token', init_values=0.00001, num_classes=0, dynamic_img_size=False)
        img_model.load_state_dict(torch.load(data_file, map_location='cpu'), strict=True)
    else:
        raise ValueError("'image_model' must be one of 'Resnet50', 'UNI' or 'Prov-Gigapath'.")
    repeat = int(np.floor(224/img.shape[2]))
    pad1 = int(np.floor((224-repeat*img.shape[2])/2))
    pad2 = 224-repeat*img.shape[2]-pad1
    x = np.repeat(np.repeat(img,repeat,axis=2),repeat,axis=3)
    x = np.pad(x, ((0,0), (0,0),(pad1,pad2), (pad1,pad2)), 'constant', constant_values=0)
    print('image encoding...')
    output = []
    for i in tqdm(range(len(x))):
        output.append(img_model(test_tsfm(torch.Tensor(x[i,:,:,:])).reshape(1,3, 224, 224)).detach().numpy())
    output = np.concatenate(output)
    
    return output


def binary_spotpeak(mtx):
    """
    Binarize a denoised matrix.

    For each row and column of the denoised matrix, compute a threshold based on the 
    average value of that row or column. Each element in the matrix is compared against 
    the mean of its corresponding row threshold and column threshold. If the element is 
    greater than this mean, it is binarized to 1; otherwise, it is set to 0.

    Parameters:
    - mtx (np.ndarray): Denoised matrix.

    Returns:
    - np.ndarray: A binarized matrix of the same shape as the input, containing only ``0``s and ``1``s.
    """
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