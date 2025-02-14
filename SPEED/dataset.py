import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class dataset_Peak(Dataset):
    def __init__(self, mtx, split_ratio=0.1):
        self.mtx = torch.Tensor(mtx.T)
        
    def __getitem__(self, index):
        return self.mtx[index,:]
        
    def __len__(self):
        return len(self.mtx)

def coo_features_generate(spatial, dim=200):
    coo = (spatial - spatial.min(0))
    coo = coo / coo.max(0) * np.pi * 2
    coo_features = []
    for k in range(spatial.shape[0]):
        coox = [np.sin(2**i*np.pi*coo[k,0]) for i in range(dim)]
        cooy = [np.sin(2**i*np.pi*coo[k,1]) for i in range(dim)]
        coo_features.append(coox+cooy)
    coo_features = np.array(coo_features)
    return coo_features

class dataset_row(Dataset):
    def __init__(self, img_features):
        self.row_num = img_features
    def __getitem__(self, index):
        return index
    def __len__(self):
        return self.row_num

class dataset_col(Dataset):
    def __init__(self, mtx):
        self.col_num = len(mtx.T)
    def __getitem__(self, index):
        return index
    def __len__(self):
        return self.col_num