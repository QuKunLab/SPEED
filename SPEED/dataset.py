import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class dataset_row(Dataset):
    """
     A PyTorch Dataset for iterating over row indices of a matrix.
    """
    def __init__(self, mtx):
        self.row_num = mtx
    def __getitem__(self, index):
        return index
    def __len__(self):
        return self.row_num

class dataset_col(Dataset):
    """
    A PyTorch Dataset for iterating over column indices of a matrix.
    """
    def __init__(self, mtx):
        self.col_num = len(mtx.T)
    def __getitem__(self, index):
        return index
    def __len__(self):
        return self.col_num