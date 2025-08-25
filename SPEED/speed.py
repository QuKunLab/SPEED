import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
from .dataset import *
from .model import *
from .utils import *
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import sparse
from scipy.sparse import csr_matrix
import os

class SPEED:
    """SPEED: spatial epigenomic data denoising by DMF (Deep Matrix Factorization)
    Model single-cell epigenomic data in the first stage, and Model spatial epigenomic 
    data in the second stage based on prior information from single-cell data.
    """
    def __init__(
        self, 
        adata, 
        is_spatial=False, 
        image=None, 
        k_degree=5, 
        adata_sc=None,
        peak_name='peak_SPEED',
        image_model='Resnet50'
    ):

        """Initialize SPEED
        Initialize SPEED with single-cell data in stage 1 or 
        initialize SPEED with spatial data and trained single-cell data in stage 2.
        
        Parameters:
        - adata (AnnData): The single-cell data matrix in stage 1 or the spatial epigenomics data matrix in stage 2.
        - is_spatial (bool, optional): Whether to incorporate spatial information. 
            ``False`` indicates stage 1 training, while ``True`` indicates stage 2 training. Default is False.
        - image (array-like, optional): The image data in RGB format in stage 2. If ``None``, image information will 
            not be used. This parameter is used if ``is_spatial=True``.
        - k_degree (int, optional): Degree of spatial neighbor used for spatial relative position encoding in stage 2. 
            Default is 5.This parameter is used if ``is_spatial=True``.
        - adata_sc (AnnData, optional): Reference single-cell epigenomics data in stage 2.
            This parameter is used if ``is_spatial=True``.
        - peak_name (str, optional): Name of the peak embedding variable in `adata_sc`. Default is 'peak_SPEED'. 
            This parameter is used if ``adata_sc`` is not None.
        - image_model (str, optional): Model for image feature extraction as input to SPEED. Must be one of 'Resnet50',
            'UNI' or 'Prov-Gigapath'. You can use ``ResNet50`` directly. To use ``UNI``, download the model parameters from 
            https://huggingface.co/MahmoodLab/UNI . To use ``Prov-Gigapath``, download the model parameters from 
            https://huggingface.co/prov-gigapath/prov-gigapath. This parameter is applied only if ``image`` is not None.
        """
        
        if sparse.isspmatrix_csr(adata.X):
            adata.X = adata.X.tocoo().tocsr()
        else:
            adata.X = csr_matrix(adata.X)
        print('matrix ready...')
        # print('use tfidf matrix...')
        adata.X = adata.X.tocoo().tocsr()
        self.mtx = adata.X
        print('use 0-1 matrix...')
        self.mtx.data = np.ones_like(self.mtx.data)
        # print('use raw matrix...')
        self.cell_features = self.mtx
        print('cell_features ready...')
        self.peak_features = self.mtx.T.tocsr()
        print('peak features ready...')
        self.k_degree = k_degree
        self.is_spatial = is_spatial
        if is_spatial:
            if adata_sc is None:
                print('Without single-cell reference')
                self.peak_embedding = np.zeros((adata.shape[1],32))
                self.is_ref = False
            else:
                self.peak_embedding = np.array(adata_sc[:,adata.var_names].varm[peak_name])
                self.is_ref = True
            self.coo_mtx = find_pairwise_dis_matrix_sparse_cKDTree(adata.obsm['spatial'],self.k_degree,up_num=10000)
            self.coo_mtx[self.coo_mtx!=0] = 1+np.log(self.coo_mtx[self.coo_mtx!=0])
            if image is not None:
                self.img = Encode_image_resnet(image, adata.obsm['spatial'], image_model=image_model)
                self.is_image = True
            else:
                self.img = np.zeros((adata.shape[0],2048))
                self.is_image = False
        else:
            self.k_degree = 0
            self.peak_embedding = None
            
    def setup_data(
        self, 
        num_workers=4, 
        data_type='sparse',
        batch_size_cell=None, 
        batch_size_peak=None,
        split_ratio=[1/6,1/6]
    ):
        """
        Prepare the dataset for training and validation.

        Parameters:
        - num_workers (int, optional): Number of subprocesses for data loading. Default is 4.
        - data_type (str, optional): Input data format used by SPEED. Must be either 'sparse' 
            or 'dense'. SPEED will handle this format internally, so no external action is 
            required from the user. For faster training, 'sparse' is recommended on GPU, while 
            'dense' is recommended on CPU.
        - batch_size_cell (int, optional): Batch size for cell-level features.
        - batch_size_peak (int, optional): Batch size for peak-level features.
        - split_ratio (list, optional): The proportion of the validation set for cell-level and peak-level. Default is [1/6, 1/6].
        """
        if data_type not in ("dense", "sparse"):
            raise ValueError("'data_type' must be either 'dense' or 'sparse'")
        self.dense = data_type=='dense'
        np.random.seed(1)
        split_ratio_cell=split_ratio[0]
        split_ratio_peak=split_ratio[1]
        if batch_size_cell is None:
            batch_size_cell=2**int(np.round(np.log(self.mtx.shape[0]/10)/np.log(2)))
        if batch_size_peak is None:
            batch_size_peak=2**int(np.round(np.log(self.mtx.shape[1]/10)/np.log(2)))
        print(f'batch_size_cell = {batch_size_cell}, batch_size_peak = {batch_size_peak}')
        idx_row = np.arange(self.mtx.shape[0])
        idx_col = np.arange(self.mtx.shape[1])
        np.random.shuffle(idx_row)
        np.random.shuffle(idx_col)
        train_idx_row = idx_row[int(split_ratio_cell*len(idx_row)):]
        train_idx_col = idx_col[int(split_ratio_peak*len(idx_col)):]
        val_idx_row = idx_row[:int(split_ratio_cell*len(idx_row))]
        val_idx_col = idx_col[:int(split_ratio_peak*len(idx_col))]
        self.train_idx_row = train_idx_row
        self.train_idx_col = train_idx_col
        self.val_idx_row = val_idx_row
        self.val_idx_col = val_idx_col
        print('split ready...')
        self.label_train = self.mtx[train_idx_row,:][:,train_idx_col]
        self.label_val = self.mtx[val_idx_row,:][:,val_idx_col]
        print('labels ready...')
        if self.peak_embedding is not None:
            print('peak embedding is given')
            self.peak_embedding_train = self.peak_embedding[train_idx_col,:]
            self.peak_embedding_val = self.peak_embedding[val_idx_col,:]
        
        if self.dense:
            self.peak_features = self.peak_features.toarray()
            self.cell_features = self.cell_features.toarray()
        self.peak_train = self.peak_features[train_idx_col,:]
        self.cell_train = self.cell_features[train_idx_row,:]
        self.peak_val = self.peak_features[val_idx_col,:]
        self.cell_val = self.cell_features[val_idx_row,:]
        if self.is_spatial is True:
            self.img_train = self.img[train_idx_row,:]
            self.img_val = self.img[val_idx_row,:]
            self.coo_train = self.coo_mtx[train_idx_row,:]
            self.coo_val = self.coo_mtx[val_idx_row,:]
        
        valset_row = dataset_row(self.cell_val.shape[0])
        self.valset_row = DataLoader(valset_row, batch_size=batch_size_cell,shuffle=True,num_workers=num_workers,pin_memory=True,prefetch_factor=2) 
        valset_col = dataset_row(self.peak_val.shape[0])
        self.valset_col = DataLoader(valset_col, batch_size=batch_size_peak,shuffle=True,num_workers=num_workers,pin_memory=True,prefetch_factor=2) 

        trainset_row = dataset_row(self.cell_train.shape[0])
        self.trainset_row = DataLoader(trainset_row, batch_size=batch_size_cell,shuffle=True,num_workers=num_workers,pin_memory=True,prefetch_factor=2) 
        trainset_col = dataset_row(self.peak_train.shape[0])
        self.trainset_col = DataLoader(trainset_col, batch_size=batch_size_peak,shuffle=True,num_workers=num_workers,pin_memory=True,prefetch_factor=2) 
        
        self.batch_size_cell=batch_size_cell
        self.batch_size_peak=batch_size_peak
        print('dataset ready...')
        
    
    def build_model(
        self, 
        emb_features=32, 
        dropout_p=0.4,
        nn_width_coef=1.0
    ):
        """
        Build the neural network model for SPEED.

        Parameters:
        - emb_features (int, optional): Number of embedding features. Default is 32.
        - dropout_p (float, optional): Dropout probability. Default is 0.4.
        - nn_width_coef (float, optional): Coefficient for the hidden layer width. The final network width 
            equals the default width multiplied by ``nn_width_coef``. Default is 1. When the number of cells 
            in the dataset exceeds 1 million, it can be set to 0.1 to reduce memory and GPU usage.
        """
        
        h_f = [int(np.sqrt((self.mtx.shape[1]+self.mtx.shape[0])/2)*10/4),int(np.sqrt((self.mtx.shape[1]+self.mtx.shape[0])/2)*10/10)]
        h_peak_f = h_f
        h_cell_f = h_f
        peak_batch_norm=False
        peak_layer_norm=True
        cell_batch_norm=False
        cell_layer_norm=True
        cell_depth_mode=1
        eps=1e-8
        peak_training=True
        weight1 = float(self.mtx.shape[0]*self.mtx.shape[1]/self.mtx.nnz/2)
        weight0 = float(self.mtx.shape[0]*self.mtx.shape[1]/(self.mtx.shape[0]*self.mtx.shape[1]-self.mtx.nnz)/2)
        self.pt = peak_training
        
        self.model = SPEED_model(
            self.peak_features.shape[1], 
            self.cell_features.shape[1],
            h_peak_features=h_peak_f,
            h_cell_features=h_cell_f, 
            dim_img_features=self.img.shape[1], 
            emb_features=emb_features,
            peak_batch_norm=peak_batch_norm, 
            peak_layer_norm=peak_layer_norm, 
            cell_batch_norm=cell_batch_norm, 
            cell_layer_norm=cell_layer_norm,
            dropout_p=[dropout_p,dropout_p],
            eps=eps,
            weight1=weight1,
            weight0=weight0,
            cell_depth_mode=cell_depth_mode,
            is_spatial=self.is_spatial,
            peak_train=peak_training
        )
    
    def train(
        self,
        lr=1e-5,
        alpha=10,
        beta=1,
        epoch_num = 500,
        epo_max = 30,
        val_epoch = 1,
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        weight_decay=1e-3,
        save=False,
        save_path='model'
    ):
        """
        Train the SPEED model.

        Parameters:
        - lr (float, optional): Learning rate. Default is 1e-5.
        - alpha (float, optional): Weight of constraint on the similarity between the peak embedding of spatial data 
            and single-cell data in stage 2. Default is 10. (Ignored during stage 1 training)
        - beta (float, optional): The importance of image information for spot embedding in stage 2. Default is 1. 
            (Ignored during stage 1 training)
        - epoch_num (int, optional): Total number of epochs. Default is 500.
        - epo_max (int, optional): Early stopping patience. If no improvement is observed on the validation set 
            within `epo_max` epochs, training stops. Default is 30.
        - val_epoch (int, optional): Validation interval (epochs). Default is 1.
        - device (torch.device, optional): Device to train on ('cuda' or 'cpu').
        - weight_decay (float, optional): Weight decay for optimizer. Default is 1e-3.
        - save (bool, optional): Whether to save the model during training. Default is False.
        - save_path (str, optional): Path to save the model if ``save=True``. (Ignored when ``save=False``)
        """
        
        if self.is_spatial:
            print('Use spatial information...')
        if save:
            if os.path.exists(save_path) is False:
                os.makedirs(save_path)
                print(f'Model will be saved at {save_path}')
        if self.is_image is False:
            beta = 0
        if self.is_ref is False:
            alpha = 0
        if save:
            os.makedirs(save_path, exist_ok=True)
        cell_num = self.cell_features.shape[0]
        reduction='mean'
        model = self.model.to(device)
        torch.manual_seed(1)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_record = {'loss':[], 'emb_loss': [], 'img_loss': [],'val_loss': [], 'val_emb_loss': [],'val_img_loss': []}
        epoch_num = epoch_num
        epo_max = epo_max
        epo_conv = 0
        loss_before = np.Inf
        print('Starting training...')
        print('trainset: ',self.cell_train.shape,self.peak_train.shape)
        for epoch in range(1,epoch_num):
            loss_total = 0
            emb_loss_total = 0
            img_loss_total = 0
            model.train()
            for row in tqdm(self.trainset_row):
                if len(row) < 2:
                    continue
                y = torch_slice(self.cell_train,row)
                y = y.to(device)
                if self.is_spatial:
                    img = torch.Tensor(self.img_train[row,:]).to(device)
                    coo = torch_slice(self.coo_train,row).to(device)
                else:
                    img, coo = None, None
                if self.dense is False:
                    y = y.to_sparse_coo()
                for col in self.trainset_col:
                    if len(col) < 2:
                        continue
                    if self.pt:
                        x = torch_slice(self.peak_train,col)
                        x = x.to(device)
                    else:
                        x = torch.Tensor(self.peak_embedding_train[col,:]).to(device)
                    label=torch_slice(self.label_train[row,:][:,col],None,'array')
                    label=label.to(device)
                    if self.dense is False:
                        x = x.to_sparse_coo()
                        label=label.to_dense()
                    x_re = model(x,y,img,coo)
                    if self.is_spatial is False:
                        loss = model.loss(label, x_re,peak=None,reduction=reduction,alpha=alpha,beta=beta)
                    else:
                        p = torch.Tensor(self.peak_embedding_train[col,:])
                        p = p.to(device)
                        loss, emb_loss, img_loss = model.loss(label, x_re,peak=p,reduction=reduction,alpha=alpha,beta=beta)
                        loss = loss + emb_loss + img_loss
                        emb_loss_total += emb_loss.item()
                        img_loss_total += img_loss.item()
                    loss_total += loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            loss_record['loss'].append(loss_total/len(self.trainset_row)/len(self.trainset_col))
            loss_record['emb_loss'].append(emb_loss_total/len(self.trainset_row)/len(self.trainset_col))
            loss_record['img_loss'].append(img_loss_total/len(self.trainset_row)/len(self.trainset_col))

            if epoch % val_epoch == 0:
                model.eval()
                with torch.no_grad():
                    val_loss_total = 0
                    val_emb_loss_total = 0
                    val_img_loss_total = 0
                    for row in tqdm(self.valset_row):
                        if len(row) < 2:
                            continue
                        y = torch_slice(self.cell_val,row)
                        y = y.to(device)
                        if self.is_spatial:
                            img = torch.Tensor(self.img_val[row,:]).to(device)
                            coo = torch_slice(self.coo_val,row).to(device)
                        else:
                            img, coo = None, None
                        if self.dense is False:
                            y = y.to_sparse_coo()
                        for col in self.valset_col:
                            if len(col) < 2:
                                continue
                            if self.pt:
                                x = torch_slice(self.peak_val,col)
                                x = x.to(device)
                            else:
                                x = torch.Tensor(self.peak_embedding_val[col,:]).to(device)
                            label=torch_slice(self.label_val[row,:][:,col],None,'array')
                            label=label.to(device)
                            if self.dense is False:
                                x = x.to_sparse_coo()
                                label=label.to_dense()
                            x_re = model(x,y,img,coo)
                            if self.is_spatial is False:
                                loss = model.loss(label, x_re,peak=None,reduction=reduction,alpha=alpha,beta=beta)
                            else:
                                p = torch.Tensor(self.peak_embedding_val[col,:])
                                p = p.to(device)
                                loss, emb_loss, img_loss = model.loss(label, x_re,peak=p,reduction=reduction,alpha=alpha,beta=beta)
                                loss = loss + emb_loss + img_loss
                                val_emb_loss_total += emb_loss.sum().item()
                                val_img_loss_total += img_loss.sum().item()
                            val_loss_total += loss.sum().item()

                    if val_loss_total > loss_before:
                        epo_conv += 1 
                        # if save:
                        #     torch.save(model.state_dict(), f'{save_path}/SPEED_model.pth')
                    else:
                        if save:
                            save_path_file = f'{save_path}/SPEED_model.pth'
                            torch.save(model.state_dict(), save_path_file)
                            if cell_num < 1000:
                                self.model = deepcopy(model).cpu()
                            loss_before = val_loss_total
                            epo_conv = 0
                        else:
                            self.model = deepcopy(model).cpu()
                            loss_before = val_loss_total
                            epo_conv = 0
                loss_record['val_loss'].append(val_loss_total/len(self.valset_row)/len(self.valset_col))
                loss_record['val_emb_loss'].append(val_emb_loss_total/len(self.valset_row)/len(self.valset_col))
                loss_record['val_img_loss'].append(val_img_loss_total/len(self.valset_row)/len(self.valset_col))
                        
                print ("Epoch[{}/{}], Loss: {:.5f}, Val Loss: {:.5f}, " 
                       .format(epoch, epoch_num, loss_record['loss'][-1], loss_record['val_loss'][-1]))
                if epo_conv == epo_max:
                    print('convinient')
                    self.last_model=model.cpu()
                    break
                self.loss_record = loss_record
            if save and cell_num >= 1000:
                self.model.load_state_dict(torch.load(save_path_file))

    def get_img_embedding(self, device='cpu', return_np=True):
        """
        Get image embeddings from the trained model.
        """
        model = deepcopy(self.model).to(device)
        model.eval()
        with torch.no_grad():
            emb = model.ImgEncoder(torch.Tensor(self.img).to(device))
            self.img_embedding = emb.cpu().detach().numpy()
        if return_np: return self.img_embedding
    
    def get_coo_embedding(self, device='cpu', split=5, return_np=True):
        """
        Get coordinate-based embeddings from the trained model.
        """
        model = deepcopy(self.model).to(device)
        model.eval()
        n=self.cell_features.shape[0]//split +1
        with torch.no_grad():
            coo_emb = []
            for i in tqdm(range(split)):
                # print(self.coo_mtx[(i*n):((i+1)*n),:].shape)
                coo_emb.append(model.CooEncoder(torch_slice(self.coo_mtx[(i*n):((i+1)*n),:],None).to(device)).cpu().detach().numpy())
            coo_emb = np.concatenate(coo_emb)
            self.coo_embedding = coo_emb
            print('the shape of embedding:', coo_emb.shape)
        if return_np: return coo_emb
            
    def get_cell_depth_emb(self, device='cpu', return_np=True):
        """
        Get cell depth embeddings from the trained model.
        """
        model = deepcopy(self.model).to(device)
        model.eval()
        with torch.no_grad():
            cell_emb = model.CellEncoder(torch.Tensor(self.cell_features).to(device)).cpu().detach().numpy()
            if return_np: return cell_emb[:,-1].flatten()
    
    def get_peak_embedding(self, device='cpu', split=5, return_np=True):
        """
        Get peak embeddings from the trained model.
        """
        model = deepcopy(self.model).to(device)
        n=self.peak_features.shape[0]//split +1
        model.eval()
        with torch.no_grad():
            peak_emb = []
            for i in tqdm(range(split)):
                peak_emb.append(model.peakEncoder(torch_slice(self.peak_features[(i*n):((i+1)*n),:],None).to(device)).cpu().detach().numpy())
            peak_emb = np.concatenate(peak_emb)
            self.peak_embedding = peak_emb
        if return_np: return peak_emb
    
    def get_cell_embedding(self, device='cpu', split=5, return_np=True):
        """
        Get cell/signal embeddings from the trained model.
        """
        model = deepcopy(self.model).to(device)
        model.eval()
        n=self.cell_features.shape[0]//split +1
        with torch.no_grad():
            cell_emb = []
            for i in tqdm(range(split)):
                cell_emb.append(model.CellEncoder(torch_slice(self.cell_features[(i*n):((i+1)*n),:],None).to(device)).cpu().detach().numpy())
            cell_emb = np.concatenate(cell_emb)
            self.cell_embedding = cell_emb
        if return_np: return cell_emb
    
    def get_embedding(self, adata,device='cpu'):
        """Get embeddings
        Compute and store embeddings for cells/spots and peaks.
        The spot/cell embeddings will be stored in ``adata.obsm['X_SPEED']``
        The peak embeddings will be stored in ``adata.varm['peak_SPEED']``

        Parameters:
        - adata (AnnData): The single-cell data in stage 1 or the spatial epigenomics data in stage 2.
        - device (str, optional): Device for computation ('cpu' or 'cuda'). Default is 'cpu'.

        Returns:
            An updated ``AnnData`` object with computed embeddings.

        """
        print('get cell/spot embedding...')
        self.get_cell_embedding(device=device)
        print('get peak embedding...')
        adata.varm['peak_SPEED'] = self.get_peak_embedding(device=device)
        if self.is_spatial is True:
            print('get spatial embedding...')
            self.get_img_embedding()
            self.get_coo_embedding()
            adata.obsm['X_SPEED'] = self.cell_embedding[:,:-1]+self.coo_embedding+self.img_embedding
            adata.obs['SPEED_depth'] = self.cell_embedding[:,-1].flatten()
        else:
            adata.obsm['X_SPEED'] = self.cell_embedding[:,:-1]
            adata.obs['SPEED_depth'] = self.cell_embedding[:,-1].flatten()
        return adata

    def get_denoise_result(self, device='cpu'):
        """
        Generate the denoised matrix

        Parameters:
        - device (str, optional): Device for computation ('cpu' or 'cuda'). Default is 'cpu'.

        Returns:
            An ``array`` of denoised matrix.
        """
        try:
            self.cell_embedding,self.peak_embedding
        except NameError:
            raise NameError("Please run 'get_embedding' first.")
        weight_cell=torch.Tensor(self.cell_embedding).to(device)
        weight_peak=torch.Tensor(self.peak_embedding).to(device)
        depth_cell = weight_cell[:,-1].reshape(-1,1)
        weight_cell = weight_cell[:,:-1]
        if self.is_spatial is True:
            weight_img=torch.Tensor(self.img_embedding).to(device)
            weight_coo=torch.Tensor(self.coo_embedding).to(device)
            pred_i = torch.mm(weight_cell+weight_img+weight_coo,weight_peak.T)+depth_cell
        else:
            pred_i = torch.mm(weight_cell,weight_peak.T)+depth_cell
        pred_i = torch.sigmoid(pred_i)
        return pred_i.cpu().detach().numpy()
    
    def norm_seq_depth(self, adata, device='cpu'):
        """
        Normalize sequencing depth in denoised matrix.

        Parameters:
        - adata (AnnData): The spatial epigenomics data.
        - device (str, optional): Device for computation ('cpu' or 'cuda'). Default is 'cpu'.

        Returns:
            An ``array`` of denoised matrix after normalizing sequencing depth.

        """
        try:
            adata.obsm['X_SPEED'],adata.obs['SPEED_depth'],adata.varm['peak_SPEED']
        except NameError:
            raise NameError("'X_SPEED' or 'SPEED_depth 'does not exist. Please run 'get_embedding' first.")
        data = torch.sigmoid(torch.mm(torch.Tensor(adata.obsm['X_SPEED']).to(device),torch.Tensor(adata.varm['peak_SPEED']).T).to(device)).numpy()
        return data
    
    def binarize(self, adata, device='cpu'):
        """
        Binarize the data after denoising. The binarized results will be stored in ``adata.obsm['binary_SPEED']``

        Parameters:
        - adata (AnnData): The spatial epigenomics data.
        - device (str, optional): Device for computation ('cpu' or 'cuda'). Default is 'cpu'.

        Returns:
            An updated ``AnnData`` object with binarized results.
        """
        try:
            adata.obsm['X_SPEED'],adata.obs['SPEED_depth'],adata.varm['peak_SPEED']
        except NameError:
            raise NameError("'X_SPEED' or 'SPEED_depth 'does not exist. Please run 'get_embedding' first.")
        data = self.norm_seq_depth(adata,device=device)
        data_binary = binary_spotpeak(data)
        adata.obsm['binary_SPEED'] = data_binary
        return adata