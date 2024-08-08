import torch
import os
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from sklearn.decomposition import PCA


def get_scheduler(args, optimizer):
    """
    cosine will change learning rate every iteration, others change learning rate every epoch
    :param batches: the number of iterations in each epochs
    :return: scheduler
    """
    if args.max_epoch<=1000: 
        milestones=[150, 300]
    else: 
        milestones=[args.max_epoch*0.5, args.max_epoch*1.0]
        
    SCHEDULERS = {
        'multi_step': optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.2),
        'cosine': optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch),
    }
    return SCHEDULERS[args.scheduler]


def get_feat_pred(model, loader):
    model.eval()
    all_feats, all_preds, all_labels = [], [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for i, batch in enumerate(loader):
            input = batch['input'].to(device)
            target = batch['target'].to(device)
            input, target = input.to(device), target.to(device)
            pred, feat = model(input, ret_feat=True)

            all_feats.append(feat)
            all_preds.append(pred)
            all_labels.append(target)

        all_feats = torch.cat(all_feats)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
    return all_feats, all_preds, all_labels


def compute_cosine_norm(W):
    # Compute cosine similarity between row vectors

    dot_product = torch.matmul(W, W.transpose(1, 0))  # Shape: (3, 3)

    # Compute L2 norm of row vectors
    norm = torch.sqrt(torch.sum(W ** 2, dim=-1, keepdim=True))  # Shape: (3, 1)

    # Compute cosine similarity
    cosine = dot_product / (norm * norm.transpose(1, 0))  # [3, 3]

    return cosine, norm


def gram_schmidt(W):
    dim = W.shape[0]

    U = torch.empty_like(W)
    U[0, :] = W[0, :] / torch.norm(W[0, :], p=2)

    for i in range(1, dim):
        j = i - 1
        ortho_vector = W[i, :]
        while j >= 0:
            proj = torch.dot(U[j, :], W[i, :]) * U[j, :]
            ortho_vector -= proj
            j -= 1
        U[i, :] = ortho_vector / torch.norm(ortho_vector, p=2)

    return U


# ============== NC metrics ==============
def compute_metrics(W, H, y_dim=None):
    result = {}

    if y_dim == None:
        y_dim = W.shape[0]

    # NC1
    H_np = H.cpu().numpy()
    pca_for_H = PCA(n_components=y_dim)
    try:
        pca_for_H.fit(H_np)
    except Exception as e:
        print(e)
        result['nc1'] = -1
    H_pca = pca_for_H.components_[:y_dim, :]  # First two principal components [2,5]

    # NRC1 with Gram-Schmidt
    H_pca = torch.tensor(H_pca, device=H.device)
    H_U = gram_schmidt(H_pca)
    P_H = torch.mm(H_U.T, H_U)   # Projection matrix
    result['nc1'] = torch.sum((H @ P_H - H)**2).item() / len(H)

    H_row_norms = np.linalg.norm(H, axis=1, keepdims=True)
    H_normalized = H / H_row_norms
    result['nc1_n'] = torch.mean( torch.norm( H_normalized @ P_H - H_normalized, p=2, dim=1) ** 2 ).item()
    del H_np, H_pca, pca_for_H

    # Projection error with Gram-Schmidt
    U = gram_schmidt(W)
    P_W = torch.mm(U.T, U)
    result['nc2'] = torch.sum((H @ P_W - H) ** 2).item() / len(H)
    result['nc2_n'] = torch.mean(
        torch.norm(H_normalized @ P_W - H_normalized, p=2, dim=1) ** 2
    ).item()

    inverse_mat = torch.inverse(W @ W.T)
    H_proj_W = (W.T @ inverse_mat @ W @ H.T).T
    result['nc2a'] = torch.sum((H - H_proj_W) ** 2).item() / len(H)

    del H, H_normalized, P_H, P_W

    return result


class Graph_Vars:
    def __init__(self, dim=2):
        self.epoch = []

        self.train_mse = []
        self.train_nc1 = []
        self.train_nc3 = []
        self.train_nc3a = []

        self.val_mse = []
        self.val_nc1 = []
        self.val_nc3 = []
        self.val_nc3a = []

        self.nc2 = []
        self.h_norm=[]

        self.ww00 = []
        if dim==2:
            self.ww01 = []
            self.ww11 = []
            self.w_cos = []

    def load_dt(self, nc_dt, epoch):
        self.epoch.append(epoch)
        for key in nc_dt:
            try:
                self.__getattribute__(key).append(nc_dt[key])
            except:
                print('{} is not attribute of Graph var'.format(key))


class Train_Vars:
    def __init__(self, dim=2):
        self.epoch = []
        self.lr = []

        self.train_mse = []
        self.train_proj_error = []
        self.w_outer_d = []
        self.ww00 = []
        if dim==2:
            self.ww01 = []
            self.ww11 = []

    def load_dt(self, nc_dt, epoch):
        self.epoch.append(epoch)
        for key in nc_dt:
            try:
                self.__getattribute__(key).append(nc_dt[key])
            except:
                print('{} is not attribute of Graph var'.format(key))


def get_theoretical_solution(train_loader, args, bias=None, all_labels=None, center=False):
    if all_labels is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        all_labels = []
        with torch.no_grad():
            for i, batch in enumerate(train_loader):
                target = batch['target'].to(device)
                all_labels.append(target)
            all_labels = torch.cat(all_labels)   # [N, 2]

    if bias is not None:
        center_labels = all_labels - bias  # [N,2] - [2]
    else:
        center_labels = all_labels
    if center:
        mu = torch.mean(all_labels, dim=0)
        center_labels = all_labels - mu
    else:
        mu = torch.mean(all_labels, dim=0)
        center_labels = all_labels
    Sigma = torch.matmul(center_labels.T, center_labels)/len(center_labels)
    Sigma = Sigma.cpu().numpy()

    if args.num_y >= 2:
        eigenvalues, eigenvectors = np.linalg.eig(Sigma)
        sqrt_eigenvalues = np.sqrt(eigenvalues)
        Sigma_sqrt = eigenvectors @ np.diag(sqrt_eigenvalues) @ np.linalg.inv(eigenvectors)
        min_eigval = eigenvalues[0]
        max_eigval = eigenvalues[-1]
    elif args.num_y == 1: 
        Sigma_sqrt = np.sqrt(Sigma)
        min_eigval, max_eigval = Sigma_sqrt, Sigma_sqrt

    W_outer = args.lambda_H * (Sigma_sqrt/np.sqrt(args.lambda_H*args.lambda_W) - np.eye(args.num_y))
    theory_stat = {
            'mu': mu.cpu().numpy(), 
            'Sigma': Sigma, 
            'Sigma_sqrt': Sigma_sqrt,
            'min_eigval': min_eigval,
            'max_eigval': max_eigval,
        }
    return W_outer, mu.cpu().numpy(), Sigma_sqrt, all_labels, theory_stat  # all_labels is still tensor