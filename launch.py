import os
import h5py
import pdb 
import torch
import random
import numpy as np
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
import torch.nn.functional as F
import wandb
import logging
import datetime
import argparse
from torch.profiler import profile, record_function, ProfilerActivity
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from torch.nn.functional import mse_loss
from dataset import ImageTargetDataset, H5Dataset, transform
from model import RegressionResNet
from train_utils import Graph_Vars, get_feat_pred, compute_cosine, gram_schmidt
from utils import print_model_param_nums, set_log_path, log, print_args
        
def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.environ["WANDB_API_KEY"] = "0c0abb4e8b5ce4ee1b1a4ef799edece5f15386ee"
    os.environ["WANDB_MODE"] = "online"  # "dryrun"
    os.environ["WANDB_CACHE_DIR"] = "/scratch/lg154/sseg/.cache/wandb"
    os.environ["WANDB_CONFIG_DIR"] = "/scratch/lg154/sseg/.config/wandb"
    wandb.login(key='0c0abb4e8b5ce4ee1b1a4ef799edece5f15386ee')
    wandb.init(project='lr_' + args.dataset,
               name=args.exp_name.split('/')[-1]
               )
    wandb.config.update(args)

    pdb.set_trace()
    # train_dataset = ImageTargetDataset('/vast/zz4330/Carla_JPG/Train/images', '/vast/zz4330/Carla_JPG/Train/targets', transform=transform)
    # val_dataset = ImageTargetDataset('/vast/zz4330/Carla_JPG/Val/images', '/vast/zz4330/Carla_JPG/Val/targets', transform=transform)
    train_dataset = H5Dataset('/vast/zz4330/Carla_h5/SeqTrain', transform=transform)
    val_dataset = H5Dataset('/vast/zz4330/Carla_h5/SeqVal', transform=transform)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    model = RegressionResNet(pretrained=True, num_outputs=2).to(device)
    _ = print_model_param_nums(model=model)
    if torch.cuda.is_available():
        model = model.cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    criterion = nn.MSELoss()
    if args.ufm:
        optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=args.lr, weight_decay=0)
    else:
        optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=args.lr, weight_decay=args.wd)

    # ================== Training ==================
    nc_tracker = Graph_Vars()
    wandb.watch(model, criterion, log="all", log_freq=10)
    for epoch in range(args.start_epoch, args.max_epoch):
        model.train()
        running_loss = 0.0
        for batch_idx, batch in enumerate(train_data_loader):
            images = batch['image'].to(device)
            targets = batch['target'].to(device)
            optimizer.zero_grad()
            outputs, feats = model(images, ret_feat=True)

            loss = criterion(outputs, targets)
            if args.ufm:
                l2reg_H = args.lambda_H * torch.norm(feats, 2)
                l2reg_W = args.lambda_W * torch.norm(model.fc.weight, 2)
                loss = loss + l2reg_H + l2reg_W

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        log(f"Epoch [{epoch+1}/{args.max_epoch}], Loss: {running_loss/len(train_data_loader)}")

        feats, preds, labels = get_feat_pred(model, train_data_loader)

        train_loss = criterion(preds, labels)

        # cosine between Wi's
        W = model.fc.weight.data
        cosine_W = compute_cosine(W)

        # compute projection error
        U = gram_schmidt(W)
        P_E = torch.mm(U.T, U)  # Projection matrix using orthonormal basis
        h_projected = torch.mm(feats, P_E)
        projection_error_train = mse_loss(h_projected, feats).item()

        # ================== Evaluation ==================
        feats, preds, labels = get_feat_pred(model, val_data_loader)
        val_loss = criterion(preds, labels)

        h_projected = torch.mm(feats, P_E)
        projection_error_val = mse_loss(h_projected, feats).item()

        nc_dt = {
            'epoch': epoch,
            'train_mse': train_loss,
            'val_mse': val_loss,
            'cos_w12': cosine_W[0, 1].item(),
            'train_proj_error': projection_error_train,
            'val_proj_error': projection_error_val
        }
        nc_tracker.load_dt(nc_dt, epoch=epoch)

        wandb.log({'mse/train_mse': train_loss,
                   'mse/val_mse': val_loss,
                   'nc/cos_w12': cosine_W[0, 1].item(),
                   'nc/train_proj_error': projection_error_train,
                   'nc/val_proj_error': projection_error_val},
                  step=epoch)
        log("Epoch [{}/{}], Train_loss: {:.4f}, Val_loss: {:.4f}, Train_proj_error: {:.2f}, Val_proj_error: {:.2f}".format(
            epoch, args.max_epoch, train_loss, val_loss, nc_dt['train_proj_error'], nc_dt['val_proj_error']
        ))

        if epoch % args.save_freq == 0:
            ckpt_path = os.path.join(args.save_dir, 'ep{}_ckpt.pth'.format(epoch))
            ckpt = {'epoch': epoch, 'state_dict': model.state_dict(), 'lr': optimizer.param_groups[0]['lr'] }
            torch.save(ckpt, ckpt_path)

   
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Regression NC")
    parser.add_argument('--dataset', type=str, default='Carla')
    parser.add_argument('--max_epoch', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_y', type = int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lambda_H', type=float, default=1e-5)
    parser.add_argument('--lambda_W', type=float, default=5e-2)
    parser.add_argument('--wd', type=float, default=5e-4)

    parser.add_argument('--ufm', default=False, action='store_true')

    parser.add_argument('--resume', type = str, default=None)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--save_freq', default=10, type=int)

    parser.add_argument('--exp_name', type=str, default='exp')
    args = parser.parse_args()
    args.save_dir = os.path.join("./result/", args.exp_name)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    set_log_path(args.save_dir)
    log('save log to path {}'.format(args.save_dir))
    log(print_args(args))

    main(args)
