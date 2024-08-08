import os
import pdb
import pickle
import torch
import random
import numpy as np
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import wandb
import logging
import datetime
import argparse
from torch.nn.functional import mse_loss
from dataset import ImageTargetDataset, SubDataset, transform
from model import RegressionResNet
from train_utils import Graph_Vars, get_feat_pred, compute_cosine_norm, gram_schmidt, get_scheduler, Train_Vars
from utils import print_model_param_nums, set_log_path, log, print_args


def train_one_epoch(model, data_loader, optimizer, criterion, args):
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    running_loss = 0.0
    all_feats = []
    for batch_idx, batch in enumerate(data_loader):
        images = batch['image'].to(device, non_blocking=True)
        targets = batch['target'].to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs, feats = model(images, ret_feat=True)
        all_feats.append(feats.data)

        loss = criterion(outputs, targets)
        if args.ufm:
            l2reg_H = torch.sum(feats**2) * args.lambda_H / args.batch_size
            l2reg_W = torch.sum(model.fc.weight ** 2) * args.lambda_W
            loss = loss + l2reg_H + l2reg_W

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_train_loss = running_loss / len(data_loader)

    all_feats = torch.cat(all_feats)
    return all_feats, running_train_loss


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

    train_dataset = SubDataset('/vast/lg154/Carla_JPG/Train/train_list.txt',
                               '/vast/lg154/Carla_JPG/Train/sub_targets.pkl', transform=transform, dim=args.num_y)
    # val_dataset = ImageTargetDataset('/vast/zz4330/Carla_JPG/Val/images', '/vast/zz4330/Carla_JPG/Val/targets', transform=transform, dim=args.num_y)
    
    # train_dataset = ImageTargetDataset('/vast/zz4330/Carla_JPG/Train/images', '/vast/zz4330/Carla_JPG/Train/targets', transform=transform, dim=args.num_y)
    # train_dataset = NumpyDataset('/scratch/zz4330/Carla/Train/images.npy', '/scratch/zz4330/Carla/Train/targets.npy', transform=transform)
    # val_dataset = NumpyDataset('/scratch/zz4330/Carla/Val/images.npy', '/scratch/zz4330/Carla/Val/targets.npy', transform=transform)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    # val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    model = RegressionResNet(pretrained=True, num_outputs=args.num_y, args=args).to(device)
    _ = print_model_param_nums(model=model)
    if torch.cuda.is_available():
        model = model.cuda()
    if model.fc.bias is not None:
        log("classification layer has bias terms.")
    else:
        log("classification layer DO NOT have bias terms.")

    if args.ufm:
        optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=args.lr, weight_decay=0)
    else:
        optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=args.lr, weight_decay=args.wd)
    scheduler = get_scheduler(args, optimizer)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            args.start_epoch = checkpoint['epoch']
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # =================== theoretical solution ================
    all_labels = []
    with torch.no_grad():
        for i, batch in enumerate(train_data_loader):
            target = batch['target'].to(device)
            all_labels.append(target)
        all_labels = torch.cat(all_labels)  # [N, 2]
    Sigma = torch.matmul(all_labels.T, all_labels) / len(all_labels)
    Sigma = Sigma.cpu().numpy()

    eigenvalues, eigenvectors = np.linalg.eig(Sigma)
    sqrt_eigenvalues = np.sqrt(eigenvalues)
    Sigma_sqrt = eigenvectors.dot(np.diag(sqrt_eigenvalues)).dot(np.linalg.inv(eigenvectors))

    W_outer = args.lambda_H * (Sigma_sqrt / np.sqrt(args.lambda_H * args.lambda_W) - np.eye(args.num_y))

    filename = os.path.join(args.save_dir, 'theory.pkl')
    with open(filename, 'wb') as f:
        pickle.dump({'target': all_labels.cpu().numpy(), 'W_outer': W_outer, 'lambda_H': args.lambda_H, 'lambda_W': args.lambda_W}, f)
        log('--store theoretical result to {}'.format(filename))
        log('====> Theoretical ww00: {:.4f}'.format(W_outer[0, 0]))

    # ================== Training ==================
    criterion = nn.MSELoss()
    nc_tracker = Train_Vars(dim=args.num_y)
    wandb.watch(model, criterion, log="all", log_freq=10)
    for epoch in range(args.start_epoch, args.max_epoch):

        all_feats, running_train_loss = train_one_epoch(model, train_data_loader, optimizer, criterion, args=args)
        scheduler.step()

        # === cosine between Wi's
        W = model.fc.weight.data  # [2, 512]

        W_outer_pred = torch.matmul(W, W.T)
        W_outer_d = W_outer - W_outer_pred.cpu().numpy()
        W_outer_d = np.sum(W_outer_d ** 2)

        # === compute projection error
        U = gram_schmidt(W)
        P_E = torch.mm(U.T, U)  # Projection matrix using orthonormal basis
        h_projected = torch.mm(all_feats, P_E)
        projection_error_train = mse_loss(h_projected, all_feats).item()

        nc_dt = {'lr': optimizer.param_groups[0]['lr'],
                 'train_mse': running_train_loss,
                 'train_proj_error': projection_error_train,
                 'ww00': W_outer_pred[0, 0].item(),
                 'w_outer_d': W_outer_d}
        nc_tracker.load_dt(nc_dt, epoch=epoch)

        wandb.log(
            {'train/lr': optimizer.param_groups[0]['lr'],
             'train/train_mse': running_train_loss,
             'train/project_error': projection_error_train,
             'W/ww00': W_outer_pred[0, 0].item(),
             'W/W_outer_d': W_outer_d
             },
            step=epoch)

        log(
            'Epoch {}/{}, runnning train mse: {:.4f}, ww00: {:.4f}, W_outer_d: {:.4f}'.format(
                epoch, args.max_epoch, running_train_loss, W_outer_pred[0, 0].item(), W_outer_d
            ))

        if epoch % args.save_freq == 0:
            ckpt_path = os.path.join(args.save_dir, 'ep{}_ckpt.pth'.format(epoch))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, ckpt_path)

            log('--save model to {}'.format(ckpt_path))

        if epoch % (args.save_freq * 10) == 0:
            filename = os.path.join(args.save_dir, 'train_nc{}.pkl'.format(epoch))
            with open(filename, 'wb') as f:
                pickle.dump(nc_tracker, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Regression NC")
    parser.add_argument('--dataset', type=str, default='Carla')
    parser.add_argument('--max_epoch', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_y', type=int, default=2)
    parser.add_argument('--feat', type=str, default='b')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lambda_H', type=float, default=1e-3)
    parser.add_argument('--lambda_W', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--scheduler', type=str, default='multi_step')

    parser.add_argument('--ufm', default=False, action='store_true')
    parser.add_argument('--bias', default=False, action='store_true')

    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--save_freq', default=5, type=int)

    parser.add_argument('--exp_name', type=str, default='exp')
    args = parser.parse_args()
    
    args.save_dir = os.path.join("./result/", args.exp_name)
    if args.resume is not None: 
        args.resume = os.path.join('./result', args.resume)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    set_log_path(args.save_dir)
    log('save log to path {}'.format(args.save_dir))
    log(print_args(args))

    main(args)
