import os
import pdb
import torch
import wandb
import random
import pickle
import argparse
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
from sklearn.decomposition import PCA

from model import RegressionResNet, MLP
from dataset import SubDataset, get_dataloader, MujocoBuffer_Noise, DATA_FOLDER
from train_utils import get_feat_pred, gram_schmidt, get_scheduler, Train_Vars, get_theoretical_solution, compute_metrics
from utils import print_model_param_nums, set_log_path, log, print_args, matrix_with_angle


def train_one_epoch(model, data_loader, optimizer, criterion, args):
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    running_loss = 0.0
    all_feats = []
    for batch_idx, batch in enumerate(data_loader):
        images = batch['input'].to(device, non_blocking=True)
        targets = batch['target_noisy'].to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs, feats = model(images, ret_feat=True)
        all_feats.append(feats.data)

        loss = criterion(outputs, targets)
        if args.ufm:
            l2reg_H = torch.sum(feats ** 2) * args.lambda_H / args.batch_size
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

    # ================== load data ==================
    train_dataset = MujocoBuffer_Noise(data_folder=DATA_FOLDER, env=args.dataset, split='train',
                                       data_ratio=args.data_ratio, args=args)
    val_dataset = MujocoBuffer_Noise(data_folder=DATA_FOLDER, env=args.dataset, split='test',
                                     data_ratio=args.data_ratio, args=args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    args.num_x = train_loader.dataset.state_dim
    if args.which_y == -1:
        args.num_y = train_loader.dataset.action_dim
    else:
        args.num_y = 1

    # ================== setup wandb  ==================

    os.environ["WANDB_API_KEY"] = "0c0abb4e8b5ce4ee1b1a4ef799edece5f15386ee"
    os.environ["WANDB_MODE"] = "online"  # "dryrun"
    os.environ["WANDB_CACHE_DIR"] = "/scratch/lg154/sseg/.cache/wandb"
    os.environ["WANDB_CONFIG_DIR"] = "/scratch/lg154/sseg/.config/wandb"
    os.environ["WANDB_ARTIFACT_DIR"] = "/scratch/lg154/sseg/wandb"
    os.environ["WANDB_DATA_DIR"] = "/scratch/lg154/sseg/wandb/data"
    wandb.login(key='0c0abb4e8b5ce4ee1b1a4ef799edece5f15386ee')
    wandb.init(project='noisy_training',  # + args.dataset,
               name=args.exp_name.split('/')[-1]
               )
    wandb.config.update(args)

    # =================== theoretical solution ================
    if args.dataset in ['swimmer', 'reacher', 'hopper']:
        theory_stat = train_loader.dataset.get_theory_stats(noisy=False, center=args.bias)

    # ===================    Load model   ===================
    if args.arch.startswith('res'):
        model = RegressionResNet(pretrained=True, num_outputs=args.num_y, args=args).to(device)
    elif args.arch.startswith('mlp'):
        model = MLP(in_dim=args.num_x, out_dim=args.num_y, args=args, arch=args.arch.replace('mlp', ''))
    if torch.cuda.is_available():
        model = model.cuda()

    num_params = sum([param.nelement() for param in model.parameters()])
    log('--- total num of params: {} ---'.format(num_params))

    if model.fc.bias is not None:
        log("--- classification layer has bias terms. ---")
    else:
        log("--- classification layer DO NOT have bias terms. ---")

    # ======= optimizer and scheduler
    if args.ufm:
        optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=args.lr, weight_decay=0)
    else:
        optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=args.lr, weight_decay=args.wd)
    scheduler = get_scheduler(args, optimizer)
    if args.warmup > 0:
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: (epoch + 1) / args.warmup)

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

    # ================== Training ==================
    if args.loss in ['mse', 'l2']:
        criterion = nn.MSELoss()
    elif args.loss in ['mae', 'l1']:
        criterion = nn.L1Loss()
    wandb.watch(model, criterion, log="all", log_freq=20)
    for epoch in range(args.start_epoch, args.max_epoch):

        if epoch == 0 or epoch % args.log_freq == 0:

            # === weight Matrix
            W = model.fc.weight.data  # [2, 512]
            WWT = (W @ W.T).cpu().numpy()

            # ===============compute train mse and projection error==================
            all_feats, preds, labels = get_feat_pred(model, train_loader)
            if args.y_norm not in ['null', 'n']:
                y_shift, std = torch.tensor(train_loader.dataset.y_shift).to(preds.device), torch.tensor(train_loader.dataset.std).to(preds.device)
                preds = preds @ std + y_shift
                labels = labels @ std + y_shift
            if preds.shape[-1] == 2:
                train_loss0 = torch.sum((preds[:, 0] - labels[:, 0]) ** 2) / preds.shape[0]
                train_loss1 = torch.sum((preds[:, 1] - labels[:, 1]) ** 2) / preds.shape[0]
            train_loss = criterion(preds, labels)
            train_mse = ((preds - labels)**2).mean()

            nc_train = compute_metrics(W, all_feats)
            train_hnorm = torch.norm(all_feats, p=2, dim=1).mean().item()
            train_wnorm = torch.norm(W, p=2, dim=1).mean().item()

            # ===============compute val mse and projection error==================
            all_feats, preds, labels = get_feat_pred(model, val_loader)
            if args.y_norm not in ['null', 'n']:
                y_shift, std = torch.tensor(train_loader.dataset.y_shift).to(preds.device), torch.tensor(train_loader.dataset.std).to(preds.device)
                preds = preds @ std + y_shift
                labels = labels @ std + y_shift
            if preds.shape[-1] == 2:
                val_loss0 = torch.sum((preds[:, 0] - labels[:, 0]) ** 2) / preds.shape[0]
                val_loss1 = torch.sum((preds[:, 1] - labels[:, 1]) ** 2) / preds.shape[0]
            val_loss = criterion(preds, labels)
            val_mse = ((preds-labels)**2).mean()

            nc_val = compute_metrics(W, all_feats)
            val_hnorm = torch.norm(all_feats, p=2, dim=1).mean().item()
            del all_feats, preds, labels

            # ================ NC2 ================
            WWT_normalized = WWT / np.linalg.norm(WWT)
            min_eigval = theory_stat['min_eigval']
            Sigma_sqrt = theory_stat['Sigma_sqrt']
            W_outer = args.lambda_H * (Sigma_sqrt / np.sqrt(args.lambda_H * args.lambda_W) - np.eye(args.num_y))

            c_to_plot = np.linspace(0, min_eigval, num=1000)
            NC2_to_plot = []
            for c in c_to_plot:
                c_sqrt = c ** 0.5
                A = Sigma_sqrt - c_sqrt * np.eye(Sigma_sqrt.shape[0])
                A_normalized = A / np.linalg.norm(A)
                diff_mat = WWT_normalized - A_normalized
                NC2_to_plot.append(np.linalg.norm(diff_mat))

            data = [[a, b] for (a, b) in zip(c_to_plot, NC2_to_plot)]
            table = wandb.Table(data=data, columns=["c", "NC2"])
            wandb.log(
                {"NC2(c)": wandb.plot.line(table, "c", "NC2", title="NC2 as a Function of c")},
                step=epoch
            )
            best_c = c_to_plot[np.argmin(NC2_to_plot)]
            NC2 = min(NC2_to_plot)

            # ================ log to wandb ================
            nc_dt = {
                'train/train_nc1': nc_train['nc1'],
                'train/train_nc3': nc_train['nc3'],
                'train/train_nc3a': nc_train['nc3a'],
                'train/train_loss': train_loss,
                'train/train_mse': train_mse,

                'val/val_loss': val_loss,
                'val/val_mse': val_mse,
                'val/val_nc1': nc_val['nc1'],
                'val/val_nc3': nc_val['nc3'],
                'val/val_nc3a': nc_val['nc3a'],

                'ww00': WWT[0, 0].item(),
                'ww01': WWT[0, 1].item() if args.num_y == 2 else 0,
                'ww11': WWT[1, 1].item() if args.num_y == 2 else 0,
                'w_cos': F.cosine_similarity(W[0], W[1], dim=0).item() if args.num_y == 2 else 0,
                'W/nc2': NC2,
                'W/best_c': best_c,

                'NC2/ww00_d': abs(WWT[0, 0].item() - W_outer[0, 0]) / (abs(W_outer[0, 0]) + 1e-8),
                'NC2/ww01_d': abs(WWT[0, 1].item() - W_outer[0, 1]) / (abs(W_outer[0, 1]) + 1e-8) if args.num_y == 2 else 0,
                'NC2/ww11_d': abs(WWT[1, 1].item() - W_outer[1, 1]) / (abs(W_outer[1, 1]) + 1e-8) if args.num_y == 2 else 0,
                'NC2/ww_d': np.sum((WWT / np.linalg.norm(WWT) - W_outer / np.linalg.norm(W_outer)) ** 2),
                'NC2/ww_d1': np.sum(((WWT - W_outer) / np.linalg.norm(W_outer)) ** 2),

                'other/lr': optimizer.param_groups[0]['lr'],
                'other/train_hnorm': train_hnorm,
                'other/val_hnorm': val_hnorm,
                'other/wnorm': train_wnorm
            }
            wandb.log(nc_dt, step=epoch)

            if args.which_y == -1 and args.num_y == 2:
                wandb.log({'train/train_mse0': train_loss0, 'train/train_mse1': train_loss1,
                           'val/val_mse0': val_loss0, 'val/val_mse1': val_loss1}, step=epoch)
            elif args.which_y == 0:
                wandb.log({'train/train_mse0': train_loss, 'val/val_mse0': val_loss}, step=epoch)
            elif args.which_y == 1:
                wandb.log({'train/train_mse1': train_loss, 'val/val_mse1': val_loss}, step=epoch)

            log('Epoch {}/{}, runnning train loss: {:.4f}, ww00: {:.4f}, ww01: {:.4f}, ww11: {:.4f}'.format(
                epoch, args.max_epoch, train_loss, nc_dt['ww00'], nc_dt['ww01'], nc_dt['ww11']
            ))

        if (epoch == 0 or epoch % args.save_freq == 0) and args.save_freq > 0:
            ckpt_path = os.path.join(args.save_dir, 'ep{}_ckpt.pth'.format(epoch))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, ckpt_path)

            log('--save model to {}'.format(ckpt_path))

        # =============== train model ==================
        all_feats, train_loss = train_one_epoch(model, train_loader, optimizer, criterion, args=args)
        if epoch < args.warmup:
            warmup_scheduler.step()
        else:
            scheduler.step()
        # =============== save w
        if args.save_w == 't' and (epoch + 1) % 100 == 0:
            import pickle
            with open(os.path.join(args.save_dir, 'fc_w.pkl'), 'wb') as f:
                pickle.dump({'w': model.fc.weight.data.cpu().numpy(), 'wwt': W_outer, 'Sigma_sqrt': Sigma_sqrt}, f)


def set_seed(SEED=666):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Regression NC")
    parser.add_argument('--dataset', type=str, default='Carla')
    parser.add_argument('--data_ratio', type=float, default=1.0)
    parser.add_argument('--noise_ratio', type=float, default=0.0)
    parser.add_argument('--loss', type=str, default='l2')

    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--num_y', type=int, default=2)
    parser.add_argument('--which_y', type=int, default=-1)
    parser.add_argument('--y_norm', type=str, default='null')
    parser.add_argument('--x_norm', type=str, default='null')
    parser.add_argument('--act', type=str, default='relu')
    parser.add_argument('--w', type=str, default='null')
    parser.add_argument('--bn', type=str, default='f')  # f|t|p false|true|parametric
    parser.add_argument('--init_s', type=float, default='1.0')

    parser.add_argument('--max_epoch', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--feat', type=str, default='b')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument('--lambda_H', type=float, default=1e-3)
    parser.add_argument('--lambda_W', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--scheduler', type=str, default='multi_step')
    parser.add_argument('--save_w', type=str, default='f')

    parser.add_argument('--ufm', default=False, action='store_true')
    parser.add_argument('--bias', default=False, action='store_true')

    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--save_freq', default=10, type=int)
    parser.add_argument('--log_freq', default=20, type=int)

    parser.add_argument("--seed", type=int, default=2021, help="random seed")
    parser.add_argument('--exp_name', type=str, default='exp')
    args = parser.parse_args()

    args.save_dir = os.path.join("./result/{}/".format(args.dataset), args.exp_name)
    if args.resume is not None:
        args.resume = os.path.join('./result', args.resume)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    set_log_path(args.save_dir)
    log('save log to path {}'.format(args.save_dir))
    log(print_args(args))

    set_seed(args.seed)
    main(args)
