
import argparse
import os, pickle, torch, io
from matplotlib import pyplot as plt
from train_utils import Graph_Vars
import numpy as np
from dataset import get_dataloader
from model import RegressionResNet, MLP



input = ['--dataset', 'reacher', '--resume', 'ufm_dr0.1_f_lambda1e-21e-2_ob_sv', '--ep', '490']

input = ['--dataset', 'reacher', '--resume', 'wd1e-2_dr0.1_null_ob', '--ep', '490', '--feat', 'null']


# input = ['--dataset', 'swimmer', '--resume', 'ufm_dr0.1_f_lambda1e-31e-3_ob_sv', '--ep', '400']


parser = argparse.ArgumentParser(description="Regression NC")
parser.add_argument('--dataset', type=str, default='swimmer')
parser.add_argument('--data_ratio', type=float, default=0.0022222222)
parser.add_argument('--resume', type=str, default='ufm_dr0.1_f_ob_sv') # 'ufm_dr0.1_f_ob_sv, 'dr1_f_ob_lr2_sv'
parser.add_argument('--ep', type=int, default=200)

parser.add_argument('--arch', type=str, default='mlp256_256')
parser.add_argument('--feat', type=str, default='f')
parser.add_argument('--bias', default=False, action='store_true')

parser.add_argument('--y_norm', type=str, default='null')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_y', type=int, default=2)
args = parser.parse_args(input)

if 'dr1' in args.resume:
    args.data_ratio = 1
elif 'dr0.1' in args.resume:
    args.data_ratio = 0.1

train_loader, val_loader = get_dataloader(args, shuffle=False)

def get_error_y(args, resume):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ======================== load model ========================
    if args.dataset == 'swimmer' or args.dataset == 'reacher':
        args.num_x = train_loader.dataset.state_dim
        args.num_y = train_loader.dataset.action_dim

    if args.arch.startswith('res'):
        model = RegressionResNet(pretrained=True, num_outputs=args.num_y, args=args).to(device)
    elif args.arch.startswith('mlp'):
        model = MLP(in_dim=args.num_x, out_dim=args.num_y, args=args, arch=args.arch.replace('mlp', ''))

    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))


    # ================== get model prediction and error ==================
    if args.dataset == 'reacher':
        sqrt_sigma = np.array([[0.10558, -0.00665],[-0.00665, 0.11127]])  # s00: 0.10273, s01: -0.00232, s11: 0.11210
        # s00: 0.10558, s01: -0.00665, s11: 0.11127
    elif args.dataset == 'swimmer':
        sqrt_sigma = np.array([[1.3081717 , 0.04179913],[0.04179913, 0.59840244]])
        # np.array([[1.30639,  0.04333], [0.04333, 0.60476]])
        # 1.30639, s01: 0.04333, s11: 0.60476

    model.eval()
    all_preds, all_labels = [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            input = batch['input'].to(device)
            target = batch['target'].to(device)
            input, target = input.to(device), target.to(device)
            pred, _ = model(input, ret_feat=True)

            all_preds.append(pred)
            all_labels.append(target)

            if i >= 20:
                break

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

    error = (all_preds - all_labels).cpu().numpy()  # [N, 2]
    normalized_y = (np.linalg.inv(sqrt_sigma) @ all_labels.T.cpu().numpy()).T

    return error, normalized_y, all_labels


# ================== PLot ==================

vmin, vmax=-1, 1

fig, axes = plt.subplots(1, 2)

for i, epoch in enumerate([None, args.ep]):
    if epoch == None:
        resume = None
    else:
        resume = os.path.join('result', args.dataset, args.resume, 'ep{}_ckpt.pth'.format(epoch))
    error, normalized_y, all_labels = get_error_y(args, resume)

    p=axes[i].scatter(error[:,0], error[:,1], cmap='viridis',  c=normalized_y[:,1]/normalized_y[:,0], vmin=vmin, vmax=vmax,
                    s=3, alpha=0.5)
    axes[i].axhline(y=0, color='gray', linestyle='--', linewidth=1)
    axes[i].axvline(x=0, color='gray', linestyle='--', linewidth=1)
    if i==0:
        axes[i].set_xlim(-0.4,0.4)
        axes[i].set_ylim(-0.4, 0.4)
        axes[i].set_title('Random initialized model')

    elif i==1:
        axes[i].set_xlim(-0.2,0.2)
        axes[i].set_ylim(-0.2, 0.2)
        axes[i].set_title('Model after convergence')
    axes[i].set_xlabel('Prediction error $\epsilon^{(1)}$')
    axes[i].set_ylabel('Prediction error $\epsilon^{(2)}$')
cbar = fig.colorbar(p, ax=[axes[0], axes[1]], orientation='vertical')
# cbar.set_label('Color Based on'+r'$\tilde{y}^{2}/\tilde{y}^{1}$')


# ================== PLot with erro0==================

error = [0] * 3
normalized_y = [0] * 3
all_labels = [0] * 3

for i, epoch in enumerate([None, args.ep]):
    if epoch == None:
        resume = None
    else:
        resume = os.path.join('result', args.dataset, args.resume, 'ep{}_ckpt.pth'.format(epoch))
    error_, normalized_y_, all_labels_ = get_error_y(args, resume)
    error[i+1], normalized_y[i+1], all_labels[i+1] = error_, normalized_y_, all_labels_

all_labels[0], normalized_y[0] = all_labels[1], normalized_y[1]
error[0] = torch.mean(all_labels[1], dim=0) - all_labels[0]



vmin, vmax=-1, 1
fig, axes = plt.subplots(1, 3)
for i in range(3):
    p=axes[i].scatter(error[i][:,0], error[i][:,1], cmap='viridis',  c=normalized_y[i][:,1]/normalized_y[i][:,0], vmin=vmin, vmax=vmax, s=3, alpha=0.5)
    axes[i].axhline(y=0, color='gray', linestyle='--', linewidth=1)
    axes[i].axvline(x=0, color='gray', linestyle='--', linewidth=1)
    if i==0:
        axes[i].set_xlim(-0.4,0.4)
        axes[i].set_ylim(-0.4, 0.4)
        axes[i].set_title('Baseline Model')
    elif i==1:
        axes[i].set_xlim(-0.4,0.4)
        axes[i].set_ylim(-0.4, 0.4)
        axes[i].set_title('Random initialized model')
    elif i==2:
        axes[i].set_xlim(-0.2,0.2)
        axes[i].set_ylim(-0.2, 0.2)
        axes[i].set_title('Model after convergence')
    axes[i].set_xlabel('Prediction error $\epsilon^{(1)}$')
    axes[i].set_ylabel('Prediction error $\epsilon^{(2)}$')
cbar = fig.colorbar(p, ax=axes, orientation='vertical')
# cbar.set_label('Color Based on'+r'$\tilde{y}^{2}/\tilde{y}^{1}$')


