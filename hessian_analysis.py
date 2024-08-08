import numpy as np
import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from utils import * # get the dataset
from pyhessian import hessian # Hessian computation
from density_plot import get_esd_plot # ESD plot
import matplotlib.pyplot as plt
from density_plot import density_generate

from model import MLP
from dataset import get_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === model config ====
config = ['--arch', 'mlp256',
          '--y_norm', 'std',
          ]

parser = argparse.ArgumentParser(description="Regression NC")
parser.add_argument('--dataset', type=str, default='reacher')
parser.add_argument('--data_ratio', type=float, default=0.1)

parser.add_argument('--arch', type=str, default='mlp256')
parser.add_argument('--num_y', type=int, default=1)
parser.add_argument('--which_y', type=int, default=0)
parser.add_argument('--y_norm', type=str, default='null')
parser.add_argument('--x_norm', type=str, default='null')
parser.add_argument('--act', type=str, default='relu')
parser.add_argument('--w', type=str, default='null')
parser.add_argument('--bn', type=str, default='f')  # f|t|p false|true|parametric
parser.add_argument('--init_s', type=float, default='1.0')
parser.add_argument('--bias', default=False, action='store_true')

parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--feat', type=str, default='null')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--warmup', type=int, default=0)
parser.add_argument('--wd', type=float, default=1e-2)
args = parser.parse_args(config)
args.exp_name = f'dr0.1_{args.arch}null_YD{args.which_y}_WD{format(args.wd, ".0e").replace("e-0", "e-")}_BN{args.bn}_{args.y_norm}'

# ==== get data ====

train_loader, val_loader = get_dataloader(args)
args.num_x = train_loader.dataset.state_dim
if args.which_y == -1:
    args.num_y = train_loader.dataset.action_dim
else:
    args.num_y = 1

# for illustrate, we only use one batch to do the tutorial
for batch in train_loader:
    inputs, targets = batch['input'].to(device), batch['target'].to(device)
    break

# ===== get the model =====
ep_hessian = {}

for ep in [0, 400]:
    model = MLP(in_dim=args.num_x, out_dim=args.num_y, args=args, arch=args.arch.replace('mlp', ''))

    args.exp_name = f'dr0.1_{args.arch}null_YD{args.which_y}_WD{format(args.wd, ".0e").replace("e-0", "e-")}_BN{args.bn}_{args.y_norm}'
    model_path = os.path.join("./result/{}/".format(args.dataset), args.exp_name, 'ep{}_ckpt.pth'.format(ep))
    if not torch.cuda.is_available():
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # change the model to eval mode to disable running stats upate
    model = model.to(device)
    model.eval()
    criterion = nn.MSELoss()

    # create the hessian computation module
    hessian_comp = hessian(model, criterion, data=(inputs, targets), cuda=False)

    # top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=128)

    density_eigen, density_weight = hessian_comp.density()
    # get_esd_plot(density_eigen, density_weight)

    ep_hessian[ep] = [density_eigen, density_weight]

min_eigen, max_eigen = 0.0, 0.0
for ep in ep_hessian.keys():
    density_eigen, density_weight = ep_hessian[ep]
    density, grids = density_generate(density_eigen, density_weight)
    plt.plot(grids, density + 1.0e-7, label = 'ep' + str(ep))
    min_eigen = np.min([min_eigen, np.min(density_eigen)])
    max_eigen = np.max([max_eigen, np.max(density_eigen)])

plt.yscale('log')
plt.ylabel('Density (Log Scale)', fontsize=14, labelpad=10)
plt.xlabel('Eigenvlaue', fontsize=14, labelpad=10)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.axis([min_eigen - 1, max_eigen + 1, None, None])
plt.tight_layout()
plt.show()
plt.legend()

