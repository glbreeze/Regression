import os
import h5py
import pickle 
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

DATA_FOLDER = '../dataset/mujoco_data/'


def get_dataloader(args):
    if args.dataset == 'Carla' or args.dataset == 'carla':
        train_dataset = SubDataset('/vast/lg154/Carla_JPG/Train/train_list.txt', '/vast/lg154/Carla_JPG/Train/sub_targets.pkl', transform=transform, dim=args.num_y)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
        val_dataset = SubDataset('/vast/lg154/Carla_JPG/Val/val_list.txt', '/vast/lg154/Carla_JPG/Val/sub_targets.pkl', transform=transform, dim=args.num_y)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    elif args.dataset in ["swimmer", 'reacher', 'hopper']:
        train_dataset = MujocoBuffer(data_folder=DATA_FOLDER,
            env=args.dataset,
            split='train',
            data_ratio=args.data_ratio,
            args=args
        )
        val_dataset = MujocoBuffer(
            data_folder=DATA_FOLDER,
            env=args.dataset,
            split='test',
            data_ratio=args.data_ratio,
            args=args,
            y_shift=train_dataset.y_shift,
            div=train_dataset.div,
            x_shift=train_dataset.x_shift,
            x_div=train_dataset.x_div,
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
    elif args.dataset.endswith('_ab'):
        train_dataset = MujocoAblate(data_folder=DATA_FOLDER,
                                     env=args.dataset.replace('_ab', ''),
                                     split='train',
                                     data_ratio=args.data_ratio,
                                     args=args
                                     )
        val_dataset = MujocoAblate(
            data_folder=DATA_FOLDER,
            env=args.dataset.replace('_ab', ''),
            split='test',
            data_ratio=args.data_ratio,
            args=args,
            y_shift=train_dataset.y_shift,
            div=train_dataset.div,
            x_shift=train_dataset.x_shift,
            x_div=train_dataset.x_div,
        )

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    return train_loader, val_loader


class ImageTargetDataset(Dataset):

    def __init__(self, images_dir, targets_dir, transform=None, dim=2):
        """
        Args:
            images_dir (str): Path to the directory containing image files.
            targets_dir (str): Path to the directory containing target files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images_dir = images_dir
        self.targets_dir = targets_dir
        self.transform = transform
        self.dim = dim

        self.image_filenames = [f for f in sorted(os.listdir(images_dir)) if f.endswith('.jpeg')]
        self.target_filenames = [f for f in sorted(os.listdir(targets_dir)) if f.endswith('.npy')]

        if len(self.image_filenames) != len(self.target_filenames):
            raise ValueError("The number of images and targets do not match!")

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):

        img_path = os.path.join(self.images_dir, self.image_filenames[idx])
        target_path = os.path.join(self.targets_dir, self.target_filenames[idx])

        image = Image.open(img_path).convert('RGB')
        target = np.load(target_path)

        if self.transform:
            image = self.transform(image)

        if self.dim==1:
            return {'input': image, 'target': torch.tensor(target[0], dtype=torch.float)}
        elif self.dim==2:
            return {'input': image, 'target': torch.tensor(target[[0, 10]], dtype=torch.float)}


class SubDataset(Dataset):

    def __init__(self, train_list, target_file, transform=None, dim=2):
        """
        Args:
            images_dir (str): Path to the directory containing image files.
            targets_dir (str): Path to the directory containing target files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        folder = os.path.dirname(train_list)
        self.img_path = []

        with open(train_list, 'r') as f:
            for line in f:
                self.img_path.append(os.path.join(folder, 'sub_images', line.strip()))
        
        with open(target_file, 'rb') as file:
            self.targets = pickle.load(file)
                
        self.transform = transform
        self.dim = dim

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):

        img_path = self.img_path[idx]
        image = Image.open(img_path).convert('RGB')
    
        target = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        if self.dim==1:
            return {'input': image, 'target': torch.tensor(target[0], dtype=torch.float)}
        elif self.dim==2:
            return {'input': image, 'target': torch.tensor(target[[0, 10]], dtype=torch.float)}


class MujocoBuffer(Dataset):
    def __init__(
            self,
            data_folder: str,
            env: str,
            split: str,
            data_ratio,
            args = None,
            y_shift = None,
            div = None,
            x_shift=None,
            x_div=None,
    ):
        self.size = 0
        self.args=args
        self.state_dim = 0
        self.action_dim = 0
        self.env = env

        self.states, self.actions = None, None
        self._load_dataset(data_folder, env, split, data_ratio)

        self.y_shift, self.div = None, None
        self.x_shift, self.x_div = None, None
        if args.y_norm not in ['null', 'n']:
            self.normalize_y(split=split, y_shift=y_shift, div=div)
        if args.x_norm not in ['null', 'n']:
            self.normalize_x(split=split, x_shift=x_shift, x_div=x_div)

    def normalize_x(self, split, x_shift, x_div):
        if split == 'train':
            if self.args.x_norm == 'norm':
                self.x_shift = np.mean(self.states, axis=0)
                centered_data = self.states - self.x_shift  # [B, d]
                covariance_matrix = centered_data.T @ centered_data / len(self.states)
                if self.env == 'reacher':    # last dim value is constant 0 
                    covariance_matrix[-1, -1] = 1.0
                self.x_div = np.diag(1 / np.sqrt(np.diag(covariance_matrix)))
                self.states = centered_data @ self.x_div
        else:
            self.x_shift = x_shift
            self.x_div = x_div
            centered_data = self.states - x_shift
            self.states = centered_data @ self.x_div

    def normalize_y(self, split, y_shift, div):

        assert self.args.y_norm in ['norm', 'norm0', 'std', 'scale', 'std2']
        if split == 'train':
            if self.args.y_norm == 'norm':
                self.y_shift = np.mean(self.actions, axis=0)
                centered_data = self.actions - self.y_shift
                covariance_matrix = np.dot(centered_data.T, centered_data) / len(self.actions)
                if self.args.which_y == -1:
                    self.div = np.diag(1 / np.sqrt(np.diag(covariance_matrix)))
                    self.std = np.diag(np.sqrt(np.diag(covariance_matrix)))
                else:
                    self.div, self.std = 1/np.sqrt(covariance_matrix), np.sqrt(covariance_matrix)
            elif self.args.y_norm == 'norm0':
                self.y_shift = np.zeros(self.actions.shape[-1])
                centered_data = self.actions          # no centering
                covariance_matrix = np.dot(centered_data.T, centered_data) / len(self.actions)
                if self.args.which_y == -1:
                    self.div = np.diag(1 / np.sqrt(np.diag(covariance_matrix)))
                    self.std = np.diag(np.sqrt(np.diag(covariance_matrix)))
                else:
                    self.div, self.std = 1/np.sqrt(covariance_matrix), np.sqrt(covariance_matrix)
            elif self.args.y_norm in ['std', 'std2']:
                self.y_shift = np.mean(self.actions, axis=0)
                centered_data = self.actions - self.y_shift
                covariance_matrix = np.dot(centered_data.T, centered_data) / len(self.actions)
                if self.args.which_y == -1:
                    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
                    self.div = eigenvectors @ np.diag(1 / np.sqrt(eigenvalues)) @ np.linalg.inv(eigenvectors)
                    self.std = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ np.linalg.inv(eigenvectors)
                else:
                    self.div, self.std = 1/np.sqrt(covariance_matrix), np.sqrt(covariance_matrix)
                if len(self.args.y_norm) > 3:
                        self.div = self.div * float(self.args.y_norm[3:])
                        self.std = self.std / float(self.args.y_norm[3:])
        else:  # test
            self.y_shift = y_shift
            self.div = div
            centered_data = self.actions - self.y_shift

        self.actions = centered_data @ self.div
        # self.actions = centered_data / self.div

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def _load_dataset(self, data_folder, env, split, data_ratio):
        file_name = '%s_%s.pkl' % (env, split)
        file_path = os.path.join(data_folder, file_name)
        try:
            with open(file_path, 'rb') as file:
                dataset = pickle.load(file)
                if data_ratio <= 1: 
                    self.size = int(dataset['observations'].shape[0] * data_ratio)
                else: 
                    self.size = int(data_ratio) if data_ratio<= dataset['observations'].shape[0] else dataset['observations'].shape[0]
                self.states = dataset['observations'][:self.size, :]
                self.actions = dataset['actions'][:self.size, :]
            print('Successfully load dataset from: ', file_path)
            if self.args.which_y == -1:
                pass
            elif self.args.which_y >= 0:
                self.actions = self.actions[:, self.args.which_y].reshape(-1, 1)
        except Exception as e:
            print(e)

        self.state_dim = self.states.shape[1]
        self.action_dim = 1 if self.actions.ndim == 1 else self.actions.shape[1]
        print(f"Dataset size: {self.size}; State Dim: {self.state_dim}; Action_Dim: {self.action_dim}.")

    def get_state_dim(self):
        return self.state_dim

    def get_action_dim(self):
        return self.action_dim

    def get_theory_stats(self, center=False):
        actions = self.actions
        mu = np.mean(actions, axis=0)
        if center:
            centered_actions = actions - mu
            Sigma = centered_actions.T @ centered_actions / centered_actions.shape[0]
        else:
            Sigma = actions.T @ actions / actions.shape[0]

        if self.args.which_y == -1: 
            eig_vals, eig_vecs = np.linalg.eigh(Sigma)
            sqrt_eig_vals = np.sqrt(eig_vals)
            Sigma_sqrt = eig_vecs @ np.diag(sqrt_eig_vals) @ np.linalg.inv(eig_vecs)
            min_eigval, max_eigval = eig_vals[0], eig_vals[-1]
        else: 
            Sigma_sqrt = np.sqrt(Sigma)
            min_eigval, max_eigval = Sigma_sqrt, Sigma_sqrt

        return {
            'mu': mu, 
            'Sigma': Sigma, 
            'Sigma_sqrt': Sigma_sqrt,
            'min_eigval': min_eigval,
            'max_eigval': max_eigval,
        }

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        states = self.states[idx]
        actions = self.actions[idx]
        return {
            'input': self._to_tensor(states),
            'target': self._to_tensor(actions)
        }


class MujocoBuffer_Noise(Dataset):
    def __init__(
            self,
            data_folder: str,
            env: str,
            split: str,
            data_ratio,
            args = None,
            y_shift = None,
            div = None,
            x_shift=None,
            x_div=None,
    ):
        self.size = 0
        self.args=args
        self.env = env

        self.state_dim, self.action_dim = 0, 0
        self.states, self.actions = None, None
        self._load_dataset(data_folder, env, split, data_ratio)

        self.y_shift, self.div = None, None
        self.x_shift, self.x_div = None, None
        if args.y_norm not in ['null', 'n']:
            self.normalize_y(split=split, y_shift=y_shift, div=div)
        if args.x_norm not in ['null', 'n']:
            self.normalize_x(split=split, x_shift=x_shift, x_div=x_div)

    def normalize_x(self, split, x_shift, x_div):
        if split == 'train':
            if self.args.x_norm == 'norm':
                self.x_shift = np.mean(self.states, axis=0)
                centered_data = self.states - self.x_shift  # [B, d]
                covariance_matrix = centered_data.T @ centered_data / len(self.states)
                if self.env == 'reacher':    # last dim value is constant 0
                    covariance_matrix[-1, -1] = 1.0
                self.x_div = np.diag(1 / np.sqrt(np.diag(covariance_matrix)))
                self.states = centered_data @ self.x_div
        else:
            self.x_shift = x_shift
            self.x_div = x_div
            centered_data = self.states - x_shift
            self.states = centered_data @ self.x_div

    def normalize_y(self, split, y_shift, div):

        assert self.args.y_norm in ['norm', 'norm0', 'std', 'scale', 'std2']
        if split == 'train':
            if self.args.y_norm == 'norm':
                self.y_shift = np.mean(self.actions, axis=0)
                centered_data = self.actions - self.y_shift
                covariance_matrix = np.dot(centered_data.T, centered_data) / len(self.actions)
                if self.args.which_y == -1:
                    self.div = np.diag(1 / np.sqrt(np.diag(covariance_matrix)))
                    self.std = np.diag(np.sqrt(np.diag(covariance_matrix)))
                else:
                    self.div, self.std = 1/np.sqrt(covariance_matrix), np.sqrt(covariance_matrix)
            elif self.args.y_norm == 'norm0':
                self.y_shift = np.zeros(self.actions.shape[-1])
                centered_data = self.actions          # no centering
                covariance_matrix = np.dot(centered_data.T, centered_data) / len(self.actions)
                if self.args.which_y == -1:
                    self.div = np.diag(1 / np.sqrt(np.diag(covariance_matrix)))
                    self.std = np.diag(np.sqrt(np.diag(covariance_matrix)))
                else:
                    self.div, self.std = 1/np.sqrt(covariance_matrix), np.sqrt(covariance_matrix)
            elif self.args.y_norm in ['std', 'std2']:
                self.y_shift = np.mean(self.actions, axis=0)
                centered_data = self.actions - self.y_shift
                covariance_matrix = np.dot(centered_data.T, centered_data) / len(self.actions)
                if self.args.which_y == -1:
                    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
                    self.div = eigenvectors @ np.diag(1 / np.sqrt(eigenvalues)) @ np.linalg.inv(eigenvectors)
                    self.std = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ np.linalg.inv(eigenvectors)
                else:
                    self.div, self.std = 1/np.sqrt(covariance_matrix), np.sqrt(covariance_matrix)
                if len(self.args.y_norm) > 3:
                        self.div = self.div * float(self.args.y_norm[3:])
                        self.std = self.std / float(self.args.y_norm[3:])
        else:  # test
            self.y_shift = y_shift
            self.div = div
            centered_data = self.actions - self.y_shift

        self.actions = centered_data @ self.div
        # self.actions = centered_data / self.div

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def _load_dataset(self, data_folder, env, split, data_ratio):
        file_name = '%s_%s.pkl' % (env, split)
        file_path = os.path.join(data_folder, file_name)
        try:
            with open(file_path, 'rb') as file:
                dataset = pickle.load(file)
            print('Successfully load dataset from: ', file_path)
        except Exception as e:
            print(e)

        if data_ratio <= 1:
            self.size = int(dataset['observations'].shape[0] * data_ratio)
        else:
            self.size = int(data_ratio) if data_ratio <= dataset['observations'].shape[0] else dataset['observations'].shape[0]
        self.states = dataset['observations'][:self.size, :]
        self.actions = dataset['actions'][:self.size, :]
        if self.args.which_y >= 0:
            self.actions = self.actions[:, self.args.which_y].reshape(-1, 1)
        self.state_dim = self.states.shape[1]
        self.action_dim = self.actions.shape[1]
        print(f"Dataset size: {self.size}; State Dim: {self.state_dim}; Action_Dim: {self.action_dim}.")

        self.noisy_flag = np.random.choice([0, 1], size=self.size, p=[1 - self.args.noise_ratio, self.args.noise_ratio]).reshape(-1, 1)
        mu = np.mean(self.actions, axis=0)
        Sigma = (self.actions - mu).T @ (self.actions - mu) / len(self.actions)

        noise = np.random.multivariate_normal(np.zeros_like(mu), np.diag(np.diag(Sigma)), self.size)
        self.actions_noisy = self.actions + noise * self.noisy_flag

    def get_state_dim(self):
        return self.state_dim

    def get_action_dim(self):
        return self.action_dim

    def get_theory_stats(self, noisy=False, center=False):
        actions = self.actions_noisy if noisy == True else self.actions
        mu = np.mean(actions, axis=0)
        if center:
            actions = actions - mu
        Sigma = actions.T @ actions / actions.shape[0]

        if self.args.which_y == -1:
            eig_vals, eig_vecs = np.linalg.eigh(Sigma)
            sqrt_eig_vals = np.sqrt(eig_vals)
            Sigma_sqrt = eig_vecs @ np.diag(sqrt_eig_vals) @ np.linalg.inv(eig_vecs)
            min_eigval, max_eigval = eig_vals[0], eig_vals[-1]
        else:
            Sigma_sqrt = np.sqrt(Sigma)
            min_eigval, max_eigval = Sigma_sqrt, Sigma_sqrt

        return {
            'mu': mu,
            'Sigma': Sigma,
            'Sigma_sqrt': Sigma_sqrt,
            'min_eigval': min_eigval,
            'max_eigval': max_eigval,
        }

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        states = self.states[idx]
        actions = self.actions[idx]
        actions_noisy = self.actions_noisy[idx]
        noisy_flag = self.noisy_flag[idx]
        return {
            'input': self._to_tensor(states),
            'target': self._to_tensor(actions),
            'target_noisy': self._to_tensor(actions_noisy),
            'noisy_flag': self._to_tensor(noisy_flag)
        }


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class MujocoAblate(Dataset):
    def __init__(
            self,
            data_folder: str,
            env: str,
            split: str,
            data_ratio,
            args = None,
            y_shift = None,
            div = None,
            x_shift=None,
            x_div=None,
    ):
        self.size = 0
        self.args=args
        self.state_dim = 0
        self.action_dim = 0
        self.env = env

        self.states, self.actions = None, None
        self._load_dataset(data_folder, env, split, data_ratio)

        self.y_shift, self.div = None, None
        self.x_shift, self.x_div = None, None
        if args.y_norm not in ['null', 'n']:
            self.normalize_y(split=split, y_shift=y_shift, div=div)
        if args.x_norm not in ['null', 'n']:
            self.normalize_x(split=split, x_shift=x_shift, x_div=x_div)

    def normalize_x(self, split, x_shift, x_div):
        if split == 'train':
            if self.args.x_norm == 'norm':
                self.x_shift = np.mean(self.states, axis=0)
                centered_data = self.states - self.x_shift  # [B, d]
                covariance_matrix = centered_data.T @ centered_data / len(self.states)
                self.x_div = np.diag(1 / np.sqrt(np.diag(covariance_matrix)))
                self.states = centered_data @ self.x_div
            elif self.args.x_norm == 'std':
                self.x_shift = np.mean(self.states, axis=0)
                centered_data = self.states - self.x_shift  # [B, d]
                covariance_matrix = centered_data.T @ centered_data / len(self.states)
                eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
                self.x_div = eigenvectors @ np.diag(1 / np.sqrt(eigenvalues)) @ np.linalg.inv(eigenvectors)
                self.states = centered_data @ self.x_div
        else:
            self.x_shift = x_shift
            self.x_div = x_div
            centered_data = self.states - x_shift
            self.states = centered_data @ self.x_div

    def normalize_y(self, split, y_shift, div):

        assert self.args.y_norm in ['norm', 'norm0', 'std', 'scale', 'std2']
        if split == 'train':
            if self.args.y_norm == 'norm':
                self.y_shift = np.mean(self.actions, axis=0)
                centered_data = self.actions - self.y_shift
                covariance_matrix = np.dot(centered_data.T, centered_data) / len(self.actions)
                if self.args.which_y == -1:
                    self.div = np.diag(1 / np.sqrt(np.diag(covariance_matrix)))
                    self.std = np.diag(np.sqrt(np.diag(covariance_matrix)))
                else:
                    self.div, self.std = 1/np.sqrt(covariance_matrix), np.sqrt(covariance_matrix)
            elif self.args.y_norm == 'norm0':
                self.y_shift = np.zeros(self.actions.shape[-1])
                centered_data = self.actions          # no centering
                covariance_matrix = np.dot(centered_data.T, centered_data) / len(self.actions)
                if self.args.which_y == -1:
                    self.div = np.diag(1 / np.sqrt(np.diag(covariance_matrix)))
                    self.std = np.diag(np.sqrt(np.diag(covariance_matrix)))
                else:
                    self.div, self.std = 1/np.sqrt(covariance_matrix), np.sqrt(covariance_matrix)
            elif self.args.y_norm in ['std', 'std2']:
                self.y_shift = np.mean(self.actions, axis=0)
                centered_data = self.actions - self.y_shift
                covariance_matrix = np.dot(centered_data.T, centered_data) / len(self.actions)
                if self.args.which_y == -1:
                    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
                    self.div = eigenvectors @ np.diag(1 / np.sqrt(eigenvalues)) @ np.linalg.inv(eigenvectors)
                    self.std = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ np.linalg.inv(eigenvectors)
                else:
                    self.div, self.std = 1/np.sqrt(covariance_matrix), np.sqrt(covariance_matrix)
                if len(self.args.y_norm) > 3:
                        self.div = self.div * float(self.args.y_norm[3:])
                        self.std = self.std / float(self.args.y_norm[3:])
        else:  # test
            self.y_shift = y_shift
            self.div = div
            centered_data = self.actions - self.y_shift

        self.actions = centered_data @ self.div
        # self.actions = centered_data / self.div

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def _load_dataset(self, data_folder, env, split, data_ratio):
        file_name = '%s_%s.pkl' % (env, split)
        file_path = os.path.join(data_folder, file_name)
        try:
            with open(file_path, 'rb') as file:
                dataset = pickle.load(file)
                if data_ratio <= 1:
                    self.size = int(dataset['observations'].shape[0] * data_ratio)
                else:
                    self.size = int(data_ratio) if data_ratio<= dataset['observations'].shape[0] else dataset['observations'].shape[0]
                self.actions = dataset['actions'][:self.size, :]
            print('Successfully load dataset from: ', file_path)
            if self.args.which_y == -1:
                pass
            elif self.args.which_y >= 0:
                self.actions = self.actions[:, self.args.which_y].reshape(-1, 1)
        except Exception as e:
            print(e)
        self.states = self.actions.copy()

        self.state_dim = self.states.shape[1]
        self.action_dim = self.actions.shape[1]
        print(f"Dataset size: {self.size}; State Dim: {self.state_dim}; Action_Dim: {self.action_dim}.")

    def get_state_dim(self):
        return self.state_dim

    def get_action_dim(self):
        return self.action_dim

    def get_theory_stats(self, center=False):
        actions = self.actions
        mu = np.mean(actions, axis=0)
        if center:
            centered_actions = actions - mu
            Sigma = centered_actions.T @ centered_actions / centered_actions.shape[0]
        else:
            Sigma = actions.T @ actions / actions.shape[0]

        if self.args.which_y == -1:
            eig_vals, eig_vecs = np.linalg.eigh(Sigma)
            sqrt_eig_vals = np.sqrt(eig_vals)
            Sigma_sqrt = eig_vecs @ np.diag(sqrt_eig_vals) @ np.linalg.inv(eig_vecs)
            min_eigval, max_eigval = eig_vals[0], eig_vals[-1]
        else:
            Sigma_sqrt = np.sqrt(Sigma)
            min_eigval, max_eigval = Sigma_sqrt, Sigma_sqrt

        return {
            'mu': mu,
            'Sigma': Sigma,
            'Sigma_sqrt': Sigma_sqrt,
            'min_eigval': min_eigval,
            'max_eigval': max_eigval,
        }

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        states = self.states[idx]
        actions = self.actions[idx]
        return {
            'input': self._to_tensor(states),
            'target': self._to_tensor(actions)
        }