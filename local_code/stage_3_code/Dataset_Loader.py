import os, pickle
import torch
from torch.utils.data import Dataset

class ORLDataset(Dataset):
    def __init__(self, data_dir, train=True):
        with open(os.path.join(data_dir, 'ORL'), 'rb') as f:
            data = pickle.load(f)
        split = 'train' if train else 'test'
        self.images = [inst['image'] for inst in data[split]]
        self.labels = [inst['label'] - 1 for inst in data[split]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = torch.tensor(self.images[idx], dtype=torch.float32)
        img = img[:, :, 0].unsqueeze(0) / 255.0
        lbl = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, lbl


class MNISTDataset(Dataset):
    def __init__(self, data_dir, train=True):
        with open(os.path.join(data_dir, 'MNIST'), 'rb') as f:
            data = pickle.load(f)
        split = 'train' if train else 'test'
        self.images = [inst['image'] for inst in data[split]]
        self.labels = [inst['label'] for inst in data[split]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = torch.tensor(self.images[idx], dtype=torch.float32).unsqueeze(0) / 255.0
        lbl = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, lbl


class CIFARDataset(Dataset):
    def __init__(self, data_dir, train=True):
        with open(os.path.join(data_dir, 'CIFAR'), 'rb') as f:
            data = pickle.load(f)
        split = 'train' if train else 'test'
        self.images = [inst['image'] for inst in data[split]]
        self.labels = [inst['label'] for inst in data[split]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = torch.tensor(self.images[idx], dtype=torch.float32).permute(2, 0, 1) / 255.0
        lbl = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, lbl
