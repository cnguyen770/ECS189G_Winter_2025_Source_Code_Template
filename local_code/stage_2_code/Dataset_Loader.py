import torch
from torch.utils.data import Dataset
import pandas as pd

class CSVDigitDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path, header=None)
        data = df.values
        self.labels = torch.tensor(data[:, 0], dtype=torch.long)
        self.features = torch.tensor(data[:, 1:], dtype=torch.float32) / 255.0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
