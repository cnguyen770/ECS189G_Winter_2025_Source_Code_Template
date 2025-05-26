import os
from torch.utils.data import Dataset
import torch
from glob import glob

class TextClassificationDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        root_dir/
          train/pos, train/neg, test/pos, test/neg
        split: 'train' or 'test'
        """
        self.samples = []
        for label, sentiment in enumerate(['neg','pos']):  # 0=neg,1=pos
            path = os.path.join(root_dir, split, sentiment)
            for filepath in glob(f"{path}/*.txt"):
                self.samples.append((filepath, label))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        with open(filepath, encoding='utf8') as f:
            text = f.read()
        if self.transform:
            text = self.transform(text)
        return text, label

class TextGenerationDataset(Dataset):
    def __init__(self, data_path):
        """
        data_path: either
          • the directory containing the file “data”, or
          • the path directly to the file “data”
        Each non‑empty line in that file is treated as one joke.
        """
        # determine the actual file
        if os.path.isdir(data_path):
            file_path = os.path.join(data_path, 'data')
        else:
            file_path = data_path

        # read all non‑blank lines
        self.jokes = []
        with open(file_path, encoding='utf8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.jokes.append(line)

    def __len__(self):
        return len(self.jokes)

    def __getitem__(self, idx):
        return self.jokes[idx]