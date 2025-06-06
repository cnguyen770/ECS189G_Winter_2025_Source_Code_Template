'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

import os
import torch
import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

class GraphNodeDataset(Dataset):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name.lower()
        self.folder_path = os.path.join(
           os.path.dirname(__file__),
           'stage_5_data', 'stage_5_data', self.dataset_name
        )
        assert os.path.isdir(self.folder_path), \
            f"Folder not found: {self.folder_path}"

        node_path = os.path.join(self.folder_path, 'node')
        data = np.genfromtxt(node_path, dtype=str)
        self.raw_ids = data[:, 0]
        features = data[:, 1:-1].astype(np.float32)
        labels_raw = data[:, -1]

        self.label_encoder = LabelEncoder()
        labels = self.label_encoder.fit_transform(labels_raw).astype(np.int64)

        self.features = torch.FloatTensor(features)  # [N, F]
        self.labels   = torch.LongTensor(labels)      # [N]

        link_path = os.path.join(self.folder_path, 'link')
        edges_unordered = np.genfromtxt(link_path, dtype=np.int32)
        idx_map = {int(self.raw_ids[i]): i for i in range(len(self.raw_ids))}
        edges = np.array(
            list(map(idx_map.get, edges_unordered.flatten())),
            dtype=np.int32
        ).reshape(edges_unordered.shape)
        N = labels.shape[0]
        adj = sp.coo_matrix(
            (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
            shape=(N, N),
            dtype=np.float32
        )
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = adj + sp.eye(N)

        self.adj = self._normalize_adj(adj)

    def _normalize_adj(self, adj):
        rowsum = np.array(adj.sum(1)).flatten()
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        D_inv_sqrt = sp.diags(d_inv_sqrt)
        return (D_inv_sqrt @ adj @ D_inv_sqrt).tocoo()

    def sparse_matrix_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
        )
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def get_torch_data(self):
        return self.features, \
               self.sparse_matrix_to_torch_sparse_tensor(self.adj), \
               self.labels

    def sample_splits(self, num_per_class_train, num_per_class_test, seed=42):
        np.random.seed(seed)
        labels = self.labels.numpy()
        classes = np.unique(labels)
        train_idx, test_idx = [], []

        for cls in classes:
            cls_nodes = np.where(labels == cls)[0]
            assert len(cls_nodes) >= (num_per_class_train + num_per_class_test), \
                f"Not enough nodes in class {cls} to sample"
            perm = np.random.permutation(cls_nodes)
            train_nodes = perm[:num_per_class_train]
            test_nodes  = perm[num_per_class_train:num_per_class_train+num_per_class_test]
            train_idx.extend(train_nodes.tolist())
            test_idx.extend(test_nodes.tolist())

        idx_train = torch.LongTensor(train_idx)
        idx_test  = torch.LongTensor(test_idx)
        return idx_train, idx_test