import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x, adj):
        x = torch.spmm(adj, x)
        return self.linear(x)


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        super(GCN, self).__init__()
        self.gc1 = GCNLayer(nfeat, nhid)
        self.gc2 = GCNLayer(nhid, nclass)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        h = self.gc1(x, adj)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.gc2(h, adj)
        return h
