# stage_5_code/train_gcn.py

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt

from Dataset_Loader_Node_Classification import GraphNodeDataset
from Method_GCN import GCN

def train(model, features, adj, labels, idx_train, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    outputs = model(features, adj)
    loss = criterion(outputs[idx_train], labels[idx_train])
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, features, adj, labels, idx_test):
    model.eval()
    with torch.no_grad():
        logits = model(features, adj)
        preds = logits[idx_test].argmax(dim=1).cpu().numpy()
        trues = labels[idx_test].cpu().numpy()
    acc = accuracy_score(trues, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        trues, preds, average='weighted'
    )
    return acc, prec, rec, f1

def plot_loss(losses, save_path):
    plt.figure()
    plt.plot(range(1, len(losses)+1), losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('GCN Training Loss')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved loss curve: {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['cora','citeseer','pubmed'], required=True)
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    dataset = GraphNodeDataset(args.dataset)
    features, adj, labels = dataset.get_torch_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features = features.to(device)
    adj      = adj.to(device)
    labels   = labels.to(device)

    if args.dataset == 'cora':
        train_per_class, test_per_class = 20, 150
    elif args.dataset == 'citeseer':
        train_per_class, test_per_class = 20, 200
    else:
        train_per_class, test_per_class = 20, 200

    idx_train, idx_test = dataset.sample_splits(train_per_class, test_per_class, seed=args.seed)
    idx_train = idx_train.to(device)
    idx_test  = idx_test.to(device)

    nfeat   = features.shape[1]
    nclass  = int(labels.max().item()) + 1
    model = GCN(nfeat=nfeat, nhid=args.hidden_dim,
                nclass=nclass, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)

    losses = []
    for epoch in range(1, args.epochs + 1):
        loss = train(model, features, adj, labels, idx_train, optimizer, criterion)
        losses.append(loss)
        if epoch % 20 == 0 or epoch == 1:
            print(f"{args.dataset.upper()} Epoch {epoch}/{args.epochs}, Loss={loss:.4f}")

    plot_fname = f"loss_curve_{args.dataset}.png"
    plot_loss(losses, plot_fname)

    acc, prec, rec, f1 = test(model, features, adj, labels, idx_test)
    print(f"{args.dataset.upper()} Test Performance >> "
          f"Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

if __name__ == '__main__':
    main()
