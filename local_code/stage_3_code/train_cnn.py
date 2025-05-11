import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt

from Dataset_Loader import ORLDataset, MNISTDataset, CIFARDataset
from Method_CNN import SimpleCNN

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            logits = model(X)
            p = logits.argmax(dim=1).cpu().numpy()
            preds.extend(p)
            trues.extend(y.numpy())
    acc = accuracy_score(trues, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        trues, preds, average='weighted'
    )
    return acc, prec, rec, f1

def plot_loss(losses, filename):
    plt.figure()
    plt.plot(range(1, len(losses)+1), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss vs. Epoch')
    plt.savefig(filename)
    plt.close()
    print(f"Saved loss curve: {filename}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['orl','mnist','cifar'], required=True)
    p.add_argument('--data_dir', default='stage_3_data')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-3)
    # Ablation flags
    p.add_argument('--extra_conv', action='store_true',
                  help="Add a 3rd conv block (B1)")
    p.add_argument('--kernel_size', type=int, default=3,
                  help="Conv kernel size (3 baseline, 5 for B2)")
    p.add_argument('--dropout', type=float, default=0.0,
                  help="Dropout prob after each pool (B3)")
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset selection
    if args.dataset == 'orl':
        train_ds = ORLDataset(args.data_dir, train=True)
        test_ds  = ORLDataset(args.data_dir, train=False)
        in_ch, H, W, nc = 1, 112, 92, 40
    elif args.dataset == 'mnist':
        train_ds = MNISTDataset(args.data_dir, train=True)
        test_ds  = MNISTDataset(args.data_dir, train=False)
        in_ch, H, W, nc = 1, 28, 28, 10
    else:
        train_ds = CIFARDataset(args.data_dir, train=True)
        test_ds  = CIFARDataset(args.data_dir, train=False)
        in_ch, H, W, nc = 3, 32, 32, 10

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)

    model = SimpleCNN(in_ch, nc, (H, W),
                      extra_conv=args.extra_conv,
                      kernel_size=args.kernel_size,
                      dropout=args.dropout).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    losses = []
    for epoch in range(1, args.epochs+1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        losses.append(loss)
        print(f"Epoch {epoch}/{args.epochs}, Loss={loss:.4f}")

    # Save plot and evaluate
    plot_loss(losses, f"loss_curve_{args.dataset}.png")
    acc, prec, rec, f1 = evaluate(model, test_loader, device)
    print(f"{args.dataset.upper()} Metrics >> Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

if __name__ == '__main__':
    main()
