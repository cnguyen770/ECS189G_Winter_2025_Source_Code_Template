import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt

from Dataset_Loader import TextClassificationDataset
from Method_RNN import RNNClassifier
from preprocessing import clean_tokenize, Vocab

def collate_fn(batch, vocab, max_len=200):
    texts, labels = zip(*batch)
    token_lists = [clean_tokenize(t) for t in texts]
    idxs = [vocab.encode(tl)[:max_len] for tl in token_lists]
    padded = [seq + [vocab.stoi['<pad>']] * (max_len - len(seq)) for seq in idxs]
    return torch.tensor(padded, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

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
    prec, rec, f1, _ = precision_recall_fscore_support(trues, preds, average='weighted')
    return acc, prec, rec, f1

def plot_loss(losses, fname):
    plt.figure()
    plt.plot(range(1, len(losses)+1), losses)
    plt.xlabel('Epoch'); plt.ylabel('Training Loss'); plt.title('Classifier Loss Curve')
    plt.savefig(fname); plt.close()
    print(f"Saved loss curve: {fname}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',   default='../stage_4_data/text_classification')
    p.add_argument('--rnn_type',   choices=['RNN','LSTM','GRU'], default='RNN')
    p.add_argument('--emb_dim',    type=int, default=128)
    p.add_argument('--hidden_dim', type=int, default=128)
    p.add_argument('--num_layers', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--epochs',     type=int, default=10)
    p.add_argument('--lr',         type=float, default=1e-3)
    args = p.parse_args()

    train_ds = TextClassificationDataset(args.data_dir, split='train')
    test_ds  = TextClassificationDataset(args.data_dir, split='test')

    all_texts   = [t for t,_ in train_ds]
    token_lists = [clean_tokenize(t) for t in all_texts]
    vocab       = Vocab(token_lists)

    collate = lambda b: collate_fn(b, vocab)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  collate_fn=collate)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = RNNClassifier(len(vocab.itos), args.emb_dim,
                           args.hidden_dim, args.num_layers,
                           num_classes=2, rnn_type=args.rnn_type).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    losses = []
    for epoch in range(1, args.epochs+1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        losses.append(loss)
        print(f"Epoch {epoch}/{args.epochs}, Loss={loss:.4f}")

    plot_loss(losses, f"loss_curve_classifier_{args.rnn_type}.png")
    acc, prec, rec, f1 = evaluate(model, test_loader, device)
    print(f"{args.rnn_type} Classifier >> Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

if __name__ == '__main__':
    main()
