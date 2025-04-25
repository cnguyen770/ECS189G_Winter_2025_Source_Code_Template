import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Dataset_Loader import CSVDigitDataset
from Method_MLP import MLP
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X.size(0)
    return running_loss / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            logits = model(X)
            preds = logits.argmax(dim=1).cpu().numpy()
            preds.extend(preds)
            labels.extend(y.numpy())
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )
    return acc, prec, rec, f1

def plot_loss(losses, save_path='loss_curve.png'):
    plt.figure()
    plt.plot(range(1, len(losses)+1), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('MLP Training Loss Curve')
    plt.savefig(save_path)
    print(f"Saved loss curve to {save_path}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train = DataLoader(CSVDigitDataset('stage_2_data/train.csv'),
                       batch_size=64, shuffle=True)
    test  = DataLoader(CSVDigitDataset('stage_2_data/test.csv'),
                       batch_size=256, shuffle=False)

    model = MLP().to(device)
    # model = MLP(hidden_dims=[256, 128, 64]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # optimizer = optim.SGD(model.parameters(), lr=1e-2)

    num_epochs = 10
    losses = []

    for epoch in range(num_epochs):
        loss = train_epoch(model, train, criterion, optimizer, device)
        losses.append(loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

    # save loss curve
    plot_loss(losses)

    # final evaluation
    acc, prec, rec, f1 = evaluate(model, test, device)
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

if __name__ == '__main__':
    main()
