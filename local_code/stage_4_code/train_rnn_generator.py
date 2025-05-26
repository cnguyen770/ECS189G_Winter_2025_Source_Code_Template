import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from Dataset_Loader import TextGenerationDataset
from Method_RNN import RNNGenerator
from preprocessing import clean_tokenize, Vocab

def collate_gen_fn(batch, vocab, seq_len=30):
    token_lists = [['<bos>'] + clean_tokenize(t) + ['<eos>'] for t in batch]
    idxs = [vocab.encode(tl)[:seq_len] for tl in token_lists]
    padded = [seq + [vocab.stoi['<pad>']]*(seq_len-len(seq)) for seq in idxs]
    inputs  = [seq[:-1] for seq in padded]
    targets = [seq[1:]  for seq in padded]
    return torch.tensor(inputs,  dtype=torch.long), \
           torch.tensor(targets, dtype=torch.long)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        logits, _ = model(X)
        loss = criterion(logits.view(-1, logits.size(-1)), Y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)

def plot_loss(losses, fname):
    plt.figure()
    plt.plot(range(1, len(losses)+1), losses)
    plt.xlabel('Epoch'); plt.ylabel('Training Loss'); plt.title('Generator Loss Curve')
    plt.savefig(fname); plt.close()
    print(f"Saved loss curve: {fname}")

def sample_sequence(model, start_tokens, vocab, max_len=30):
    model.eval()
    idxs = vocab.encode(start_tokens)
    input_seq = torch.tensor([idxs], dtype=torch.long).to(next(model.parameters()).device)
    hidden = None
    generated = start_tokens.copy()
    for _ in range(max_len - len(idxs)):
        logits, hidden = model(input_seq, hidden)
        probs = torch.softmax(logits[0, -1], dim=-1)
        nxt = torch.multinomial(probs, 1).item()
        tok = vocab.itos[nxt]
        if tok == '<eos>': break
        generated.append(tok)
        input_seq = torch.tensor([[nxt]], dtype=torch.long).to(input_seq.device)
    return ' '.join(generated)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='../stage_4_data/text_generation')
    p.add_argument('--rnn_type', choices=['RNN','LSTM','GRU'], default='RNN')
    p.add_argument('--emb_dim',    type=int, default=128)
    p.add_argument('--hidden_dim', type=int, default=128)
    p.add_argument('--num_layers', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--epochs',     type=int, default=10)
    p.add_argument('--lr',         type=float, default=1e-3)
    args = p.parse_args()

    ds = TextGenerationDataset(args.data_dir)
    token_lists = [['<bos>'] + clean_tokenize(j) + ['<eos>'] for j in ds.jokes]
    vocab = Vocab(token_lists)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda b: collate_gen_fn(b, vocab))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RNNGenerator(len(vocab.itos), args.emb_dim,
                         args.hidden_dim, args.num_layers,
                         rnn_type=args.rnn_type).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi['<pad>'])
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    losses = []
    for epoch in range(1, args.epochs+1):
        loss = train_one_epoch(model, loader, criterion, optimizer, device)
        losses.append(loss)
        print(f"Epoch {epoch}/{args.epochs}, Loss={loss:.4f}")

    plot_loss(losses, f"loss_curve_generator_{args.rnn_type}.png")

    examples = [['what','did','the'], ['why','did','the'], ['i','love','to']]
    for st in examples:
        print(f"Prompt: {' '.join(st)}")
        print("Generated:", sample_sequence(model, st, vocab))
        print()

if __name__=='__main__':
    main()
