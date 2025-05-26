import torch.nn as nn

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers, num_classes, rnn_type='RNN'):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        rnn_cls = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}[rnn_type]
        self.rnn = rnn_cls(emb_dim, hidden_dim, num_layers,
                           batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths=None):
        emb = self.embed(x)
        out, _ = self.rnn(emb)
        last = out[:, -1, :]
        return self.fc(last)


class RNNGenerator(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers, rnn_type='RNN'):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        rnn_cls = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}[rnn_type]
        self.rnn = rnn_cls(emb_dim, hidden_dim, num_layers,
                           batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embed(x)
        out, hidden = self.rnn(emb, hidden)
        logits = self.fc(out)
        return logits, hidden
