import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=[128, 64], num_classes=10):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            # layers.append(nn.Dropout(0.5))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
