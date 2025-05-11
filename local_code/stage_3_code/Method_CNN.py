import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, in_channels, num_classes, input_size,
                 extra_conv=False, kernel_size=3, dropout=0.0):
        super().__init__()
        H, W = input_size
        ks, pad = kernel_size, kernel_size // 2
        layers = []

        # Block 1
        layers += [ nn.Conv2d(in_channels, 32, ks, padding=pad),
                    nn.ReLU(),
                    nn.MaxPool2d(2) ]
        if dropout>0: layers.append(nn.Dropout(dropout))

        # Block 2
        layers += [ nn.Conv2d(32, 64, ks, padding=pad),
                    nn.ReLU(),
                    nn.MaxPool2d(2) ]
        if dropout>0: layers.append(nn.Dropout(dropout))

        if extra_conv:
            layers += [ nn.Conv2d(64, 128, ks, padding=pad),
                        nn.ReLU(),
                        nn.MaxPool2d(2) ]
            if dropout>0: layers.append(nn.Dropout(dropout))

        self.features = nn.Sequential(*layers)

        # compute flattened size
        factor = 4 if not extra_conv else 8
        channels = 64 if not extra_conv else 128
        flat_dim = channels * (H//factor) * (W//factor)

        # classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))
