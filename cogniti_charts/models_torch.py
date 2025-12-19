import torch, torch.nn as nn

class TorchCNN1D(nn.Module):
    def __init__(self, num_features, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(num_features, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):       # x: [B, T, F]
        x = x.transpose(1, 2)   # -> [B, F, T]
        x = self.net(x)         # -> [B, 64, 1]
        return self.head(x)     # -> [B, C]
