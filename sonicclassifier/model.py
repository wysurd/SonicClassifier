import torch
from torch import nn


class GenreCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        layers = []
        channels = 1
        for width in (16, 32, 64):
            layers.extend([nn.Conv2d(channels, width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)])
            channels = width
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.25), nn.Linear(64, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def get_device(choice='auto'):
    if choice == 'auto':
        return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    if choice not in ('cpu', 'mps'):
        raise ValueError('Device must be auto, cpu, or mps')
    if choice == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS is unavailable; use cpu or auto')
    return torch.device(choice)
