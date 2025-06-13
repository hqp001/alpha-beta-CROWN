import torch
import torch.nn as nn

class FlexibleMLP(nn.Module):
    def __init__(self, n_pixel_1d: int, layer_size: int, n_layers: int):
        super().__init__()
        layers = []

        layers.append(nn.Flatten())
        layers.append(nn.Linear(n_pixel_1d ** 2, layer_size))
        layers.append(nn.ReLU())

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(layer_size, layer_size))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(layer_size, 10))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

def model_28_4():
    return FlexibleMLP(28, 64, 4)
