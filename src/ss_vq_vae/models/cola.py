import torch
from torch import nn
from nn import EfficientNetEmbedding, EfficientNetType


class COLA(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(COLA, self).__init__()
        self.encoder = EfficientNetEmbedding(EfficientNetType.EFFICIENTNET_B0, hidden_size)
        self.projection = nn.Linear(hidden_size, output_size)
        self.layernorm = nn.LayerNorm(output_size)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(1).repeat(1, 3, 1, 1)
        x = self.encoder(x)
        x = self.projection(x)
        x = self.layernorm(x)
        x = self.tanh(x)

        return x
