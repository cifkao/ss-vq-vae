import torch
import torch.nn as nn
from models.encoder import Encoder, EncoderBackend


class COLA(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(COLA, self).__init__()
        self.encoder = Encoder(EncoderBackend.EFFICIENTNET_B0, hidden_size)
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