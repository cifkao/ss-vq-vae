import confugue
import torch
import torch.nn as nn


@confugue.configurable
class BilinearSimilarity(nn.Module):
    def __init__(self, projection_dim):
        super(BilinearSimilarity, self).__init__()
        # TODO: maybe a better random??
        self.W = nn.Parameter(torch.randn(projection_dim, projection_dim))

    def forward(self, gx, gx_prime):
        return gx.matmul(self.W).matmul(gx_prime.T)
