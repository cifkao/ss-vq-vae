# Copyright 2020 InterDigital R&D and Télécom Paris.
# Author: Ondřej Cífka
# License: Apache 2.0

import confugue
import torch
from torch import nn


@confugue.configurable
class VQEmbedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, use_codebook_loss=True, axis=-1):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self._use_codebook_loss = use_codebook_loss
        self._cfg['init'].bind(nn.init.kaiming_uniform_)(self.embedding.weight)
        self._axis = axis

    def forward(self, input):
        if self._axis != -1:
            input = input.transpose(self._axis, -1)

        distances = (torch.sum(input ** 2, axis=-1, keepdim=True)
                     - 2 * torch.matmul(input, self.embedding.weight.T)
                     + torch.sum(self.embedding.weight ** 2, axis=-1))
        ids = torch.argmin(distances, axis=-1)
        quantized = self.embedding(ids)

        losses = {
            'commitment': ((quantized.detach() - input) ** 2).mean(axis=-1)
        }
        if self._use_codebook_loss:
            losses['codebook'] = ((quantized - input.detach()) ** 2).mean(axis=-1)

            # Straight-through gradient estimator as in the VQ-VAE paper
            # No gradient for the codebook
            quantized = (quantized - input).detach() + input
        else:
            # Modified straight-through gradient estimator
            # The gradient of the result gets copied to both inputs (quantized and non-quantized)
            quantized = input + quantized - input.detach()

        if self._axis != -1:
            quantized = quantized.transpose(self._axis, -1).contiguous()

        return quantized, ids, losses
