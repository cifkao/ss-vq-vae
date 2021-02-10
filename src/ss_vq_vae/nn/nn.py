# Copyright 2020 InterDigital R&D and Télécom Paris.
# Author: Ondřej Cífka
# License: Apache 2.0

import confugue
import torch
from torch import nn


@confugue.configurable(params=['return_state'])
class RNNWrapper(nn.Module):
    """Wrapper for plugging an RNN into a CNN."""

    def __init__(self, rnn=None, return_state=False, return_output=None):
        super().__init__()
        if rnn is not None:
            self.rnn = rnn
        else:
            self.rnn = self._cfg['rnn'].configure(batch_first=True)
        self.return_state = return_state
        self.return_output = return_output or not return_state

    def forward(self, input):
        output, state = self.rnn(input.transpose(1, 2))
        output = output.transpose(1, 2)
        state = state.transpose(0, 1).reshape(input.shape[0], -1)
        if self.return_output and self.return_state:
            return output, state
        if self.return_state:
            return state
        return output


@confugue.configurable(params=[])
class ResidualWrapper(nn.Module):
    """Wrapper for adding a skip connection around a module."""

    def __init__(self, module=None):
        super().__init__()
        if module is not None:
            self.module = module
        elif 'module' in self._cfg:
            self.module = self._cfg['module'].configure()
        elif 'modules' in self._cfg:
            self.module = nn.Sequential(*self._cfg['modules'].configure_list())

    def forward(self, input):
        output = self.module(input)
        if output.shape != input.shape:
            raise RuntimeError(f'Expected output to have shape {input.shape}, got {output.shape}')
        return output + input
