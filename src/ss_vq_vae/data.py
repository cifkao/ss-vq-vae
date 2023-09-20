# Copyright 2020 InterDigital R&D and Télécom Paris.
# Author: Ondřej Cífka
# License: Apache 2.0

import os
import random

import librosa
from tqdm import tqdm
import torch

from . import util


class AudioTupleDataset(torch.utils.data.Dataset):

    def __init__(self, path, sr=22050, lazy=True, preprocess_fn=None,
                 mel_preprocess_fn=None,
                 sample_size=None, seed=42, no_ground=True):
        self._sr = sr
        self._lazy = lazy
        self._preprocess_fn = preprocess_fn
        self._mel_preprocess_fn = mel_preprocess_fn
        self._path_tuples = []
        self._no_ground = no_ground

        data_path = os.path.dirname(path)
        with open(path) as f:
            for line in f:
                path_tuple = line.rstrip('\n').split('\t')
                path_tuple = tuple(os.path.join(data_path, p) for p in path_tuple)
                self._path_tuples.append(path_tuple)

        if sample_size:
            self._path_tuples = random.Random(seed).sample(
                self._path_tuples, sample_size)

        if not self._lazy:
            self._data = [self._load(i) for i in tqdm(range(len(self)))]

    def __getitem__(self, index):
        if self._lazy:
            return self._load(index)
        else:
            return self._data[index]

    def _load(self, index):
        audios = [librosa.load(path, sr=self._sr)[0]
                  for path in self._path_tuples[index]]
        
        style_mels = None
        if self._mel_preprocess_fn is not None:
            style_mels = self._mel_preprocess_fn(y=audios[1])
        if self._preprocess_fn is not None:
            audios = [self._preprocess_fn(audio) for audio in audios]
            
        if self._no_ground:
            return tuple([audios[0], style_mels if style_mels is not None else audios[1]])
        else:
            return tuple([audios[0], style_mels if style_mels is not None else audios[1], audios[2]])

    def __len__(self):
        return len(self._path_tuples)