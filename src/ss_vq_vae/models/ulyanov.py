# Copyright 2020 InterDigital R&D and Télécom Paris.
# Author: Ondřej Cífka
# License: Apache 2.0

"""An implementation of the algorithm of Ulyanov and Lebedev.

Original blog post: https://dmitryulyanov.github.io/audio-texture-synthesis-and-style-transfer/
"""

import argparse
import os

import librosa
import numpy as np
import soundfile as sf
import torch
from torch import nn

from ..data import AudioTupleDataset
from .. import util


N_BINS = 1025
N_FILTERS = 4096
OUTPUT_CLIP_MAX = 40


def spectrogram(x):
    s = np.abs(librosa.stft(x)) ** 2
    s /= np.mean(s) + np.finfo(s.dtype).eps
    s = np.log1p(s)
    return s


def invert_spectrogram(s):
    s = np.clip(np.nan_to_num(s), a_min=0, a_max=OUTPUT_CLIP_MAX)
    a = librosa.griffinlim(np.sqrt(np.expm1(s)), random_state=0)
    a = librosa.util.normalize(a) * 0.99
    return a


def style_repr(features):
    return 0.5 * features @ features.transpose(-1, -2) / features.shape[2]


def l2_loss(features, features_target):
    l2 = (features - features_target) ** 2
    return 0.5 * torch.sum(l2)


def apply_mask(features, length):
    mask = (torch.arange(features.shape[2], device=features.device) < length[:, None])[:, None, :]
    return features * mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pairs_path', metavar='PAIRS_FILE')
    parser.add_argument('output_list_path', metavar='OUT_LIST_FILE')
    parser.add_argument('output_prefix', metavar='OUTPUT_PREFIX')
    parser.add_argument('--sr', type=int, default=16000)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--content-weight', type=float, default=1.)
    parser.add_argument('--content-weight-log', type=lambda x: 10 ** float(x), dest='content_weight')
    parser.add_argument('--style-weight', type=float, default=1.)
    parser.add_argument('--style-weight-log', type=lambda x: 10 ** float(x), dest='style_weight')
    args = parser.parse_args()

    dataset = AudioTupleDataset(
        path=args.pairs_path,
        sr=args.sr, preprocess_fn=spectrogram, lazy=False)
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        collate_fn=util.collate_padded_tuples,
        batch_size=args.batch_size)

    net = nn.Sequential(
        nn.Conv1d(in_channels=N_BINS,
                  out_channels=N_FILTERS,
                  kernel_size=11,
                  bias=False),
        nn.ReLU(),
    )
    net.requires_grad_(False)
    net.cuda()
    torch.manual_seed(0)
    conv_layer = net[0]
    nn.init.normal_(conv_layer.weight, std=np.sqrt(2) * np.sqrt(2.0 / ((N_BINS + N_FILTERS) * 11)))

    def style_transfer(s_content, s_style, content_len, style_len, content_weight, style_weight):
        # Normalize the weights so that they sum up to one
        content_weight, style_weight = np.array([content_weight, style_weight]) / (content_weight + style_weight)

        s_content, content_len, s_style, style_len = (x.cuda() for x in [s_content, content_len, s_style, style_len])
        f_content_len = content_len - conv_layer.kernel_size[0] + 1
        f_style_len = style_len - conv_layer.kernel_size[0] + 1
        f_content = apply_mask(net(s_content), f_content_len)
        f_style = style_repr(apply_mask(net(s_style), f_style_len))

        s_output = torch.zeros_like(s_content)
        torch.manual_seed(0)
        s_output.normal_(std=1e-3)
        s_output.requires_grad_()

        optimizer = torch.optim.LBFGS([s_output], lr=1, tolerance_grad=1e-05, max_iter=10)
        for _ in range(10):
            def closure():
                s_output_masked = apply_mask(s_output, content_len)
                f_output_content = apply_mask(net(s_output_masked), f_content_len)
                f_output_style = style_repr(f_output_content)
                loss = (content_weight * l2_loss(f_output_content, f_content) +
                        style_weight * l2_loss(f_output_style, f_style))
                optimizer.zero_grad()
                loss.backward()
                return loss
            optimizer.step(closure)
        return s_output.detach().cpu()

    # The output_list_path file will contain paths relative to parent_dir.
    parent_dir = os.path.dirname(args.output_list_path)
    i_len = len(str(len(dataset) - 1))
    i = 0

    with open(args.output_list_path, 'w') as f_triplets:
        for (s_content, content_len), (s_style, style_len) in loader:
            s_outputs = style_transfer(s_content, s_style, content_len, style_len,
                                       args.content_weight, args.style_weight)
            for s_output in s_outputs.numpy():
                a_output = invert_spectrogram(s_output)

                p_output = args.output_prefix + str(i).zfill(i_len) + '.wav'
                p_output_full = os.path.join(parent_dir, p_output)
                os.makedirs(os.path.dirname(p_output_full), exist_ok=True)
                sf.write(p_output_full, a_output, samplerate=args.sr)
                print(p_output, file=f_triplets)

                i += 1


if __name__ == '__main__':
    main()
