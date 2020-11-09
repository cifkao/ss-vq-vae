# Copyright 2020 InterDigital R&D and Télécom Paris.
# Author: Ondřej Cífka
# License: TBD

"""Self-supervised VQ-VAE for one-shot style transfer."""

import argparse
import logging
import os
import random

import confugue
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import torch
import torch.utils.tensorboard
from torch import nn

from ..nn import VQEmbedding
from ..data import AudioTupleDataset
from .. import util


LOGGER = logging.getLogger('model')


@confugue.configurable
class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.content_encoder = nn.Sequential(*self._cfg['content_encoder'].configure_list())
        self.vq = self._cfg['vq'].configure(VQEmbedding, axis=1)

        self.style_encoder_1d = nn.Sequential(*self._cfg['style_encoder_1d'].configure_list())
        self.style_encoder_rnn = self._cfg['style_encoder_rnn'].maybe_configure(nn.GRU,
                                                                                batch_first=True)
        self.style_encoder_0d = nn.Sequential(*self._cfg['style_encoder_0d'].configure_list())

        self.decoder_modules = nn.ModuleList([
            nn.Sequential(*self._cfg['decoder'][i].configure_list())
            for i in range(len(self._cfg['decoder']))
        ])

    def forward(self, input_c, input_s, length_c, length_s, return_losses=False):
        encoded_c, _, losses_c = self.encode_content(input_c)
        encoded_s, losses_s = self.encode_style(input_s, length_s)
        decoded = self.decode(encoded_c, encoded_s, length=length_c, max_length=input_c.shape[2])

        if not return_losses:
            return decoded

        losses = {
            'reconstruction': ((decoded - input_c) ** 2).mean(axis=1),
            **losses_c
        }

        # Sum losses over time and batch, normalize by total time
        assert all(len(loss.shape) == 2 for loss in losses.values())
        losses = {name: loss.sum() / (length_c.sum() + torch.finfo(loss.dtype).eps)
                  for name, loss in losses.items()}

        # Add losses which don't have the time dimension
        assert all(len(loss.shape) == 1 for loss in losses_s.values())
        losses.update({name: loss.mean() for name, loss in losses_s.items()})

        return decoded, losses

    def encode_content(self, input):
        encoded = self.content_encoder(input)
        if self.vq is None:
            return encoded, encoded, {}
        return self.vq(encoded)

    def encode_style(self, input, length):
        encoded = self.style_encoder_1d(input)

        # Mask positions corresponding to padding
        length = (length // (input.shape[2] / encoded.shape[2])).to(torch.int)
        mask = (torch.arange(encoded.shape[2], device=encoded.device) < length[:, None])[:, None, :]
        encoded *= mask

        if self.style_encoder_rnn is not None:
            encoded = encoded.transpose(1, 2)
            encoded = nn.utils.rnn.pack_padded_sequence(
                encoded, length.clamp(min=1),
                batch_first=True, enforce_sorted=False)
            _, encoded = self.style_encoder_rnn(encoded)
            # Get rid of layer dimension
            encoded = encoded.transpose(0, 1).reshape(input.shape[0], -1)
        else:
            # Compute the Gram matrix, normalized by the length squared
            encoded /= mask.sum(dim=2, keepdim=True) + torch.finfo(encoded.dtype).eps
            encoded = torch.matmul(encoded, encoded.transpose(1, 2))
        encoded = encoded.reshape(encoded.shape[0], -1)

        encoded = self.style_encoder_0d(encoded)

        return encoded, {}

    def decode(self, encoded_c, encoded_s, length=None, max_length=None):
        encoded_s = encoded_s[:, :, None]

        decoded = encoded_c
        for module in self.decoder_modules:
            decoded = torch.cat([
                decoded,
                encoded_s.expand(-1, -1, decoded.shape[-1])
            ], axis=1)
            decoded = module(decoded)

        # Make sure the output tensor has the same shape as the input tensor
        if max_length is not None or length is not None:
            if max_length is None:
                max_length = length.max()

            decoded = decoded.narrow(-1, 0, max_length)

        # Make sure output lengths are the same as input lengths
        if length is not None:
            mask = (torch.arange(max_length, device=decoded.device) < length[:, None])[:, None, :]
            decoded *= mask

        return decoded


@confugue.configurable
class Experiment:

    def __init__(self, logdir, config_path=None, device='cuda', sr=22050):
        self.logdir = logdir
        self.config_path = config_path
        self.sr = sr

        self._spec_fn = self._cfg['spectrogram'].bind(librosa.stft)
        self._inv_spec_fn = self._cfg['invert_spectrogram'].bind(
            librosa.griffinlim, random_state=0)

        self.model = self._cfg['model'].configure(Model)
        LOGGER.info(self.model)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.optimizer = None

    def train(self):
        with torch.utils.tensorboard.SummaryWriter(log_dir=self.logdir) as tb_writer:
            if self.config_path:
                with open(self.config_path) as f:
                    tb_writer.add_text('config', util.markdown_format_code(f.read()))
            tb_writer.add_text('model', util.markdown_format_code(repr(self.model)))

            self.model.train(True)
            if not self.optimizer:
                self.optimizer = self._cfg['optimizer'].configure(
                    torch.optim.Adam, params=self.model.parameters())

            loader_train = self._cfg['train_loader'].configure(
                torch.utils.data.DataLoader,
                dataset=self._get_dataset('train'),
                collate_fn=util.collate_padded_tuples,
                shuffle=True)
            loader_val = self._cfg['val_loader'].configure(
                torch.utils.data.DataLoader,
                dataset=self._get_dataset('val', lazy=False),
                collate_fn=util.collate_padded_tuples)

            num_epochs = self._cfg.get('epochs', 1)
            val_period = self._cfg.get('val_period', np.nan)
            log_period = self._cfg.get('log_period', 1)
            sample_period = self._cfg.get('sample_period', 1)

            i = 0
            for epoch in range(num_epochs):
                LOGGER.info('Starting epoch %d / %d', epoch + 1, num_epochs)
                for (input_c, length_c), (input_s, length_s) in loader_train:
                    input_c, length_c, input_s, length_s = (
                        x.to(self.device) for x in (input_c, length_c, input_s, length_s))

                    # Validation
                    if i % val_period == 0:
                        with torch.no_grad():
                            LOGGER.info('Starting validation at step %d', i)
                            self._validate(loader=loader_val, tb_writer=tb_writer, step=i,
                                           write_samples=(i // val_period) % sample_period == 0,
                                           write_model=True)
                            LOGGER.info('Validation done')

                    # Forward pass
                    if not self.model.training:
                        self.model.train(True)
                    _, losses = self.model(input_c, input_s, length_c, length_s, return_losses=True)
                    self._add_total_loss(losses, step=i)

                    # Logging
                    if i % log_period == 0:
                        for name, loss in losses.items():
                            tb_writer.add_scalar(f'loss_train/{name}', loss, i)

                        for dev_idx in range(torch.cuda.device_count()):
                            tb_writer.add_scalar(
                                f'mem/cuda:{dev_idx}', torch.cuda.memory_allocated(dev_idx), i)

                    # Backward pass
                    self.optimizer.zero_grad()
                    losses['total'].backward()
                    if i % log_period == 0:
                        self._log_params(tb_writer, i)
                    self.optimizer.step()

                    i += 1

                LOGGER.info('Epoch %d finished (%d steps)', epoch + 1, i)

    def run(self, data_loader):
        self.model.train(False)
        all_outputs, all_losses = [], []
        input_device = None
        for (input_c, length_c), (input_s, length_s) in data_loader:
            output, losses = self.model(
                input_c.to(self.device), input_s.to(self.device),
                length_c.to(self.device), length_s.to(self.device),
                return_losses=True)
            all_losses.append(losses)
            all_outputs.extend(output.to(input_c.device))

        all_losses = {name: torch.mean(torch.stack([x[name] for x in all_losses])).to(input_device)
                      for name in all_losses[0]}
        self._add_total_loss(all_losses)
        return all_outputs, all_losses

    def run_files(self, pairs_path, output_list_path, output_prefix, batch_size=None):
        loader_fn = self._cfg['val_loader'].bind(
            torch.utils.data.DataLoader,
            dataset=AudioTupleDataset(
                path=pairs_path, sr=self.sr, preprocess_fn=self.preprocess, lazy=False),
            collate_fn=util.collate_padded_tuples)
        loader = loader_fn(**(dict(batch_size=batch_size) if batch_size else {}))

        self.model.load_state_dict(torch.load(os.path.join(self.logdir, 'model_state.pt')))
        with torch.no_grad():
            outputs, _ = self.run(loader)

        # The output_list_path file will contain paths relative to parent_dir.
        parent_dir = os.path.dirname(output_list_path)
        with open(output_list_path, 'w') as f_triplets:
            i_len = len(str(len(outputs) - 1))
            for i, s_output in enumerate(outputs):
                a_output = self.postprocess(s_output.numpy())
                p_output = output_prefix + str(i).zfill(i_len) + '.wav'
                p_output_full = os.path.join(parent_dir, p_output)
                os.makedirs(os.path.dirname(p_output_full), exist_ok=True)
                sf.write(p_output_full, a_output, samplerate=self.sr)
                print(p_output, file=f_triplets)

    def _validate(self, loader, tb_writer=None, step=None,
                  write_losses=True, write_samples=False, write_model=False):
        outputs, losses = self.run(loader)
        if write_losses and tb_writer:
            for name, loss in losses.items():
                tb_writer.add_scalar(f'loss_valid/{name}', loss, step)

        if write_samples:
            num_samples = self._cfg.get('num_val_samples', 4)
            sample_ids = random.Random(42).sample(range(len(outputs)), num_samples)
            for j in sample_ids:
                output = outputs[j].numpy()

                fig = plt.figure()
                librosa.display.specshow(output, figure=fig)
                tb_writer.add_figure(f'ex/{j}/spec', fig, global_step=step)

                audio = self.postprocess(output)
                tb_writer.add_audio(f'ex/{j}/audio', audio, sample_rate=self.sr,
                                    global_step=step)

        if write_model:
            torch.save(self.model, os.path.join(self.logdir, 'model.pt'))
            torch.save(self.model.state_dict(), os.path.join(self.logdir, 'model_state.pt'))

        return outputs, losses

    def _log_params(self, tb_writer, step):
        for name, param in self.model.named_parameters():
            try:
                tb_writer.add_histogram(f'params/{name}', param, step)
            except ValueError:
                LOGGER.exception('Error while logging "%s"', name)
                LOGGER.info(param)
            if param.grad is not None:
                try:
                    tb_writer.add_histogram(f'grads/{name}', param.grad, step)
                except ValueError:
                    LOGGER.exception('Error while logging "%s" gradient', name)
                    LOGGER.info(param.grad)

    def _add_total_loss(self, losses, step=None):
        beta_c = self._cfg.get('commitment_loss_weight')
        beta_s = self._cfg.get('style_kl_weight', 1.)
        beta_s_anneal_start = self._cfg.get('style_kl_anneal_start', 0)
        beta_s_anneal_steps = self._cfg.get('style_kl_anneal_steps', 0)
        if step is not None:
            if beta_s_anneal_steps == 0:
                beta_s = 0. if step < beta_s_anneal_start else beta_s
            else:
                beta_s *= min(1., max(0., (step - beta_s_anneal_start) / beta_s_anneal_steps))

        losses['total'] = sum(loss for name, loss in losses.items()
                              if name not in ['commitment', 'style_kl'])
        losses['total'] += beta_c * losses.get('commitment', 0.)
        losses['total'] += beta_s * losses.get('style_kl', 0.)
        return losses

    def preprocess(self, audio):
        if len(audio) == 0:
            audio = np.zeros(shape=[1], dtype=audio.dtype)
        return np.log1p(np.abs(self._spec_fn(y=audio)))

    def postprocess(self, spectrogram):
        return self._inv_spec_fn(S=np.expm1(np.maximum(spectrogram, 0)))

    def _get_dataset(self, section, **kwargs):
        return self._cfg['data'][section].configure(
            AudioTupleDataset, sr=self.sr, preprocess_fn=self.preprocess,
            **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--logdir', type=str, required=True)
    actions = parser.add_subparsers(title='action')
    train_parser = actions.add_parser('train')
    train_parser.set_defaults(action='train')
    run_parser = actions.add_parser('run')
    run_parser.set_defaults(action='run')
    run_parser.add_argument('pairs_path', metavar='PAIRS_FILE')
    run_parser.add_argument('output_list_path', metavar='OUT_LIST_FILE')
    run_parser.add_argument('output_prefix', metavar='OUTPUT_PREFIX')
    run_parser.add_argument('--batch-size', type=int, metavar='SIZE')
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    cfg_path = os.path.join(args.logdir, 'config.yaml')
    cfg = confugue.Configuration.from_yaml_file(cfg_path)
    exp = cfg.configure(Experiment, config_path=cfg_path, logdir=args.logdir)
    if args.action == 'train':
        exp.train()
    elif args.action == 'run':
        exp.run_files(pairs_path=args.pairs_path, output_list_path=args.output_list_path,
                      output_prefix=args.output_prefix, batch_size=args.batch_size)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s')
    main()
