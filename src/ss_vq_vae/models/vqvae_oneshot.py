# Copyright 2020 InterDigital R&D and Télécom Paris.
# Author: Ondřej Cífka
# License: Apache 2.0

"""Self-supervised VQ-VAE for one-shot style transfer."""

import argparse
import logging
import os
import random

import confugue
import librosa
import librosa.display
import librosa.feature
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import torch
import torch.utils.tensorboard
from torch import nn
import wandb

from .cola import COLA
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
class ModelZeroShot(nn.Module):
    def __init__(self):
        super().__init__()
        self.content_encoder = nn.Sequential(*self._cfg['content_encoder'].configure_list())
        self.vq = self._cfg['vq'].configure(VQEmbedding, axis=1)

        self.style_encoder = COLA(1280, 1024)
        self.decoder_modules = nn.ModuleList([
            nn.Sequential(*self._cfg['decoder'][i].configure_list())
            for i in range(len(self._cfg['decoder']))
        ])

    def forward(self, input_c, input_s, length_c, length_s, output_c=None, output_l=None, return_losses=False):
        encoded_c, _, losses_c = self.encode_content(input_c)
        with torch.no_grad():
            encoded_s, losses_s = self.encode_style(input_s, length_s)
        decoded = self.decode(encoded_c, encoded_s, length=length_c, max_length=input_c.shape[2])
        
        # Ensure the tensors have the same size
        if output_c is not None and decoded.size(-1) != output_c.size(-1):
            max_len = max(decoded.size(-1), output_c.size(-1))
            decoded = torch.nn.functional.pad(decoded, (0, max_len - decoded.size(-1))).to(device='cuda')
            output_c = torch.nn.functional.pad(output_c, (0, max_len - output_c.size(-1))).to(device='cuda')

        if not return_losses:
            return decoded

        losses = {
            'reconstruction': ((decoded - (output_c if output_c is not None else input_c)) ** 2).mean(axis=1),
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
        encoded = self.style_encoder(input)
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
            decoded = decoded * mask

        return decoded
    
@confugue.configurable
class ModelZeroShotUnfrozenStyle(ModelZeroShot):
    def forward(self, input_c, input_s, length_c, length_s, output_c=None, output_l=None, return_losses=False):
        encoded_c, _, losses_c = self.encode_content(input_c)
        encoded_s, losses_s = self.encode_style(input_s, length_s)
        decoded = self.decode(encoded_c, encoded_s, length=length_c, max_length=input_c.shape[2])
        
        # Ensure the tensors have the same size
        if output_c is not None and decoded.size(-1) != output_c.size(-1):
            max_len = max(decoded.size(-1), output_c.size(-1))
            decoded = torch.nn.functional.pad(decoded, (0, max_len - decoded.size(-1))).to(device='cuda')
            output_c = torch.nn.functional.pad(output_c, (0, max_len - output_c.size(-1))).to(device='cuda')

        if not return_losses:
            return decoded

        losses = {
            'reconstruction': ((decoded - (output_c if output_c is not None else input_c)) ** 2).mean(axis=1),
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
                dataset=self._get_dataset('train', lazy=False),
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
    
@confugue.configurable
class ExperimentZeroShot:

    def __init__(self, logdir, config_path=None, device='cuda', sr=22050, pretrained_cola=None, continue_training=False, frozen_style_encoder=True, continue_step=None):
        self.logdir = logdir
        self.config_path = config_path
        self.sr = sr

        self._spec_fn = self._cfg['spectrogram'].bind(librosa.stft)
        self._mel_spec_fn = self._cfg['melspectrogram'].bind(librosa.feature.melspectrogram)
        self._inv_spec_fn = self._cfg['invert_spectrogram'].bind(
            librosa.griffinlim, random_state=0)

        torch.autograd.set_detect_anomaly(True)
        self.frozen_style_encoder = frozen_style_encoder
        if self.frozen_style_encoder:
            self.model = self._cfg['model_zero_shot'].configure(ModelZeroShot)
        else:
            self.model = self._cfg['model_zero_shot'].configure(ModelZeroShotUnfrozenStyle)
        LOGGER.info(self.model)
        self.device = torch.device(device)
        
        if continue_training:
            self.model.load_state_dict(torch.load(os.path.join(self.logdir, 'model_state.pt')))
        elif pretrained_cola:
            self.model.style_encoder.load_state_dict(torch.load(pretrained_cola))
        self.model.to(self.device)

        self.optimizer = None
        self.continue_training = continue_training
        self.continue_step = continue_step
        self.lr = self._cfg['optimizer'].get('lr')
        self.style_encoder_lr = self._cfg.get('style_encoder_lr') or None

    def train(self):
        with torch.utils.tensorboard.SummaryWriter(log_dir=self.logdir) as tb_writer:
            if self.config_path:
                with open(self.config_path) as f:
                    tb_writer.add_text('config', util.markdown_format_code(f.read()))
            tb_writer.add_text('model', util.markdown_format_code(repr(self.model)))

            self.model.train(True)
            if not self.optimizer:
                if self.frozen_style_encoder:
                    # The model is definitely an instance of ModelZeroShot
                    params = self.model.parameters()
                else:
                    # The model is definitely an instance of ModelZeroShotUnfrozenStyle
                    params=[
                        {"params": self.model.content_encoder.parameters()},
                        {"params": self.model.vq.parameters()},
                        {"params": self.model.style_encoder.parameters(), "lr": self.style_encoder_lr},
                        {"params": self.model.decoder_modules.parameters()},
                    ]
                self.optimizer = self._cfg['optimizer'].configure(
                    torch.optim.Adam, params=params)
            if self.continue_training:
                try:
                    self.optimizer.load_state_dict(torch.load(os.path.join(self.logdir, 'optimizer.pt')))
                except:
                    LOGGER.warning('Optimizer incompatible (Check frozen vs unfrozen style encoder). Moving on with fresh optimizer...')

            loader_train = self._cfg['train_loader'].configure(
                torch.utils.data.DataLoader,
                dataset=self._get_dataset('train'),
                collate_fn=util.collate_padded_tuples,
                shuffle=True,
                num_workers=6
            )
            loader_val = self._cfg['val_loader'].configure(
                torch.utils.data.DataLoader,
                dataset=self._get_dataset('val', lazy=False),
                collate_fn=util.collate_padded_tuples,
                num_workers=6
            )
            loader_val2 = None
            if 'val2' in self._cfg['data']:
                loader_val2 = self._cfg['val_loader'].configure(
                    torch.utils.data.DataLoader,
                    dataset=self._get_dataset('val2', lazy=False),
                    collate_fn=util.collate_padded_tuples,
                    num_workers=6
                )

            num_epochs = self._cfg.get('epochs', 1)
            val_period = self._cfg.get('val_period', np.nan)
            log_period = self._cfg.get('log_period', 1)
            save_period = self._cfg.get('save_period', 20)
            sample_period = self._cfg.get('sample_period', 1)

            i = self.continue_step
            for epoch in range(num_epochs):
                LOGGER.info('Starting epoch %d / %d', epoch + 1, num_epochs)
                for (input_c, length_c), (input_s, length_s), (input_out, length_out) in loader_train:
                    input_c, length_c, input_s, length_s, input_out, length_out = (
                        x.to(self.device) for x in (input_c, length_c, input_s, length_s, input_out, length_out))

                    # Validation
                    if i % val_period == 0:
                        with torch.no_grad():
                            LOGGER.info('Starting validation at step %d', i)
                            self._validate(loader=loader_val, tb_writer=tb_writer, step=i,
                                           write_samples=True,
                                           write_model=True)
                            self._validate(loader=loader_val2, tb_writer=tb_writer, step=i,
                                           write_samples=True,
                                           write_model=True, suffix='zero_shot')
                            LOGGER.info('Validation done')
                    if i % save_period == 0:
                        LOGGER.info('Saving model at step %d', i)
                        torch.save(self.model.state_dict(), os.path.join(self.logdir, 'model_state.pt'))
                        torch.save(self.optimizer.state_dict(), os.path.join(self.logdir, 'optimizer.pt'))
                        with open(os.path.join(self.logdir, 'last_step.txt'), 'w') as f:
                            print(i, file=f)

                    # Forward pass
                    if not self.model.training:
                        self.model.train(True)
                    _, losses = self.model(input_c, input_s, length_c, length_s, input_out, length_out, return_losses=True)
                    self._add_total_loss(losses, step=i)
                    print(f"Step {i}, loss: {losses['total']}")

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
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    if i % log_period == 0:
                        self._log_params(tb_writer, i)
                    self.optimizer.step()

                    i += 1
                    tb_writer.flush()

                LOGGER.info('Epoch %d finished (%d steps)', epoch + 1, i)
                
    def run_ground(self, data_loader):
        self.model.train(False)
        all_outputs, all_losses = [], []
        all_ground_truths = []
        input_device = None
        for (input_c, length_c), (input_s, length_s), (input_gt, length_gt) in data_loader:
            output, losses = self.model(
                input_c.to(self.device), input_s.to(self.device),
                length_c.to(self.device), length_s.to(self.device),
                input_gt.to(self.device), length_gt.to(self.device),
                return_losses=True)
            all_losses.append(losses)
            all_outputs.extend(output.to(input_c.device))
            all_ground_truths.extend(input_gt.to(input_c.device))

        all_losses = {name: torch.mean(torch.stack([x[name] for x in all_losses])).to(input_device)
                      for name in all_losses[0]}
        self._add_total_loss(all_losses)
        return all_outputs, all_losses, all_ground_truths

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
                path=pairs_path, sr=self.sr, preprocess_fn=self.preprocess, lazy=False, no_ground=True),
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
                  write_losses=True, write_samples=False, write_model=False, suffix=''):
        if loader is None:
            return
        
        outputs, losses, ground_truths = self.run_ground(loader)
        if write_losses and tb_writer:
            for name, loss in losses.items():
                tb_writer.add_scalar(f'loss_valid_{suffix}/{name}', loss, step)

        if write_samples:
            num_samples = self._cfg.get('num_val_samples', 4)
            sample_ids = random.Random(42).sample(range(len(outputs)), num_samples)
            for j in sample_ids:
                output = outputs[j].numpy()
                ground_truth = ground_truths[j].numpy()

                fig = plt.figure()
                librosa.display.specshow(output, figure=fig)
                tb_writer.add_figure(f'ex_{suffix}/{j}/spec', fig, global_step=step)
                fig_gt = plt.figure()
                librosa.display.specshow(ground_truth, figure=fig_gt)
                tb_writer.add_figure(f'gt_{suffix}/{j}/spec', fig_gt, global_step=step)

                audio = self.postprocess(output)
                tb_writer.add_audio(f'ex_{suffix}/{j}/audio', audio, sample_rate=self.sr,
                                    global_step=step)
                audio_gt = self.postprocess(ground_truth)
                tb_writer.add_audio(f'gt_{suffix}/{j}/audio', audio_gt, sample_rate=self.sr,
                                    global_step=step)

        if write_model:
            torch.save(self.model.state_dict(), os.path.join(self.logdir, 'model_state.pt'))
            torch.save(self.optimizer.state_dict(), os.path.join(self.logdir, 'optimizer.pt'))

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
    
    def melspectrogram_preprocess(self, audio, sr=16_000):
        return self._mel_spec_fn(y=audio, sr=sr)

    def postprocess(self, spectrogram):
        return self._inv_spec_fn(S=np.expm1(np.maximum(spectrogram, 0)))

    def _get_dataset(self, section, **kwargs):
        return self._cfg['data'][section].configure(
            AudioTupleDataset, sr=self.sr, preprocess_fn=self.preprocess,
            mel_preprocess_fn=self._mel_spec_fn, no_ground=False,
            **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--logdir', type=str, required=True)
    parser.add_argument('--pretrained_cola', type=str, help='Path to pretrained cola model')
    parser.add_argument('--continue_training', dest='continue_training', action='store_true')
    parser.add_argument('--unfrozen_style_encoder', dest='frozen_style_encoder', action='store_false')
    actions = parser.add_subparsers(title='action')
    train_parser = actions.add_parser('train', help='Train the model')
    train_parser.set_defaults(action='train')
    run_parser = actions.add_parser('run', help='Run the trained model on a dataset')
    run_parser.set_defaults(action='run')
    run_parser.add_argument('pairs_path', metavar='PAIRS_FILE',
                            help='a file listing on each line a pair of audio files to use as '
                                 'the content and style input, respectively; the paths need to '
                                 'be relative to the directory containing %(metavar)s')
    run_parser.add_argument('output_list_path', metavar='OUT_LIST_FILE',
                            help='a file to write the list of output audio files to')
    run_parser.add_argument('output_prefix', metavar='OUTPUT_PREFIX',
                            help='a prefix (e.g. a directory path followed by a slash) for the '
                                 'output audio files')
    
    run_parser.add_argument('--batch-size', type=int, metavar='SIZE')
    args = parser.parse_args()
    
    if args.frozen_style_encoder is None:
        args.frozen_style_encoder = True
        
    last_step_path = os.path.join(args.logdir, 'last_step.txt')
    try:
        with open(last_step_path, 'r') as f:
            continue_step = int(f.read())
    except:
        continue_step = 0
        LOGGER.warning("No file at %s exists. Assuming continue_step of 0...")

    torch.manual_seed(0)
    np.random.seed(0)

    cfg_path = os.path.join(args.logdir, 'config-zero-shot.yaml')
    cfg = confugue.Configuration.from_yaml_file(cfg_path)
    exp = cfg.configure(ExperimentZeroShot, device='cuda', config_path=cfg_path, logdir=args.logdir, pretrained_cola=args.pretrained_cola, continue_training=args.continue_training, frozen_style_encoder=args.frozen_style_encoder, continue_step=continue_step)
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
