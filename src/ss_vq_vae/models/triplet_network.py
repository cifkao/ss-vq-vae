# Copyright 2020 InterDigital R&D and Télécom Paris.
# Authors: Ondřej Cífka, Brian McFee, Jongpil Lee, Juhan Nam
# License: Apache 2.0

"""Triplet network for metric learning.

Gently modified from the ISMIR 2020 Tutorial for Metric Learning in MIR by Brian McFee, Jongpil Lee
and Juhan Nam, originally available at https://github.com/bmcfee/ismir2020-metric-learning/
and dedicated to the public domain under the CC0-1.0 license.
"""

import json
import os

import librosa
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from tensorflow.keras.layers import (Conv1D, MaxPool1D, BatchNormalization, GlobalAvgPool1D, Dense, dot,
                                     Activation, Input, Flatten, Lambda, Embedding, Concatenate, Layer, Reshape)
from sklearn.preprocessing import normalize


def build_model(num_features):
    """Build model."""
    def basic_block(x, num_features, fp_length):
        x = Conv1D(num_features, fp_length, padding='same', use_bias=True, kernel_initializer='he_uniform')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPool1D(pool_size=fp_length, padding='valid')(x)
        return x

    # Backbone model.
    num_frames = None
    x_in = Input(shape=(num_frames, num_features))
    x = basic_block(x_in, 64, 4)
    x = basic_block(x, 64, 4)
    x = basic_block(x, 64, 4)
    x = basic_block(x, 64, 2)
    x = GlobalAvgPool1D()(x)
    backbone_model = Model(inputs=[x_in], outputs=[x], name='backbone')
    backbone_model.summary()

    # Triplet model.
    anchor = Input(shape=(num_frames, num_features), name='anchor_input')
    positive = Input(shape=(num_frames, num_features), name='positive_input')
    negative = Input(shape=(num_frames, num_features), name='negative_input')

    anchor_embedding = backbone_model(anchor)
    positive_embedding = backbone_model(positive)
    negative_embedding = backbone_model(negative)

    # Cosine similarity.
    dist_fn = Lambda(lambda x: dot(x, axes=1, normalize=True))
    dist_anchor_positive = dist_fn([anchor_embedding, positive_embedding])
    dist_anchor_negative = dist_fn([anchor_embedding, negative_embedding])

    # Stack the similarity scores [1,0] and triplet model.
    similarity_scores = Lambda(lambda vects: K.stack(vects, axis=1))([dist_anchor_positive, dist_anchor_negative])
    tripletmodel = Model(inputs=[anchor, positive, negative], outputs=similarity_scores, name='triplet')
    tripletmodel.summary()

    return tripletmodel, backbone_model


def triplet_hinge_loss(y_true, y_pred):
    """Triplet hinge loss."""
    del y_true
    # Always the first dimension of the similarity score is true.
    # Margin is set to 0.1
    y_pos = y_pred[:, 0]
    y_neg = y_pred[:, 1]
    loss = K.mean(K.maximum(0., 0.1 + y_neg - y_pos))
    return loss


def data_loader(path, load_fn, batch_size, shuffle=False, repeat=False):
    """Data loader."""
    rng = np.random.default_rng(seed=0)

    base_path = os.path.dirname(path)
    with open(path) as f:
        dataset = [tuple(os.path.join(base_path, p) for p in l.rstrip('\n').split('\t'))
                   for l in f]

    steps_per_epoch = len(dataset) // batch_size

    def loader():
        count_triplet = 0
        while True:
            if shuffle:
                rng.shuffle(dataset)

            for _ in range(0, steps_per_epoch * batch_size, batch_size):
                if count_triplet > len(dataset) - batch_size:
                    count_triplet = 0

                batch_x, batch_y = batch_triplet_loader(
                    dataset[count_triplet: count_triplet + batch_size],
                    load_fn=load_fn)

                count_triplet += batch_size
                yield batch_x, batch_y

            if not repeat:
                break

    return loader, steps_per_epoch


def batch_triplet_loader(triplets, load_fn):
    """Batch loader."""

    anchor_col = []
    positive_col = []
    negative_col = []
    for p_anchor, p_positive, p_negative in triplets:
        a_anchor, a_positive, a_negative = (
            load_fn(p) for p in [p_anchor, p_positive, p_negative])

        # Stack batch data.
        anchor_col.append(a_anchor)
        positive_col.append(a_positive)
        negative_col.append(a_negative)

    # To array.
    anchor_col = np.array(anchor_col)
    positive_col = np.array(positive_col)
    negative_col = np.array(negative_col)

    batch_x = {
        'anchor_input': anchor_col,
        'positive_input': positive_col,
        'negative_input': negative_col
    }

    batch_y = np.zeros((anchor_col.shape[0], 2))
    batch_y[:, 0] = 1
    return batch_x, batch_y
