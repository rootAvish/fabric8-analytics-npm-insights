#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file contains the CVAE model definition.

Copyright © 2018 Red Hat Inc.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import tensorflow as tf
from recommendation_engine.model.layer_definitions import sdae_autoencoder_net
from recommendation_engine.config.params_training import training_params


def sdae_net_model_fn(features, labels, hidden_units, output_dim, activation, learning_rate, mode):
    """Model function for pretraining the estimator."""
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    # Define model's architecture
    logits = sdae_autoencoder_net(inputs=features,
                                  hidden_units=hidden_units,
                                  output_dim=output_dim,
                                  activation=activation,
                                  mode=mode,
                                  scope='VarAutoEnc')

    # TODO: Train and save pretrain weights.

class StackedDenoisingAutoEncoder(tf.estimator.Estimator):
    """Estimator API wrapper for CVAE autoencoder model."""

    def __init__(self, hidden_units, output_dim, activation_fn=tf.nn.sigmoid,
                 learning_rate=training_params.learning_rate, model_dir=None, config=None):
        """Create a new Stacked Denoising Autoencoder to do the pretraining."""
        def _model_fn(features, labels, mode):
            return sdae_net_model_fn(
                features=features,
                labels=labels,
                hidden_units=hidden_units,
                output_dim=output_dim,
                activation=activation_fn,
                learning_rate=learning_rate,
                mode=mode)

        super().__init__(
            model_fn=_model_fn,
            model_dir=model_dir,
            config=config)
