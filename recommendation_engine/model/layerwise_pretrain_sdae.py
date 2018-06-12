#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Defines the layerwise pretraining to initialize the SDAE module.

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
import logging

import tensorflow as tf
import daiquiri
import tensorflow.contrib.layers as layers

from recommendation_engine.config.params_training import training_params
from recommendation_engine.data_pipeline.package_representation_data import \
    PackageTagRepresentationDataset

daiquiri.setup(level=logging.DEBUG)
_logger = daiquiri.getLogger(__name__)


class LayerwisePretrain:
    """Layerwise pretrain the SDAE."""

    def __init__(self, data_store):
        """Create a new pretrain instance."""
        self.de_weights = []
        self.de_biases = []

        self.en_weights = []
        self.en_biases = []
        self.data_store = data_store
        self.session = tf.Session()

    def get_weights(self, data_x, hidden_dims, output_dim):
        """Return the weight, bias matrices after running pretrain."""
        for layer_id, dim in enumerate(hidden_dims + [output_dim], 1):
            _logger.debug("Running layerwise pretraining for layer {0}".format(layer_id))
            x = self._run_pretrain(data_x, dim, layer_id)
        return self._make_weight_dict()

    def _make_weight_dict(self):
        """Aggregate the weight lists in a dict."""
        assert(self.en_weights)
        assert(self.en_biases)
        assert(self.de_weights)
        assert(self.de_biases)

        return {
            "encoder_weights": self.en_weights,
            "encoder_biases": self.en_biases,
            "decoder_weights": self.de_weights,
            "decoder_biases": self.de_biases
        }

    def _run_pretrain(self, X, dim, layer_id):
        """Run the layerwise pretraining for this layer."""
        with tf.variable_scope("pretrain", reuse=tf.AUTO_REUSE):
            encoded = layers.fully_connected(
                X, dim, scope="enc_layer_{0}".format(layer_id),
                activation_fn=tf.nn.sigmoid)
            x_recon = layers.fully_connected(
                encoded, X.shape[1], scope='dec_layer_{}'.format(layer_id),
                activation_fn=tf.nn.sigmoid)
            tf.assign(tf.get_variable("dec_layer_{0}/weights".format(layer_id)),
                      tf.transpose(tf.get_variable("enc_layer_{0}/weights").format(layer_id)))

            loss = tf.losses.sigmoid_cross_entropy(X, x_recon)
            train_op = tf.train.AdamOptimizer(training_params.learning_rate).minimize(loss)

            session = tf.Session()
            session.run(tf.global_variables_initializer())
            for epoch in range(training_params.get('num_epochs')):
                _, loss_value = session.run([train_op, loss])
                _logger.debug("Loss at epoch {}:{}".format(epoch, loss_value))
            return session.run(encoded)
