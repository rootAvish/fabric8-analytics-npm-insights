#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module to handle pre-train and training of the model.

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
along with this program. If not, see <http://www.gnu.org/licenses/>.
"""
import tensorflow as tf

from recommendation_engine.model.collaborative_variational_autoencoder import \
    CollaborativeVariationalAutoEncoder

from recommendation_engine.model.pmf_training import PMFTraining
from recommendation_engine.data_pipeline.package_representation_data import \
    PackageTagRepresentationDataset


class TrainingJob:
    """Define the training job for the CVAE model."""

    def __init__(self):
        """Creates a new training job."""
        self.estimator = CollaborativeVariationalAutoEncoder(hidden_units=[200, 100])

    def train(self, train_input_fn):
        """Fire a training job."""
        # TODO
        self.estimator.train(input_fn=PackageTagRepresentationDataset.get_train_input_fn(
            batch_size=50000,
            num_epochs=50
        ))
