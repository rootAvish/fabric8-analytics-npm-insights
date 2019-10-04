#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train the probabilistic matrix factorization model."""

import os

import tensorflow as tf
import numpy as np
from scipy.io import savemat

import recommendation_engine.config.params_training as training_params
import recommendation_engine.config.path_constants_train as path_constants
from recommendation_engine.utils.fileutils import load_rating, create_logger
from rudra.data_store import local_data_store

logger = create_logger(__name__)


class PMFTraining:
    """Training definitions the Probabilistic Matrix factorization model."""

    def __init__(self, num_users, num_items, m_weights):
        """Create a new PMF training instance."""
        with tf.variable_scope('pmf_training', reuse=tf.AUTO_REUSE):
            self.m_items = tf.get_variable('m_V', initializer=tf.random_normal_initializer,
                                           shape=[num_users, training_params.num_latent])
            self.m_users = tf.get_variable('m_U', initializer=tf.random_normal_initializer,
                                           shape=[num_items, training_params.num_latent])
            self.m_weights = m_weights

    def __call__(self, *args, **kwargs):
        """Train the model."""
        convergence = tf.Variable(1.0, 'convergence', dtype=tf.float32)
        item_user_map = kwargs['item_to_user_matrix']
        user_item_map = kwargs['user_to_item_matrix']

        with tf.variable_scope('pmf_training', reuse=tf.AUTO_REUSE):
            a_minus_b = tf.subtract(tf.constant(training_params.a, dtype=tf.float32),
                                    tf.constant(training_params.b, dtype=tf.float32))
            likelihood = tf.Variable(-tf.exp(20.0), name='likelihood', dtype=tf.float32)
            likelihood_old = tf.Variable(0, name='likelihood_old', dtype=tf.float32)
            # Loop over the training till convergence
            for iteration in range(0, training_params.max_iter):
                likelihood = tf.assign(likelihood, 0)
                # Update the user vectors.
                item_ids = np.array([np.array(idx) for idx, row in enumerate(item_user_map)
                                     if len(row) > 0])
                rated_items = tf.gather(self.m_items, item_ids)
                user_items_sq = tf.matmul(rated_items, rated_items, transpose_a=True)
                users_items_weighted = user_items_sq * training_params.b + tf.eye(
                        training_params.num_latent) * training_params.lambda_u

                for user_id, this_user_items in enumerate(user_item_map):
                    if len(this_user_items) == 0:
                        continue
                    item_norm = users_items_weighted + tf.matmul(
                            tf.gather(self.m_items, this_user_items),
                            tf.gather(self.m_items, this_user_items), transpose_a=True) * a_minus_b
                    tf.assign(self.m_users[user_id:], tf.linalg.solve(
                            item_norm,
                            training_params.a * tf.reduce_sum(
                                    tf.gather(self.m_items, this_user_items),
                                    axis=0)))

                    likelihood = tf.assign(
                            likelihood,
                            likelihood + (-0.5) * training_params.lambda_u * tf.reduce_sum(
                                    self.m_users[user_id, :] * self.m_users[user_id, :], axis=1))

                # Update the item vectors
                user_ids = np.array([np.array(idx) for idx, row in enumerate(user_item_map)
                                     if len(row) > 0])
                all_items_user = tf.gather(self.m_users, user_ids)
                items_users_weighted = tf.matmul(all_items_user, all_items_user,
                                                 transpose_a=True) * training_params.b

                for item_id, this_item_users in enumerate(item_user_map):
                    if len(this_item_users) == 0:
                        # Never been rated
                        item_norm = items_users_weighted + tf.eye(
                                training_params.num_latent) * training_params.lambda_v
                        self.m_items[item_id, :] = tf.linalg.solve(
                                item_norm,
                                training_params.lambda_v * self.m_weights[item_id, :])

                        # now calculate the likelihood
                        epsilon = self.m_items[item_id, :] - self.m_weights[item_id, :]
                        likelihood += -0.5 * training_params.lambda_v * tf.reduce_sum(
                                tf.square(epsilon))
                    else:
                        item_norm = items_users_weighted + tf.matmul(
                                tf.gather(self.m_users, this_item_users),
                                tf.gather(self.m_users, this_item_users),
                                transpose_a=True) * a_minus_b

                        item_norm_pre_weighing = item_norm
                        item_norm += tf.eye(training_params.num_latent) * training_params.lambda_v

                        tf.assign(self.m_items[item_id:], tf.linalg.solve(
                                item_norm,
                                training_params.a * tf.reduce_sum(
                                        tf.gather(self.m_users, this_item_users),
                                        axis=0) + training_params.lambda_v * self.m_weights[item_id,
                                                                             :]))

                        likelihood += (-0.5) * len(user_ids) * training_params.a + \
                                      training_params.a * tf.reduce_sum(
                                tf.matmul(tf.gather(self.m_users, user_ids),
                                          tf.reshape(self.m_items[item_id, :],
                                                     [training_params.num_latent, 1])), axis=0)

                        likelihood += -0.5 * tf.matmul(
                                tf.matmul(self.m_items[item_id, :], item_norm_pre_weighing),
                                tf.reshape(self.m_items[item_id, :],
                                           [training_params.num_latent, 1]))

                        epsilon = self.m_items[item_id, :] - self.m_weights[item_id, :]
                        likelihood += (-0.5) * training_params.lambda_v * tf.reduce_sum(
                                tf.square(epsilon))

                iteration += 1
                convergence = tf.assign(convergence,
                                        tf.abs((likelihood - likelihood_old) / likelihood_old))
                if convergence < 1e-6 and iteration > training_params.min_iter_pmf:
                    break

                with tf.Session() as self.session:
                    self.session.run(convergence)

    def save_model(self, data_store=None):
        """Save the model in matlab format to load later for scoring."""
        if not self.session:
            logger.error("There is no session created, model has not be trained.")
            return
        local_file_path = os.path.join(path_constants.LOCAL_MODEL_DIR,
                                       path_constants.PMF_MODEL_PATH)
        savemat(local_file_path,
                {"m_U": self.session.run(self.m_users),
                 "m_V": self.session.run(self.m_items),
                 "m_theta": self.session.run(self.m_weights)})
        if data_store:
            data_store.upload_file(local_file_path)


if __name__ == "__main__":
    user_item_filepath = "packagedata-train-5-users.dat"
    item_user_filepath = "packagedata-train-5-items.dat"
    data_store = local_data_store.LocalDataStore(
            '/Users/avgupta/s3/cvae-insights/npm/training-data-node/')
    item_to_user_matrix = load_rating(item_user_filepath, data_store)
    user_to_item_matrix = load_rating(user_item_filepath, data_store)
    # TODO: Load weights
    weights = {}
    pmf = PMFTraining(len(user_to_item_matrix), len(item_to_user_matrix), weights)
    pmf(user_to_item_matrix=user_to_item_matrix,
        item_to_user_matrix=item_to_user_matrix)
    pmf.save_model()