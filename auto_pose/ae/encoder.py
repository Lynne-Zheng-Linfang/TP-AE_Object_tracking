# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

from utils import lazy_property
# tf.keras.backend.set_floatx('float32')
# import tensorflow_probability as tfp # for tensorflow >= 2.2.0

class Encoder(object):

    def __init__(self, input, latent_space_size, num_filters, kernel_size, strides, batch_norm, visable_amount = None, visable_amount_net = None, is_training=False, drop_out = False, test_x = None):
        self._input = tf.stop_gradient(tf.cast(input, dtype=tf.float32)/255)
        self._visable_amount = visable_amount
        self._visable_amount_net = visable_amount_net
        self._latent_space_size = latent_space_size
        self._num_filters = num_filters
        self._kernel_size = kernel_size
        self._drop_out = drop_out
        self._strides = strides
        self._batch_normalization = batch_norm
        self._is_training = is_training
        self.z

    @property
    def x(self):
        return self._input

    @property
    def latent_space_size(self):
        return self._latent_space_size
    
    @lazy_property
    def encoder_out(self):
        x = self._input

        for filters, stride in zip(self._num_filters, self._strides):
            padding = 'same'
            x = tf.keras.layers.Conv2D(
                input_shape=self._input.get_shape()[1:],
                filters=filters,
                kernel_size=self._kernel_size,
                strides=stride,
                padding=padding,
                kernel_initializer=tf.keras.initializers.GlorotNormal(),
                activation=tf.nn.relu
            )(x)
            if self._batch_normalization:
                x = tf.keras.layers.BatchNormalization()(x, training=self._is_training)
            if self._drop_out:
                x = tf.keras.layers.Dropout(rate = 0.2)(x, training = self._is_training)

        encoder_out = tf.keras.layers.Flatten(name='encoder_out')(x)
        
        return encoder_out

    @lazy_property
    def z(self):
        x = self.encoder_out
        z = tf.keras.layers.Dense(
            self._latent_space_size,   
            activation=None,
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
            name = 'latent_code'
        )(x)

        return z
 
    @lazy_property
    def vis_amount_pred(self):
        x = self.encoder_out
        for uints in self._visable_amount_net:
            x = tf.keras.layers.Dense(
                uints,
                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                activation=tf.nn.relu
            )(x)
            if self._batch_normalization:
                x = tf.keras.layers.BatchNormalization()(x, training=self._is_training)
            if self._drop_out:
                x = tf.keras.layers.Dropout(rate = 0.2)(x, training = self._is_training)
        x = tf.keras.layers.Dense(
            1,   
            activation=tf.nn.sigmoid,
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
        )(x)
        return x 
    
    @lazy_property
    def vis_amount_loss(self):
        vis_target = self._visable_amount
        vis_pred = self.vis_amount_pred
        loss = tf.compat.v1.losses.absolute_difference(
            vis_target,
            vis_pred,
            reduction=tf.compat.v1.losses.Reduction.MEAN
        )
        tf.compat.v1.summary.scalar('visable_amount_loss', loss)
        return loss