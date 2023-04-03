
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from operator import mul
from functools import reduce

from utils import lazy_property

class VisableAmountPredictor(object):

    def __init__(self, input_img, net_structure, visable_amount_target, kernel_size, strides, batch_norm, is_training=False):
        self._input_img = input_img
        self._net_structuret = net_structure
        self._visable_amount_target = visable_amount_target
        self._kernel_size = kernel_size
        self._strides = strides
        self._batch_normalization = batch_norm
        self._is_training = is_training
        self.vis_amount_pred

    @lazy_property
    def vis_amount_pred(self):
        x = self._input_img
        for filters, stride in zip(self._net_structuret, self._strides):
            padding = 'same'
            x = tf.compat.v1.layers.conv2d(
                inputs=x,
                filters=filters,
                kernel_size=self._kernel_size,
                strides=stride,
                padding=padding,
                kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                activation=tf.nn.relu
            )
            if self._batch_normalization:
                x = tf.compat.v1.layers.batch_normalization(x, training=self._is_training)

        img_out = tf.keras.layers.Flatten()(x)
        x = tf.compat.v1.layers.dense(
            img_out,
            1,   
            activation=tf.nn.sigmoid,
            kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
        )
        return x 
    
    @lazy_property
    def loss(self):
        vis_target = self._visable_amount_target
        vis_pred = self.vis_amount_pred
        loss = tf.compat.v1.losses.absolute_difference(
        # loss = tf.compat.v1.losses.mean_squared_error(
            vis_target,
            vis_pred,
            reduction=tf.compat.v1.losses.Reduction.MEAN
        )
        tf.compat.v1.summary.scalar('visable_amount_loss', loss)
        return loss

