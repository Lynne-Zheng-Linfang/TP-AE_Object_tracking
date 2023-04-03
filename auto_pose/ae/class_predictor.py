# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from operator import mul
from functools import reduce

from utils import lazy_property

class ClassPredictor(object):

    def __init__(self, image_input, reconstr_image, code_input, t_target, num_filters, kernel_size, strides, batch_norm, drop_out, pred_img_net, classification_t_net, t_class_num, is_training = False, fine_tune = False):
        self._image_input = image_input
        self._code_input= code_input 
        self._t_target = tf.cast(t_target, tf.int32)
        self._num_filters = num_filters
        self._kernel_size = kernel_size
        self._strides = strides
        self._batch_normalization = batch_norm
        self._is_training = is_training
        self._drop_out = drop_out
        self._pred_img_net = pred_img_net
        self._reconstr_image = reconstr_image
        self._fine_tune = fine_tune
        self._classification_t_net = classification_t_net
        self._t_class_num = t_class_num
        self.classification_tx_out
        self.classification_ty_out
        self.classification_tz_out
        self.tx
        self.ty
        self.tz
        self.t_out

    @property
    def code_input(self):
        return self._code_input

    @property
    def t_target(self):
        return self._t_target

    @property
    def latent_space_size(self):
        return self._latent_space_size
    
    @property
    def is_training(self):
        return self._is_training
    
    @lazy_property
    def img_net_input(self):
        x = self._image_input
        reconstr_img = self._reconstr_image
        retain_mask = tf.reduce_any(input_tensor=tf.greater(reconstr_img, 0.075), axis=3)
        retain_mask = tf.expand_dims(retain_mask, axis = 3)
        x = tf.cast(retain_mask, dtype=float)*x
        x = tf.stop_gradient(x)
        return x

    @lazy_property
    def img_cnn_out(self):
        x = self.img_net_input
        for filters, stride in zip(self._pred_img_net, self._strides):
            padding = 'same'
            x = tf.compat.v1.layers.conv2d(
                inputs=x,
                filters=int(filters),
                kernel_size=self._kernel_size,
                strides=stride,
                padding=padding,
                kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                activation=tf.nn.relu
            )
            if self._batch_normalization:
                x = tf.compat.v1.layers.batch_normalization(x, training=self._is_training)
        img_cnn_out = tf.keras.layers.Flatten()(x)
        return img_cnn_out 

    @property
    def t_out_dim(self):
        return self._t_class_num
    
    @lazy_property
    def loss(self):
        loss = self.classification_tx_loss + self.classification_ty_loss + self.classification_tz_loss
        return loss

    @lazy_property
    def processed_code(self):
        x = self._code_input
        if self._batch_normalization:
            x = tf.compat.v1.layers.batch_normalization(x, training=self._is_training)
        # x = tf.stop_gradient(x)
        return x

    @lazy_property
    def classified_net_out(self):
        x = self.img_cnn_out
        x = tf.concat(values=[self.img_cnn_out, self.processed_code], axis=1)
        for uints in self._classification_t_net:
            x = tf.compat.v1.layers.dense(
                x,
                uints,
                kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                activation=tf.nn.relu
            )
            if self._batch_normalization:
                x = tf.compat.v1.layers.batch_normalization(x, training=self._is_training)
            if self._drop_out:
                x = tf.keras.layers.Dropout(rate = 0.2)(x, training = self._is_training)
        return x
    
    @lazy_property
    def classification_tx_out(self):
        x = self.classified_net_out
        x = tf.compat.v1.layers.dense(
            x,
            self.t_out_dim,   
            activation=None,
            kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
        )
        return x
    
    @lazy_property
    def classification_tx_loss(self):
        target_tx = tf.reshape(self.t_target, (-1,3))[:,0]
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_tx, logits= self.classification_tx_out)
        loss = tf.reduce_mean(input_tensor=loss)
        tf.compat.v1.summary.scalar('tx_pred_loss', loss)
        return loss

    @lazy_property
    def tx(self):
        x = self.classification_tx_out
        x = tf.nn.softmax(x, axis=1)
        new_x, indices = tf.nn.top_k(x, k =100)
        tx = tf.reduce_sum(input_tensor=new_x*tf.cast(indices,dtype=tf.float32), axis=1)
        # tx = tf.argmax(x, axis=1)
        return tx

    @lazy_property
    def classification_ty_out(self):
        x = self.classified_net_out
        x = tf.compat.v1.layers.dense(
            x,
            self.t_out_dim,   
            activation=None,
            kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
        )
        return x
    
    @lazy_property
    def classification_ty_loss(self):
        target_ty = tf.reshape(self.t_target, (-1,3))[:,1]
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_ty, logits= self.classification_ty_out)
        loss = tf.reduce_mean(input_tensor=loss)
        tf.compat.v1.summary.scalar('ty_pred_loss', loss)
        return loss

    @lazy_property
    def ty(self):
        x = self.classification_ty_out
        x = tf.nn.softmax(x, axis=1)
        new_x, indices = tf.nn.top_k(x, k =100)
        ty = tf.reduce_sum(input_tensor=new_x*tf.cast(indices,dtype=tf.float32), axis=1)
        # indices = tf.range(0, self.t_out_dim, dtype=tf.float32)
        # ty = tf.reduce_sum(x*indices, axis = 1)
        # ty = tf.argmax(x, axis=1)
        return ty

    @lazy_property
    def classification_tz_out(self):
        x = self.classified_net_out
        x =  tf.compat.v1.layers.dense(
            x,
            self.t_out_dim,   
            activation=None,
            kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
        )
        return x
    
    @lazy_property
    def classification_tz_loss(self):
        target_tz = tf.reshape(self.t_target, (-1,3))[:,2]
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_tz, logits= self.classification_tz_out)
        loss = tf.reduce_mean(input_tensor=loss)
        tf.compat.v1.summary.scalar('tz_pred_loss', loss)
        return loss

    @lazy_property
    def tz(self):
        x = self.classification_tz_out
        x = tf.nn.softmax(x, axis=1)
        new_x, indices = tf.nn.top_k(x, k =100)
        tz = tf.reduce_sum(input_tensor=new_x*tf.cast(indices,dtype=tf.float32), axis=1)
        # indices = tf.range(0, self.t_out_dim, dtype=tf.float32)
        # tz = tf.reduce_sum(x*indices, axis = 1)
        # tz = tf.argmax(x, axis=1)
        return tz

    @lazy_property
    def t_out(self):
        tx = tf.reshape(self.tx, (-1,1))
        ty = tf.reshape(self.ty, (-1,1))
        tz = tf.reshape(self.tz, (-1,1))
        x = tf.concat(values=[tx, ty, tz], axis=1)
        return x 
    
