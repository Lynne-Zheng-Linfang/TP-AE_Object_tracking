# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from tf_siren.siren import SinusodialRepresentationDense
from tf_siren.siren_mlp import SIRENModel

from utils import lazy_property

class SirenEncoder(object):

    def __init__(self, input, latent_space_size, num_filters, kernel_size, strides, batch_norm, is_training=False):
        self._input = input
        self._latent_space_size = latent_space_size
        self._num_filters = num_filters
        self._kernel_size = kernel_size
        self._strides = strides
        self._batch_normalization = batch_norm
        self._is_training = is_training
        self.encoder_out
        self.z
        # self.delta_t_out
        # self.q_sigma
        # self.sampled_z
        # self.reg_loss
        # self.kl_div_loss

    @property
    def x(self):
        return self._input

    @property
    def latent_space_size(self):
        return self._latent_space_size
    
    @lazy_property
    def encoder_out(self):
        x = self._input
        h, w, c = x.get_shape().as_list()[1:]
        layer_dimensions = [ [int(h/np.prod(self._strides[i:])), int(w/np.prod(self._strides[i:]))]  for i in range(len(self._strides))]
        layer_dimensions.reverse()
        # print(layer_dimensions)
        print('num filters', self._num_filters)

        x = SinusodialRepresentationDense(self._num_filters[0], w0=30.0)(x)  
        print('first encoder layer shape:', x.get_shape())
        # x = tf.image.resize_nearest_neighbor(x, layer_dimensions[0])
        for filters, layer_size in zip(self._num_filters[1:], layer_dimensions[1:]):
            x = SinusodialRepresentationDense(filters, w0=1.0)(x)  
            print('encoder layer shape:', x.get_shape())
            # x = tf.image.resize_nearest_neighbor(x, layer_size)
        # x = SinusodialRepresentationDense(c, w0=1.0)(x)  
        # print('encoder layer shape:', x.get_shape())
        x = tf.contrib.layers.flatten(x)

        
        # h, w, c = self._input.get_shape().as_list()[1:]
        # x = self._input
        # layer_dimensions = []
        # layer_dimensions.append(int(h*w/16))
        # neuron = layer_dimensions[0]
        # for i in range(4):
        #     neuron = int(neuron/2)
        #     layer_dimensions.append(neuron)
        # # layer_dimensions.reverse()
        # print(layer_dimensions)
        # # x = tf.reshape(self._input, (-1, h*w*c))
        # x = SinusodialRepresentationDense(layer_dimensions[0], w0=30.0)(x)  
        # for layer_size in layer_dimensions[1:]:
        #     x = SinusodialRepresentationDense(layer_size, w0=1.0)(x)  
        return x 

    @lazy_property
    def z(self):
        x = self.encoder_out

        z = SinusodialRepresentationDense(self._latent_space_size,
                                  activation=None, # default activation function
                                  w0=1.0)(x)  
        return z
    
    @lazy_property
    def q_sigma(self):
        x = self.encoder_out

        q_sigma = 1e-8 + tf.layers.dense(inputs=x,
                        units=self._latent_space_size,
                        activation=tf.nn.softplus,
                        kernel_initializer=tf.zeros_initializer())

        return q_sigma

    @lazy_property
    def sampled_z(self):
        epsilon = tf.random_normal(tf.shape(self._latent_space_size), 0., 1.)
        # epsilon = tf.contrib.distributions.Normal(
        #             np.zeros(self._latent_space_size, dtype=np.float32), 
        #             np.ones(self._latent_space_size, dtype=np.float32))
        return self.z + self.q_sigma * epsilon


    @lazy_property
    def kl_div_loss(self):
        p_z = tf.contrib.distributions.Normal(
            np.zeros(self._latent_space_size, dtype=np.float32), 
            np.ones(self._latent_space_size, dtype=np.float32))
        q_z = tf.contrib.distributions.Normal(self.z, self.q_sigma)

        return tf.reduce_mean(tf.distributions.kl_divergence(q_z,p_z))


    @lazy_property
    def reg_loss(self):
        reg_loss = tf.reduce_mean(tf.abs(tf.norm(self.z,axis=1) - tf.constant(1.)))
        return reg_loss