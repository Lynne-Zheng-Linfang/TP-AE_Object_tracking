
import tensorflow as tf
import numpy as np
from utils import lazy_property

class Combiner(object):

    def __init__(self, input, target, net_structure, batch_norm, is_training=False, drop_out = False):
        self._input = input 
        self._target = target
        self._net_structure = net_structure
        self._batch_norm = batch_norm 
        self._drop_out = drop_out
        self._is_training = is_training
        self.combiner_pred

    @lazy_property
    def combiner_pred(self):
        x = self._input
        for units in self._net_structure:
            x = tf.keras.layers.Dense(
                units,
                activation=tf.nn.relu,
                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
            )(x)
            if self._batch_norm:
                x = tf.keras.layers.BatchNormalization()(x, training=self._is_training)
            if self._drop_out:
                x = tf.keras.layers.Dropout(rate = 0.2)(x, training = self._is_training)
                
        x = tf.keras.layers.Dense(
            # int(self._target.get_shape()[-1]),
            128,
            activation=None,
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
        )(x)
        return x 
    
    @lazy_property
    def loss(self):
        target = self._target
        pred = self.combiner_pred
        loss = tf.keras.losses.cosine_similarity(target, pred)
        loss = tf.reduce_mean(input_tensor=loss)
        tf.compat.v1.summary.scalar('combiner_loss', loss)
        return loss
        
    @lazy_property
    def global_step(self):
        return tf.Variable(0, dtype=tf.int64, trainable=False, name='global_step')
