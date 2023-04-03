# -*- coding: utf-8 -*-
import numpy as np

import tensorflow as tf

from utils import lazy_property
from pysixd_stuff import transform
from sixd_rot import rot_6d

class PriorPoseModule(object):

    def __init__(self, pose_target, input_seqs, net_structure, drop_out = False, batch_norm = False, is_training=False, lower_bound = 0, upper_bound = 5000, trans_loss_scale = 1):
        self._net_structure = net_structure
        self._drop_out = drop_out
        self._batch_norm = batch_norm 
        self._pose_target = pose_target
        self._input_seqs = input_seqs
        self._is_training = is_training
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._trans_loss_scale = trans_loss_scale
        self.predict
    
    @lazy_property
    def _rot_target(self):
        return self._pose_target[:,:3,:3]

    @lazy_property
    def _trans_target(self):
        return self._pose_target[:,:3,3]
    
    @lazy_property
    def _input_rot(self):
        return rot_6d.tf_matrix_to_rotation6d_seq(self._input_seqs[:,:,:3,:3])

    @lazy_property
    def _input_trans(self):
        return self._input_seqs[:,:,:3,3]

    @lazy_property
    def _input(self):
        return tf.concat([self._input_trans, self._input_rot], axis=-1)
    
    @lazy_property
    def _prior_pose(self):
        x = self._input
        x = tf.keras.layers.GRU(self._net_structure[0], dropout= 0.2 if self._is_training and self._drop_out else 0)(x)
        for units in self._net_structure[1:]:
            x = tf.keras.layers.Dense(units, activation=tf.nn.tanh)(x)
        x = tf.keras.layers.Dense(9, activation=tf.nn.tanh)(x)
        return x
    
    @lazy_property
    def _rot_pred(self):
        rot_6d_repre = self._prior_pose[:,3:]
        Rs = rot_6d.tf_rotation6d_to_matrix(rot_6d_repre)
        return tf.reshape(Rs, (-1,3,3))
    
    @lazy_property
    def _trans_pred(self):
        trans = self._prior_pose[:,:3]
        return trans

    @lazy_property
    def _rot_target(self):
        return self._pose_target[:,:3,:3]
    
    @lazy_property
    def _trans_target(self):
        return self._pose_target[:,:3,3]
    
    @lazy_property
    def _rot_loss(self):
        loss = rot_6d.loss(self._rot_target, self._rot_pred)
        # loss = tf.reduce_mean(loss)
        tf.compat.v1.summary.scalar('rot_loss', loss)
        return loss
    
    @lazy_property
    def _trans_loss(self):
        loss = tf.losses.mean_squared_error(y_true=self._trans_target, y_pred=self._trans_pred)
        loss = tf.reduce_mean(loss)*self._upper_bound
        tf.compat.v1.summary.scalar('trans_loss', loss)
        return loss
    
    @lazy_property
    def loss(self):
        loss = self._trans_loss*self._trans_loss_scale + self._rot_loss
        tf.compat.v1.summary.scalar('pose_loss', loss)
        return loss 

    @lazy_property
    def global_step(self):
        return tf.Variable(0, dtype=tf.int64, trainable=False, name='global_step')
        
    @lazy_property
    def predict(self):
        return (self._rot_pred, self._trans_pred)

