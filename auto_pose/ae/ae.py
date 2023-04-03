# -*- coding: utf-8 -*-

import tensorflow as tf

from utils import lazy_property

class AE(object):

    def __init__(self, R_encoder, t_encoder, R_decoder, t_decoder, predictor, norm_regularize, variational, vis_amnt_pred=False):
        self._R_encoder = R_encoder
        self._t_encoder = t_encoder
        self._t_decoder = t_decoder
        self._R_decoder = R_decoder
        self._predictor = predictor
        self._norm_regularize = norm_regularize
        self._variational = variational
        self._vis_amnt_pred = vis_amnt_pred
        self._pred_loss_scale_factor = 1
        self.loss
        tf.compat.v1.summary.scalar('total_loss', self.loss)
        self.global_step

    @property
    def x(self):
        return self._R_encoder.x

    @property
    def z(self):
        return self._R_encoder.z

    @property
    def reconstruction(self):
        return self._R_decoder.x

    @property
    def reconstruction_target(self):
        return self._R_decoder.reconstruction_target

    @property
    def translation_z(self):
        return self._t_encoder.z

    @property
    def translation_reconstruction(self):
        return self._t_decoder.x

    @property
    def translation_reconstruction_target(self):
        return self._t_decoder.reconstruction_target

    @lazy_property
    def global_step(self):
        return tf.Variable(0, dtype=tf.int64, trainable=False, name='global_step')

    @property
    def prediction_loss_scale_factor(self):
        return self._pred_loss_scale_factor

    @lazy_property
    def loss(self):
        loss = self._R_decoder.reconstr_loss
        tf.compat.v1.summary.scalar('r_reconst_loss', loss)
        if self._t_decoder is not None:
            t_reconstr_loss = self._t_decoder.reconstr_loss
            loss += t_reconstr_loss
            tf.compat.v1.summary.scalar('t_reconst_loss', t_reconstr_loss)
        if self._predictor is not None:
            loss += self._predictor.loss * self._pred_loss_scale_factor
        # if self._vis_amnt_pred:
        #     loss += self._t_encoder.vis_amount_loss
        if self._norm_regularize > 0:
            loss += self._encoder.reg_loss * tf.constant(self._norm_regularize,dtype=tf.float32) + self._R_encoder.reg_loss * tf.constant(self._norm_regularize,dtype=tf.float32)
            tf.compat.v1.summary.scalar('reg_loss', self._R_encoder.reg_loss)
        if self._variational:
            loss +=  self._R_encoder.kl_div_loss * tf.constant(self._variational, dtype=tf.float32) + self._R_encoder.kl_div_loss * tf.constant(self._variational, dtype=tf.float32)
            tf.compat.v1.summary.scalar('KL_loss', self._R_encoder.kl_div_loss)
            tf.compat.v1.summary.histogram('Variance', self._R_encoder.q_sigma)
        tf.compat.v1.summary.histogram('Rotation Encoder Mean', self._R_encoder.z)
        if self._t_decoder is not None:
            tf.compat.v1.summary.histogram('Translation Encoder Mean', self._t_encoder.z)
        return loss



