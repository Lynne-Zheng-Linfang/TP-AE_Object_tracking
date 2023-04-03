# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from operator import mul
from functools import reduce
# tf.keras.backend.set_floatx('float32')

from utils import lazy_property

class Predictor(object):

    def __init__(self, image_input, reconstr_image, code_input, R_code_input, \
        t_target, num_filters, kernel_size, strides, batch_norm, drop_out, \
        pred_img_net, delta_t_net, scale_factor, max_translation_shit, \
        is_training = False, fine_tune = False, t_distr = False, use_mask = False, visable_amount = None, encoder_dir_out = None, phase_2 = False, gray_bg = False):
        self._image_input = tf.stop_gradient(tf.cast(image_input[:,:,:,:6], dtype=tf.float32)/255)
        self._gray_bg = gray_bg
        self._code_input= code_input 
        self._R_code_input= tf.stop_gradient(R_code_input)
        self._t_target = t_target 
        self._num_filters = num_filters
        self._kernel_size = kernel_size
        self._strides = strides
        self._batch_normalization = batch_norm
        self._is_training = is_training
        self._drop_out = drop_out
        self._scale_factor = scale_factor
        self._pred_img_net = pred_img_net
        self._delta_t_net = delta_t_net
        self._reconstr_image = reconstr_image
        self._fine_tune = fine_tune
        self._t_distr = t_distr
        self._use_mask = use_mask
        self._visable_amount = visable_amount
        self._encoder_dir_output = encoder_dir_out
        self._phase_2 = phase_2
        self._max_translation_shift = max_translation_shit

        if self._visable_amount is not None:
            self.vis_amount_pred
        if t_distr:
            self.fine_delta_t_distribution_out
            self.coarse_delta_t_distribution_out
        else:
            self.coarse_delta_t_out
            # self.fine_delta_t_with_coarse_t_out
            self.fine_delta_t_with_code_out

    @property
    def code_input(self):
        return self._code_input

    @property
    def t_target(self):
        return self._t_target

    @property
    def coarse_t_input(self):
        return self._coarse_t_input

    @property
    def latent_space_size(self):
        return self._latent_space_size
    
    @property
    def is_training(self):
        return self._is_training

    @lazy_property
    def gray_bg_color(self):
        bg_color = np.array([127,127,127,127,127,127])/255
        return tf.constant(bg_color, dtype=tf.float32)
    
    @lazy_property
    def img_net_input(self):
        x = self._image_input
        reconstr_img = self._reconstr_image
        if self._use_mask:
            retain_mask = tf.reduce_any(input_tensor=tf.greater(reconstr_img, 0.5), axis=3)
        else:
            if self._gray_bg:
                print('Predictor: use gray background')
                retain_mask = tf.reduce_any(input_tensor=(tf.math.abs(reconstr_img - self.gray_bg_color)> 0.075), axis=3)
            else:
                print('Predictor: use black background')
                retain_mask = tf.reduce_any(input_tensor=tf.greater(reconstr_img, 0.075), axis=3)
        retain_mask = tf.expand_dims(retain_mask, axis = 3)
        x = tf.cast(retain_mask, dtype=tf.float32)*x
        x = tf.stop_gradient(x)
        return x

    @lazy_property
    def vis_amount_pred(self):
        if self._encoder_dir_output is not None:
            if not self._phase_2:
                encoder_out = tf.stop_gradient(self._encoder_dir_output)
            else:
                encoder_out = self._encoder_dir_output
            x = tf.concat(values=[self.img_cnn_out, encoder_out], axis=1)
            for uints in self._delta_t_net:
                x = tf.keras.layers.Dense(
                    uints,
                    kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                    activation=tf.nn.relu
                )(x)
                if self._batch_normalization:
                    x = tf.keras.layers.BatchNormalization()(x, training=self._is_training)
                if self._drop_out:
                    x = tf.keras.layers.Dropout(rate = 0.2)(x, training = self._is_training)
                    # x = tf.nn.dropout(x, rate=0.2)
            x = tf.keras.layers.Dense(
                1,   
                activation=tf.nn.sigmoid,
                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
            )(x)
        else:
            print('Error: Please pass encoder.encoder_out to predictor')
            exit()
        return x 
    
    @lazy_property
    def vis_amount_loss(self):
        vis_target = self._visable_amount
        vis_pred = self.vis_amount_pred
        loss = tf.compat.v1.losses.absolute_difference(
        # loss = tf.compat.v1.losses.mean_squared_error(
            vis_target,
            vis_pred,
            reduction=tf.compat.v1.losses.Reduction.MEAN
        )
        tf.compat.v1.summary.scalar('visable_amount_loss', loss)
        return loss

    @lazy_property
    def img_cnn_out(self):
        x = self.img_net_input
        for filters, stride in zip(self._pred_img_net, self._strides):
            padding = 'same'
            x = tf.keras.layers.Conv2D(
                filters=int(filters),
                kernel_size=self._kernel_size,
                strides=stride,
                padding=padding,
                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                activation=tf.nn.relu
            )(x)
            if self._batch_normalization:
                x = tf.keras.layers.BatchNormalization()(x, training=self._is_training)
        img_cnn_out = tf.keras.layers.Flatten()(x)
        return img_cnn_out 

    @lazy_property
    def ori_img_cnn_out(self):
        x = self._image_input
        for filters, stride in zip(self._pred_img_net, self._strides):
            padding = 'same'
            x = tf.keras.layers.Conv2D(
                filters=int(filters),
                kernel_size=self._kernel_size,
                strides=stride,
                padding=padding,
                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                activation=tf.nn.relu
            )(x)
            if self._batch_normalization:
                x = tf.keras.layers.BatchNormalization()(x, training=self._is_training)
        ori_img_cnn_out = tf.keras.layers.Flatten()(x)
        return ori_img_cnn_out 

    @property
    def t_out_dim(self):
        return reduce(mul, self._t_target.get_shape().as_list()[1:]) 

    @lazy_property
    def fine_delta_t_with_coarse_t_out(self):
        coarse_t = tf.stop_gradient(self.coarse_delta_t_out)
        x = tf.concat(values=[self.img_cnn_out, coarse_t], axis=1)
        # x = tf.concat(values=[self.img_cnn_out, self.coarse_delta_t_out], axis=1)

        for uints in self._delta_t_net:
            x = tf.keras.layers.Dense(
                uints,
                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                activation=tf.nn.relu
            )(x)
            if self._batch_normalization:
                x = tf.keras.layers.BatchNormalization()(x, training=self._is_training)
            if self._drop_out:
                # x = tf.nn.dropout(x, rate=0.2)
                x = tf.keras.layers.Dropout(rate = 0.2)(x, training = self._is_training)

        t_out = tf.keras.layers.Dense(
            self.t_out_dim,   
            activation=tf.nn.tanh,
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
            name ='fine_coarse_t_out'
        )(x)
        return t_out

    @lazy_property
    def counted_result(self):
        if self._t_distr:
            counted_result = tf.reshape(self._visable_amount >= 0.1, (-1,1, 1))
        else:
            counted_result = tf.reshape(self._visable_amount >= 0.1, (-1,1))
        counted_result = tf.cast(counted_result, dtype = tf.float32)
        # counted_result = tf.where(tf.contrib.layers.flatten(self._visable_amount) >= 0.1)
        return counted_result

    @lazy_property
    def effec_t_target(self):
        if self._visable_amount is not None:
            t_target = tf.squeeze(self._t_target, axis=2) * self.counted_result
            # t_target = tf.gather_nd(self._t_target, self.counted_result)
        else:
            t_target = tf.squeeze(self._t_target, axis=2)
        return t_target

    @lazy_property
    def effec_t_with_coarse_t(self):
        if self._visable_amount is not None:
            t_pred = self.fine_delta_t_with_coarse_t_out * self.counted_result
            # t_pred = tf.gather_nd(self.fine_delta_t_with_coarse_t_out, self.counted_result)
        else:
            t_pred = self.fine_delta_t_with_coarse_t_out
        return t_pred

    
    @lazy_property
    def fine_delta_t_with_coarse_t_pred_loss(self):
        t_out_flat = tf.keras.layers.Flatten()(self.effec_t_with_coarse_t)
        t_target_flat = tf.keras.layers.Flatten()(self.effec_t_target)
        loss = tf.compat.v1.losses.absolute_difference(
        # loss = tf.compat.v1.losses.mean_squared_error(
            t_target_flat,
            t_out_flat,
            reduction=tf.compat.v1.losses.Reduction.MEAN
        )
        loss *= self.loss_scale
        tf.compat.v1.summary.scalar('fine_t_pred_loss_with_coarse_t', loss)
        self.regress_final_fine_t_error_with_coarse_t
        return loss
        
    @lazy_property
    def fine_delta_t_with_code_out(self):
        code = tf.stop_gradient(self.processed_code)
        x = tf.concat(values=[self.img_cnn_out, code, self.R_processed_code], axis=1)
        for uints in self._delta_t_net:
            x = tf.keras.layers.Dense(
                uints,
                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                activation=tf.nn.relu
            )(x)
            if self._batch_normalization:
                x = tf.keras.layers.BatchNormalization()(x, training=self._is_training)
            if self._drop_out:
                x = tf.keras.layers.Dropout(rate = 0.2)(x, training = self._is_training)
                # x = tf.nn.dropout(x, rate=0.2)
        t_out = tf.keras.layers.Dense(
            self.t_out_dim,   
            activation=tf.nn.tanh,
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
        )(x)
        return t_out

    @lazy_property
    def effec_fine_t_with_code_out(self):
        if self._visable_amount is not None:
            # t_pred = tf.gather_nd(self.fine_delta_t_with_code_out, self.counted_result)
            t_pred = self.fine_delta_t_with_code_out * self.counted_result
        else:
            t_pred = self.fine_delta_t_with_code_out
        return t_pred

    @lazy_property
    def fine_delta_t_with_code_pred_loss(self):
        t_out_flat = tf.keras.layers.Flatten()(self.effec_fine_t_with_code_out)
        t_target_flat = tf.keras.layers.Flatten()(self.effec_t_target)
        loss = tf.compat.v1.losses.absolute_difference(
        # loss = tf.compat.v1.losses.mean_squared_error(
            t_target_flat,
            t_out_flat,
            # reduction=tf.compat.v1.losses.Reduction.NONE
            reduction=tf.compat.v1.losses.Reduction.MEAN
        )
        loss *= self.loss_scale
        # loss,_ = tf.nn.top_k(loss,k=loss.shape[1]//4)
        # loss = tf.reduce_mean(loss)
        tf.compat.v1.summary.scalar('fine_t_pred_loss_concat_code', loss)
        self.regress_final_fine_t_error_with_code
        return loss

    @lazy_property
    def coarse_delta_t_out(self):
        x = self.processed_code
        for uints in self._delta_t_net:
            x = tf.keras.layers.Dense(
                uints,
                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                activation=tf.nn.relu
            )(x)
            if self._batch_normalization:
                x = tf.keras.layers.BatchNormalization()(x, training=self._is_training)
            if self._drop_out:
                x = tf.keras.layers.Dropout(rate = 0.2)(x, training = self._is_training)
                # x = tf.nn.dropout(x, rate=0.2)

        coarse_t_out = tf.keras.layers.Dense(
            self.t_out_dim,   
            activation=tf.nn.tanh,
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
        )(x)
        return coarse_t_out
    
    @lazy_property
    def effec_coarse_t_out(self):
        if self._visable_amount is not None:
            # t_pred = tf.gather_nd(self.coarse_delta_t_out, self.counted_result)
            t_pred = self.coarse_delta_t_out * self.counted_result
        else:
            t_pred = self.coarse_delta_t_out
        return t_pred

    @lazy_property
    def coarse_delta_t_pred_loss(self):
        t_out_flat = tf.keras.layers.Flatten()(self.effec_coarse_t_out)
        t_target_flat = tf.keras.layers.Flatten()(self.effec_t_target)
        loss = tf.compat.v1.losses.absolute_difference(
            t_target_flat,
            t_out_flat,
            reduction=tf.compat.v1.losses.Reduction.MEAN
        )
        loss *= self.loss_scale
        if not self._phase_2:
            loss = loss - loss
        tf.compat.v1.summary.scalar('coarse_delta_t_pred_loss', loss)
        self.regress_final_coarse_t_error
        return loss
    
    @lazy_property
    def global_step(self):
        return tf.Variable(0, dtype=tf.int64, trainable=False, name='global_step')

    @lazy_property
    def loss(self):
        if self._t_distr:
            loss = self.fine_delta_t_distribution_loss + self.coarse_delta_t_distribution_loss 
        else:
            loss = self.fine_delta_t_with_code_pred_loss + self.coarse_delta_t_pred_loss
        if self._visable_amount is not None:
            loss += 0.8*self.vis_amount_loss
        return loss

    @lazy_property
    def processed_code(self):
        x = self._code_input
        if self._batch_normalization:
            x = tf.keras.layers.BatchNormalization()(x, training=self._is_training)
        return x

    @lazy_property
    def R_processed_code(self):
        x = self._R_code_input
        if self._batch_normalization:
            x = tf.keras.layers.BatchNormalization()(x, training=self._is_training)
        return x

    @lazy_property
    def t_distribution_out_dim(self):
        return self.t_distribution_num*self.t_out_dim

    @lazy_property
    def t_distribution_num(self):
        return 30

    @lazy_property
    def other_hypotheses_punish_strength(self):
        return 0.1

    @lazy_property
    def effec_coarse_t_distribution_out(self):
        if self._visable_amount is not None:
            # t_pred = tf.gather_nd(self.coarse_delta_t_distribution_out, self.counted_result)
            t_pred = self.coarse_delta_t_distribution_out * self.counted_result
        else:
            t_pred = self.coarse_delta_t_distribution_out
        return t_pred

    @lazy_property
    def coarse_delta_t_distribution_out(self):
        x = self.processed_code
        for uints in self._delta_t_net:
            x = tf.keras.layers.Dense(
                uints,
                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                activation=tf.nn.relu
            )(x)
            if self._batch_normalization:
                x = tf.keras.layers.BatchNormalization()(x, training=self._is_training)
            if self._drop_out:
                x = tf.keras.layers.Dropout(rate = 0.2)(x, training = self._is_training)
                # x = tf.nn.dropout(x, rate=0.2)

        coarse_t_out = tf.keras.layers.Dense(
            self.t_distribution_out_dim,   
            activation=tf.nn.tanh,
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
        )(x)
        # coarse_t_out = coarse_t_out*self._scale_factor + 0.5 - self._scale_factor/2
        coarse_t_out = tf.reshape(coarse_t_out, (-1, 3, self.t_distribution_num))
        return coarse_t_out

    @lazy_property
    def coarse_delta_t_distribution_loss(self):
        error_table = tf.norm(tensor=self.effec_coarse_t_distribution_out - self.effec_t_target, axis = 1)
        min_error = - tf.nn.top_k(0-error_table, k=1).values
        min_error_hypothesis_loss = tf.reduce_mean(input_tensor=min_error)*self.loss_scale

        other_error = error_table - min_error
        other_hypotheses_loss = tf.reduce_mean(input_tensor=other_error)*self.t_distribution_num/(self.t_distribution_num - 1)*self.loss_scale
        loss = min_error_hypothesis_loss + self.other_hypotheses_punish_strength*other_hypotheses_loss
        tf.compat.v1.summary.scalar('coarse_t_distribution_loss', loss)
        self.final_coarse_t_error
        return loss

    @lazy_property
    def effec_fine_t_distribution_out(self):
        if self._visable_amount is not None:
            # t_pred = tf.gather_nd(self.fine_delta_t_distribution_out, self.counted_result)
            t_pred = self.fine_delta_t_distribution_out * self.counted_result
        else:
            t_pred = self.fine_delta_t_distribution_out
        return t_pred

    @lazy_property
    def fine_delta_t_distribution_out(self):
        code = tf.stop_gradient(self.processed_code)
        x = tf.concat(values=[self.img_cnn_out, code], axis=1)
        # x = tf.concat(values=[self.img_cnn_out, self.processed_code], axis=1)
        for uints in self._delta_t_net:
            x = tf.keras.layers.Dense(
                uints,
                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                activation=tf.nn.relu
            )(x)
            if self._batch_normalization:
                x = tf.keras.layers.BatchNormalization()(x, training=self._is_training)
            if self._drop_out:
                x = tf.keras.layers.Dropout(rate = 0.2)(x, training = self._is_training)
                # x = tf.nn.dropout(x, rate=0.2)
        t_out = tf.keras.layers.Dense(
            self.t_distribution_out_dim,   
            activation=tf.nn.tanh,
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
        )(x)
        # t_out = t_out*self._scale_factor + 0.5 - self._scale_factor/2
        t_out = tf.reshape(t_out, (-1, 3, self.t_distribution_num))
        return t_out

    @lazy_property
    def fine_delta_t_distribution_loss(self):
        error_table = tf.norm(tensor=self.effec_fine_t_distribution_out - self.effec_t_target, axis = 1)
        min_error = - tf.nn.top_k(0-error_table, k=1).values
        min_error_hypothesis_loss = tf.reduce_mean(input_tensor=min_error)*self.loss_scale

        other_error = error_table - min_error
        other_hypotheses_loss = tf.reduce_mean(input_tensor=other_error)*self.t_distribution_num/(self.t_distribution_num - 1)*self.loss_scale
        loss = min_error_hypothesis_loss + self.other_hypotheses_punish_strength*other_hypotheses_loss
        tf.compat.v1.summary.scalar('fine_t_distribution_loss', loss)
        self.final_fine_t_error
        return loss
    

    @lazy_property
    def loss_scale(self):
        if self._visable_amount is not None:
            select_all_sample = tf.reshape(self._visable_amount >= 0, (-1,1))
            total_sample_num = tf.reduce_sum(input_tensor=tf.cast(select_all_sample, dtype= tf.float32))
            scale = total_sample_num/tf.reduce_sum(input_tensor=self.counted_result)
        else:
            scale = 1
        return scale

    @lazy_property
    def ori_img_delta_t_distribution_out(self):
        code = tf.stop_gradient(self.processed_code)
        x = tf.concat(values=[self.ori_img_cnn_out, code], axis=1)
        # x = tf.concat(values=[self.img_cnn_out, self.processed_code], axis=1)
        for uints in self._delta_t_net:
            x = tf.keras.layers.Dense(
                uints,
                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                activation=tf.nn.relu
            )(x)
            if self._batch_normalization:
                x = tf.keras.layers.BatchNormalization()(x, training=self._is_training)
            if self._drop_out:
                x = tf.keras.layers.Dropout(rate = 0.2)(x, training = self._is_training)
                # x = tf.nn.dropout(x, rate=0.2)
        t_out = tf.keras.layers.Dense(
            self.t_distribution_out_dim,   
            activation=tf.nn.tanh,
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
        )(x)
        t_out = tf.reshape(t_out, (-1, 3, self.t_distribution_num))
        return t_out

    @lazy_property
    def ori_img_delta_t_distribution_loss(self):
        # TODO: error table should be calculated by using effective out and target
        error_table = tf.norm(tensor=self.ori_img_delta_t_distribution_out - self._t_target, axis = 1)
        min_error = - tf.nn.top_k(0-error_table, k=1).values
        min_error_hypothesis_loss = tf.reduce_mean(input_tensor=min_error)*self.loss_scale

        other_error = error_table - min_error
        other_hypotheses_loss = tf.reduce_mean(input_tensor=other_error)*self.t_distribution_num/(self.t_distribution_num - 1)*self.loss_scale
        loss = min_error_hypothesis_loss + self.other_hypotheses_punish_strength*other_hypotheses_loss
        tf.compat.v1.summary.scalar('ori_img_t_distribution_loss', loss)
        self.final_ori_img_fine_t_error
        return loss
    
    @lazy_property
    def final_coarse_t_error(self):
        t_out = tf.reduce_mean(input_tensor=self.coarse_delta_t_distribution_out, axis=2, keepdims = True)
        t_error = tf.norm(tensor=(t_out - self._t_target)*self._max_translation_shift, axis = 1)
        t_error = tf.reduce_mean(input_tensor=t_error)*self.loss_scale
        t_error = tf.stop_gradient(t_error)
        tf.compat.v1.summary.scalar('final_coarse_t_error', t_error)

    @lazy_property
    def final_fine_t_error(self):
        t_out = tf.reduce_mean(input_tensor=self.fine_delta_t_distribution_out, axis=2, keepdims = True)
        t_error = tf.norm(tensor=(t_out - self._t_target)*self._max_translation_shift, axis = 1)
        t_error = tf.reduce_mean(input_tensor=t_error)*self.loss_scale
        t_error = tf.stop_gradient(t_error)
        tf.compat.v1.summary.scalar('final_fine_t_error', t_error)

    @lazy_property
    def final_ori_img_fine_t_error(self):
        t_out = tf.reduce_mean(input_tensor=self.ori_img_delta_t_distribution_out, axis=2, keepdims = True)
        t_error = tf.norm(tensor=(t_out - self._t_target)*self._max_translation_shift, axis = 1)
        t_error = tf.reduce_mean(input_tensor=t_error)*self.loss_scale
        t_error = tf.stop_gradient(t_error)
        tf.compat.v1.summary.scalar('final_ori_img_t_error', t_error)
    
    @lazy_property
    def regress_final_coarse_t_error(self):
        t_error = tf.norm(tensor=(self.effec_coarse_t_out - tf.reshape(self.effec_t_target, (-1,3)))*self._max_translation_shift, axis = 1)
        t_error = tf.reduce_mean(input_tensor=t_error)*self.loss_scale
        t_error = tf.stop_gradient(t_error)
        if not self._phase_2:
            t_error = 0
        tf.compat.v1.summary.scalar('regress_final_coarse_t_error', t_error)

    @lazy_property
    def regress_final_fine_t_error_with_code(self):
        t_error = tf.norm(tensor=(self.effec_fine_t_with_code_out - tf.reshape(self.effec_t_target,(-1,3)))*self._max_translation_shift, axis = 1)
        t_error = tf.reduce_mean(input_tensor=t_error)*self.loss_scale
        t_error = tf.stop_gradient(t_error)
        tf.compat.v1.summary.scalar('regress_final_fine_t_error_with_code', t_error)

    @lazy_property
    def regress_final_fine_t_error_with_coarse_t(self):
        t_error = tf.norm(tensor=(self.effec_t_with_coarse_t - tf.reshape(self.effec_t_target, (-1,3)))*self._max_translation_shift, axis = 1)
        t_error = tf.reduce_mean(input_tensor=t_error)*self.loss_scale
        t_error = tf.stop_gradient(t_error)
        tf.compat.v1.summary.scalar('regress_final_fine_t_error_with_coarse_t', t_error)
    
    @lazy_property
    def all_pred_result(self):
        fine_t = self.fine_delta_t_with_code_out
        coarse_t = self.coarse_delta_t_out
        vis_amount = self.vis_amount_pred
        return fine_t, coarse_t, vis_amount

