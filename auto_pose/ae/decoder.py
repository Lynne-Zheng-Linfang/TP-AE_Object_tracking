# -*- coding: utf-8 -*-

import numpy as np

import tensorflow as tf

from utils import lazy_property

class Decoder(object):

    def __init__(self, reconstruction_target, latent_code, num_filters, 
                kernel_size, strides, loss, bootstrap_ratio, 
                auxiliary_mask, batch_norm, is_training=False, mask_target = None, visable_amount_target = None, drop_out = False, phase_2 = False, gray_bg= False):
        self._reconstruction_target = tf.stop_gradient(tf.cast(reconstruction_target, dtype=tf.float32)/255)
        self._gray_bg = gray_bg
        self._latent_code = latent_code
        self._auxiliary_mask = auxiliary_mask
        self._phase_2 = phase_2
        if self._auxiliary_mask:
            self._target_mask = tf.cast(mask_target, dtype=tf.float32)
            self._xmask = None
        self._num_filters = num_filters
        self._drop_out = drop_out
        self._kernel_size = kernel_size
        self._strides = strides
        self._loss = loss
        self._bootstrap_ratio = bootstrap_ratio
        self._batch_normalization = batch_norm
        self._is_training = is_training
        self._visable_amount_target = visable_amount_target
        self.reconstr_loss

    @property
    def reconstruction_target(self):
        return self._reconstruction_target

    @lazy_property
    def xmask(self):
        return self._xmask

    @lazy_property
    def gray_bg_color(self):
        bg_color = np.array([127,127,127,127,127,127])/255
        return tf.constant(bg_color, dtype=tf.float32)

    @lazy_property
    def x(self):
        z = self._latent_code
        print(z.shape)

        h, w, c = self._reconstruction_target.get_shape().as_list()[1:]
        print(h,w,c)
        layer_dimensions = [ [int(h/np.prod(self._strides[i:])), int(w/np.prod(self._strides[i:]))]  for i in range(len(self._strides))]
        print(layer_dimensions)

        x = tf.keras.layers.Dense(
            layer_dimensions[0][0]*layer_dimensions[0][1]*self._num_filters[0],
            activation=tf.nn.relu,
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
        )(self._latent_code)

        if self._batch_normalization:
            x = tf.keras.layers.BatchNormalization()(x, training=self._is_training)
        if self._drop_out:
            x = tf.keras.layers.Dropout(rate = 0.2)(x, training = self._is_training)
        x = tf.reshape( x, [-1, layer_dimensions[0][0], layer_dimensions[0][1], self._num_filters[0] ] )

        for filters, layer_size in zip(self._num_filters[1:], layer_dimensions[1:]):
            x = tf.image.resize(x, layer_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

            x = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=self._kernel_size,
                padding='same',
                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                activation=tf.nn.relu
            )(x)
            if self._batch_normalization:
                x = tf.keras.layers.BatchNormalization()(x, training=self._is_training)
            if self._drop_out:
                x = tf.keras.layers.Dropout(rate = 0.2)(x, training = self._is_training)
        
        x = tf.image.resize( x, [h, w] , method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        if self._auxiliary_mask:
            self._xmask = tf.keras.layers.Conv2D(
                    filters=1,
                    kernel_size=self._kernel_size,
                    padding='same',
                    kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                    activation=tf.nn.sigmoid
                )(x)

        x = tf.keras.layers.Conv2D(
                filters=c,
                kernel_size=self._kernel_size,
                padding='same',
                kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                activation=tf.nn.sigmoid,
                name = 'decoder_out'
            )(x)
        return x

    @lazy_property
    def total_sample_num(self):
        select_all_sample = tf.reshape(self._visable_amount_target >= 0, (-1,1))
        total_sample_num = tf.reduce_sum(input_tensor=tf.cast(select_all_sample, dtype= tf.float32))
        tf.cast(total_sample_num, tf.int32)
        return total_sample_num

    @lazy_property
    def visable_loss(self):
        target_visable_mask = tf.reduce_any(input_tensor=self._reconstruction_target>0.001, axis=3, keepdims=True)
        pred_visable_mask =tf.reduce_any(input_tensor=self.x>0.075, axis=3, keepdims=True) 
        visable_mask = tf.cast(tf.math.logical_or(target_visable_mask, pred_visable_mask), dtype=tf.float32)
        pred_pixels = visable_mask * self.x
        rgb_flat = tf.keras.layers.Flatten()(pred_pixels[:,:,:,0:3])

        depth_flat = tf.keras.layers.Flatten()(pred_pixels[:,:,:,3:])
        target_pixels = visable_mask* self._reconstruction_target

        rgb_target_flat = tf.keras.layers.Flatten()(target_pixels[:,:,:,0:3])
        depth_target_flat = tf.keras.layers.Flatten()(target_pixels[:,:,:,3:])

        l2_rgb =tf.compat.v1.losses.mean_squared_error(
            rgb_target_flat,
            rgb_flat,
            reduction=tf.compat.v1.losses.Reduction.NONE
        )

        l2_depth = tf.compat.v1.losses.mean_squared_error(
            depth_target_flat,
            depth_flat,
            reduction=tf.compat.v1.losses.Reduction.NONE
        )
        return l2_rgb, l2_depth

    @lazy_property
    def _target_vis_mask(self):
        if self._gray_bg:
            return tf.reduce_any(input_tensor=(tf.math.abs(self._reconstruction_target - self.gray_bg_color)>0.01), axis=3, keepdims=True)
        else:
            return tf.reduce_any(input_tensor=(self._reconstruction_target>0.0001), axis=3, keepdims=True)
    
    @lazy_property
    def _pred_vis_mask(self):
        if self._gray_bg:
            print('use gray background')
            return tf.reduce_any(input_tensor=(tf.math.abs(self.x-self.gray_bg_color)>0.075), axis=3, keepdims=True)
        else:
            print('use black background')
            return tf.reduce_any(input_tensor=(self.x>0.075), axis=3, keepdims=True)
    
    @lazy_property
    def _silho_mask(self):
        silhoette_mask = tf.cast(tf.math.logical_xor(self._target_vis_mask, self._pred_vis_mask), dtype=tf.float32)
        silhoette_mask = tf.tile(silhoette_mask, [1,1,1,3])
        return tf.keras.layers.Flatten()(silhoette_mask)
    
    @lazy_property
    def _vis_mask(self):
        visable_mask = tf.cast(tf.math.logical_or(self._target_vis_mask, self._pred_vis_mask), dtype=tf.float32)
        visable_mask = tf.tile(visable_mask,[1,1,1,3])
        return tf.keras.layers.Flatten()(visable_mask)

    @lazy_property
    def silhoette_loss(self):
        target_visable_mask = tf.reduce_any(input_tensor=self._reconstruction_target>0.0001, axis=3, keepdims=True)
        pred_visable_mask = tf.reduce_any(input_tensor=self.x>0.075, axis=3, keepdims=True)
        silhoette_mask = tf.cast(tf.math.logical_xor(target_visable_mask, pred_visable_mask), dtype=tf.float32)
        pred_pixels = silhoette_mask*self.x
        target_pixels = silhoette_mask * self._reconstruction_target
        rgb_flat = tf.keras.layers.Flatten()(pred_pixels[:,:,:,0:3])
        depth_flat = tf.keras.layers.Flatten()(pred_pixels[:,:,:,3:])

        rgb_target_flat = tf.keras.layers.Flatten()(target_pixels[:,:,:,0:3])
        depth_target_flat = tf.keras.layers.Flatten()(target_pixels[:,:,:,3:])

        l2_rgb = tf.compat.v1.losses.mean_squared_error(
            rgb_target_flat,
            rgb_flat,
            reduction=tf.compat.v1.losses.Reduction.NONE
        )
        l2_depth = tf.compat.v1.losses.mean_squared_error(
            depth_target_flat,
            depth_flat,
            reduction=tf.compat.v1.losses.Reduction.NONE
        )
        return l2_rgb, l2_depth
        
    @lazy_property
    def reconstr_loss(self):
        if self._visable_amount_target is not None:
            # Only images with visable amount greater then 10% should be counted
            counted_images = tf.reshape(self._visable_amount_target >= 0.1, (-1,1,1,1))
            counted_images = tf.cast(counted_images, dtype = tf.float32)
            x = self.x*counted_images
            target = self._reconstruction_target*counted_images
        else:
            x = self.x
            target = self._reconstruction_target

        if self._loss == 'L2':
            if self._bootstrap_ratio > 1:
                rgb_flat = tf.keras.layers.Flatten()(x[:,:,:,0:3])
                depth_flat = tf.keras.layers.Flatten()(x[:,:,:,3:])

                reconstruction_rgb_target_flat = tf.keras.layers.Flatten()(target[:,:,:,0:3])
                reconstruction_depth_target_flat = tf.keras.layers.Flatten()(target[:,:,:,3:])

                l2_rgb = tf.compat.v1.losses.mean_squared_error(
                    reconstruction_rgb_target_flat,
                    rgb_flat,
                    reduction=tf.compat.v1.losses.Reduction.NONE
                )

                l2_depth =tf.compat.v1.losses.mean_squared_error(
                    reconstruction_depth_target_flat,
                    depth_flat,
                    reduction=tf.compat.v1.losses.Reduction.NONE
                )

                if self._phase_2:
                    l2_rgb += (l2_rgb*self._silho_mask+ l2_rgb*self._vis_mask)
                    l2_depth += (l2_depth*self._silho_mask + l2_depth*self._vis_mask)
                    # l2_silho_rgb, l2_silho_depth = self.silhoette_loss
                    # l2_visable_rgb, l2_visable_depth = self.visable_loss
                    # l2_rgb += (l2_silho_rgb + l2_visable_rgb)
                    # l2_depth += (l2_silho_depth + l2_visable_depth)

                l2_rgb_val,_ = tf.nn.top_k(l2_rgb,k=l2_rgb.shape[1]//self._bootstrap_ratio)
                l2_depth_val,_ = tf.nn.top_k(l2_depth,k=l2_depth.shape[1]//self._bootstrap_ratio)
                
                #TODO: l2_value should be changed
                l2_val = tf.concat((l2_rgb_val, l2_depth_val), 1)
 
                loss = tf.reduce_mean(input_tensor=l2_val)
            else:
                loss = tf.compat.v1.losses.mean_squared_error(
                    target,
                    x,
                    reduction=tf.compat.v1.losses.Reduction.MEAN
                )
        elif self._loss == 'L1':
            if self._bootstrap_ratio > 1:

                x_flat = tf.keras.layers.Flatten()(x)
                reconstruction_target_flat = tf.keras.layers.Flatten()(target)
                l1 = tf.compat.v1.losses.absolute_difference(
                    reconstruction_target_flat,
                    x_flat,
                    reduction=tf.compat.v1.losses.Reduction.NONE
                )
                print(l1.shape)
                l1_val,_ = tf.nn.top_k(l1,k=l1.shape[1]/self._bootstrap_ratio)
                loss = tf.reduce_mean(input_tensor=l1_val)
            else:
                x_flat = tf.keras.layers.Flatten()(x)
                reconstruction_target_flat = tf.keras.layers.Flatten()(target)
                l1 = tf.compat.v1.losses.absolute_difference(
                    reconstruction_target_flat,
                    x_flat,
                    reduction=tf.compat.v1.losses.Reduction.MEAN
                )
        else:
            print('ERROR: UNKNOWN LOSS ', self._loss)
            exit()


        # tf.compat.v1.summary.scalar('reconst_loss', loss)

        if self._auxiliary_mask:
        # if False:
            if self._visable_amount_target is not None:
                # Only images with visable amount greater then 10% should be counted
                # xmask = tf.gather_nd(tf.squeeze(self._xmask, axis=3), counted_images)
                # target_mask = tf.gather_nd(self._target_mask, counted_images)
                xmask = self._xmask*counted_images
                counted_images = tf.reshape(self._visable_amount_target >= 0.1, (-1,1,1))
                counted_images = tf.cast(counted_images, dtype = tf.float32)
                target_mask = self._target_mask*counted_images
            else:
                xmask = self._xmask
                target_mask = self._target_mask
            mask_loss = tf.compat.v1.losses.mean_squared_error(
                tf.cast(target_mask,tf.float32),
                tf.squeeze(xmask, axis=3),
                reduction=tf.compat.v1.losses.Reduction.MEAN
            )
            if self._visable_amount_target is not None:
                mask_loss = mask_loss*self.total_sample_num/tf.reduce_sum(input_tensor=counted_images)
            loss += mask_loss
            tf.compat.v1.summary.scalar('mask_loss', mask_loss)
        
        return loss

