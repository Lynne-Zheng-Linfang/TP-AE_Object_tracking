# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tf_siren.siren import SinusodialRepresentationDense

from utils import lazy_property

class SirenDecoder(object):
    def __init__(self, reconstruction_target, latent_code, num_filters, 
                kernel_size, strides, loss, bootstrap_ratio, 
                auxiliary_mask, batch_norm, is_training=False, mask_target = None, visable_amount_target = None):
        self._reconstruction_target = reconstruction_target
        self._latent_code = latent_code
        self._auxiliary_mask = auxiliary_mask
        if self._auxiliary_mask:
            self._target_mask = mask_target
            self._xmask = None
        self._num_filters = num_filters
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
    def x(self):
        z = self._latent_code
        print('code_shape', z.shape)

        h, w, c = self._reconstruction_target.get_shape().as_list()[1:]
        # print(h,w,c)
        layer_dimensions = [ [int(h/np.prod(self._strides[i:])), int(w/np.prod(self._strides[i:]))]  for i in range(len(self._strides))]

        x = SinusodialRepresentationDense(w*h*self._num_filters[0], w0=30.0)(self._latent_code)  
        x = tf.reshape(x, (-1,w, h, self._num_filters[0]))
        # x = SinusodialRepresentationDense(layer_dimensions[0][0]*layer_dimensions[0][1]*self._num_filters[0], w0=30.0)(self._latent_code)  
        # x = tf.reshape(x, (-1,layer_dimensions[0][0], layer_dimensions[0][1], self._num_filters[0]))
        print('first layer shape:', x.get_shape())
        for filters, layer_size in zip(self._num_filters[1:], layer_dimensions[1:]):
            # x = tf.image.resize_nearest_neighbor(x, layer_size)
            x = SinusodialRepresentationDense(filters, w0=1.0)(x)  
            print('layer shape:', x.get_shape())

        # x = tf.image.resize_nearest_neighbor( x, [h, w] )
        x = SinusodialRepresentationDense(c, w0=1.0)(x)  

        print('decoder output shape:', x.get_shape())
        # z = self._latent_code
        # print(z.shape)

        # h, w, c = self._reconstruction_target.get_shape().as_list()[1:]
        # print(h,w,c)
        # # layer_dimensions = [ [int(h/np.prod(self._strides[i:])), int(w/np.prod(self._strides[i:]))]  for i in range(len(self._strides))]
        # layer_dimensions = []
        # layer_dimensions.append(int(h*w/16))
        # neuron = layer_dimensions[0]
        # for i in range(4):
        #     neuron = int(neuron/2)
        #     layer_dimensions.append(neuron)
        # layer_dimensions.reverse()
        # print(layer_dimensions)

        # x = SinusodialRepresentationDense(layer_dimensions[0], w0=30.0)(self._latent_code)  
        # for  layer_size in layer_dimensions[1:]:
        #     x = SinusodialRepresentationDense(layer_size, w0=1.0)(x)  
        
            
        # x = SinusodialRepresentationDense(h*w*c, w0=1.0)(x)  
        # x = tf.reshape(x, (-1, h, w, c))

        # print('decoder output shape:', x.get_shape())

        return x

    @lazy_property
    def total_sample_num(self):
        select_all_sample = tf.reshape(self._visable_amount_target >= 0, (-1,1))
        total_sample_num = tf.reduce_sum(tf.cast(select_all_sample, dtype= tf.float32))
        tf.cast(total_sample_num, tf.int32)
        return total_sample_num

    @lazy_property
    def visable_loss(self):
        target_visable_mask = tf.reduce_any(self._reconstruction_target[:,:,:,0:3]>0.001, axis=3, keep_dims=True)
        pred_visable_mask =tf.reduce_any(self.x>0.075, axis=3, keep_dims=True) 
        visable_mask = tf.cast(tf.math.logical_or(target_visable_mask, pred_visable_mask), dtype=tf.float32)
        pred_pixels = visable_mask * self.x
        rgb_flat = tf.contrib.layers.flatten(pred_pixels[:,:,:,0:3])
        depth_flat = tf.contrib.layers.flatten(pred_pixels[:,:,:,3:])
        target_pixels = visable_mask* self._reconstruction_target

        rgb_target_flat = tf.contrib.layers.flatten(target_pixels[:,:,:,0:3])
        depth_target_flat = tf.contrib.layers.flatten(target_pixels[:,:,:,3:])

        l2_rgb = tf.losses.mean_squared_error (
            rgb_target_flat,
            rgb_flat,
            reduction=tf.compat.v1.losses.Reduction.NONE
        )

        l2_depth = tf.losses.mean_squared_error (
            depth_target_flat,
            depth_flat,
            reduction=tf.compat.v1.losses.Reduction.NONE
        )
        return l2_rgb, l2_depth

    @lazy_property
    def silhoette_loss(self):
        target_visable_mask = tf.reduce_any(self._reconstruction_target[:,:,:,0:3]>0.0001, axis=3, keep_dims=True)
        pred_visable_mask = tf.reduce_any(self.x>0.075, axis=3, keep_dims=True)
        silhoette_mask = tf.cast(tf.math.logical_xor(target_visable_mask, pred_visable_mask), dtype=tf.float32)
        pred_pixels = silhoette_mask*self.x
        target_pixels = silhoette_mask * self._reconstruction_target
        rgb_flat = tf.contrib.layers.flatten(pred_pixels[:,:,:,0:3])
        depth_flat = tf.contrib.layers.flatten(pred_pixels[:,:,:,3:])

        rgb_target_flat = tf.contrib.layers.flatten(target_pixels[:,:,:,0:3])
        depth_target_flat = tf.contrib.layers.flatten(target_pixels[:,:,:,3:])

        l2_rgb = tf.losses.mean_squared_error (
            rgb_target_flat,
            rgb_flat,
            reduction=tf.compat.v1.losses.Reduction.NONE
        )
        l2_depth = tf.losses.mean_squared_error (
            depth_target_flat,
            depth_flat,
            reduction=tf.compat.v1.losses.Reduction.NONE
        )
        return l2_rgb, l2_depth

        
    @lazy_property
    def reconstr_loss(self):
        print(self.x.shape)
        h, w, c = self._reconstruction_target.get_shape().as_list()[1:]
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
                rgb_flat = tf.contrib.layers.flatten(x[:,:,:,0:3])
                depth_flat = tf.contrib.layers.flatten(x[:,:,:,3:])

                reconstruction_rgb_target_flat = tf.contrib.layers.flatten(target[:,:,:,0:3])
                reconstruction_depth_target_flat = tf.contrib.layers.flatten(target[:,:,:,3:])

                l2_rgb = tf.losses.mean_squared_error(
                    reconstruction_rgb_target_flat,
                    rgb_flat,
                    reduction=tf.compat.v1.losses.Reduction.NONE
                )

                l2_depth = tf.losses.mean_squared_error (
                    reconstruction_depth_target_flat,
                    depth_flat,
                    reduction=tf.compat.v1.losses.Reduction.NONE
                )

                l2_rgb_val,_ = tf.nn.top_k(l2_rgb,k=l2_rgb.shape[1]//self._bootstrap_ratio)
                l2_depth_val,_ = tf.nn.top_k(l2_depth,k=l2_depth.shape[1]//self._bootstrap_ratio)
                
                #TODO: l2_value should be changed
                l2_val = tf.concat((l2_rgb_val, l2_depth_val), 1)
 
                loss = tf.reduce_mean(l2_val)
            else:
                # target_flat = tf.contrib.layers.flatten(target)
                # x_flat = tf.contrib.layers.flatten(x)
                target_flat = tf.transpose(target, (0,3,1,2))
                target_flat = tf.reshape(target_flat, (-1, c, w*h))
                x_flat = tf.transpose(x, (0,3,1,2))
                x_flat = tf.reshape(x_flat, (-1, c, w*h))
                loss = tf.compat.v1.losses.mean_squared_error(
                    target_flat,
                    x_flat,
                    reduction=tf.losses.Reduction.MEAN
                )
        elif self._loss == 'L1':
            if self._bootstrap_ratio > 1:

                x_flat = tf.contrib.layers.flatten(x)
                reconstruction_target_flat = tf.contrib.layers.flatten(target)
                l1 = tf.losses.absolute_difference(
                    reconstruction_target_flat,
                    x_flat,
                    reduction=tf.losses.Reduction.NONE
                )
                print(l1.shape)
                l1_val,_ = tf.nn.top_k(l1,k=l1.shape[1]/self._bootstrap_ratio)
                loss = tf.reduce_mean(l1_val)
            else:
                x_flat = tf.contrib.layers.flatten(x)
                reconstruction_target_flat = tf.contrib.layers.flatten(target)
                l1 = tf.losses.absolute_difference(
                    reconstruction_target_flat,
                    x_flat,
                    # reduction=tf.losses.Reduction.MEAN
                    reduction=tf.losses.Reduction.NONE
                )
                loss = tf.reduce_sum(l1_val/c)
        else:
            print('ERROR: UNKNOWN LOSS ', self._loss)
            exit()


        tf.compat.v1.summary.scalar('reconst_loss', loss)

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
            mask_loss = tf.losses.mean_squared_error(
                tf.cast(target_mask,tf.float32),
                tf.squeeze(xmask, axis=3),
                reduction=tf.losses.Reduction.MEAN
            )
            if self._visable_amount_target is not None:
                mask_loss = mask_loss*self.total_sample_num/tf.reduce_sum(counted_images)
            loss += mask_loss
            tf.summary.scalar('mask_loss', mask_loss)
        
        return loss
