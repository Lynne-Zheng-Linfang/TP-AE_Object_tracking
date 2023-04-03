# -*- coding: utf-8 -*-

import threading

import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

from utils import lazy_property
import time

class Queue(object):

    def __init__(self, dataset, num_threads, queue_size, batch_size, test_set_flag=False):
        self._dataset = dataset
        self._num_threads = num_threads
        self._queue_size = queue_size
        self._batch_size = batch_size
        self._test_set_flag = test_set_flag
        self._train_with_pred = dataset.with_pred 

        if self._train_with_pred:
            input_x_shape = self._dataset.shape[0:2] + (self._dataset.shape[-1]*2,)
        else:
            input_x_shape = self._dataset.shape[0:2] + (self._dataset.shape[-1],)

        # shapes = 3*[self._dataset.shape] + [self._dataset.mask_shape]+ [self._dataset.t_shape] + [self._dataset.visable_amount_shape]

        batch_img_shape = [None]+list(self._dataset.shape)
        batch_input_img_shape = [None]+list(input_x_shape)
        batch_rot_rgb_shape = [None] + list(self._dataset.rot_raw_img_shape)
        batch_rot_depth_shape = [None] + list(self._dataset.rot_raw_img_shape)[:2]
        batch_mask_shape = [None] + list(self._dataset.mask_shape)
        batch_R_shape = [None] + list(self._dataset.R_shape)
        batch_t_shape = [None]+list(self._dataset.t_shape)
        batch_visable_amount_shape = [None] + list(self._dataset.visable_amount_shape)
        
        # train_x, train_y, rot_x,  rot_rgbs, rot_depths, obj_visable_mask, delta_ts, noisy_Rs, visable_amount = self._placeholders
        self._placeholders = [
            tf.compat.v1.placeholder(dtype=tf.uint8, shape=batch_img_shape),
            tf.compat.v1.placeholder(dtype=tf.uint8, shape=batch_img_shape),
            tf.compat.v1.placeholder(dtype=tf.uint8, shape=batch_img_shape),
            tf.compat.v1.placeholder(dtype=tf.uint8, shape=batch_img_shape),
            tf.compat.v1.placeholder(dtype=tf.float32, shape=batch_rot_depth_shape),
            tf.compat.v1.placeholder(dtype=tf.bool, shape=batch_mask_shape),
            tf.compat.v1.placeholder(dtype=tf.bool, shape=batch_mask_shape),
            tf.compat.v1.placeholder(dtype=tf.float32, shape=batch_t_shape),
            tf.compat.v1.placeholder(dtype=tf.float32, shape=batch_R_shape),
        ]

        datatypes = ['uint8', 'uint8', 'uint8','uint8','bool', 'float32', 'float32']
        shapes = [input_x_shape] + 3*[self._dataset.shape] + [self._dataset.mask_shape]+ [self._dataset.t_shape] + [self._dataset.visable_amount_shape]
        self._queue = tf.queue.FIFOQueue(self._queue_size, datatypes, shapes=shapes)

        self.x, self.y, self.rot_x, self.rot_y, self.mask, self.delta_t, self.visable_amount= self._queue.dequeue_up_to(self._batch_size)

        self.enqueue_op = self._queue.enqueue_many(self.map_func())

        self._coordinator = tf.train.Coordinator()

        self._threads = []


    def start(self, session):
        assert len(self._threads) == 0
        tf.compat.v1.train.start_queue_runners(session, self._coordinator)
        for _ in range(self._num_threads):
            thread = threading.Thread(
                        target=Queue.__run__, 
                        args=(self, session)
                        )
            thread.deamon = True
            thread.start()
            self._threads.append(thread)


    def stop(self, session):
        self._coordinator.request_stop()
        session.run(self._queue.close(cancel_pending_enqueues=True))
        self._coordinator.join(self._threads)
        self._threads[:] = []


    def __run__(self, session):
        while not self._coordinator.should_stop():        
            batch = self._dataset.batch_dynamic_complete_acc(self._batch_size, test=self._test_set_flag)
            
            feed_dict = { k:v for k,v in zip( self._placeholders, batch ) }
            try:
                session.run(self.enqueue_op, feed_dict)
                # print 'enqueued something'
            except tf.errors.CancelledError as e:
                print('worker was cancelled')
                pass

    @tf.function
    def map_func(self):
        # train_x, train_y, rot_x, rot_rgbs, rot_depths, obj_visable_mask, gt_mask, delta_ts, noisy_Rs = self._placeholders
        train_x, train_y, rot_x, rot_y, rot_depths, obj_visable_mask, gt_mask, delta_ts, noisy_Rs = self._placeholders
# 
        # rot_y = self._dataset.rotation_target_image_preprocess_tf(rot_rgbs, rot_depths, noisy_Rs)

        # if self._train_with_pred:
        #     pred_y = self._dataset.rotation_target_image_preprocess_tf(pred_rgbs, pred_depths, noisy_Rs)
        #     train_x = tf.concat((train_x, pred_y), axis = -1)
# 
        # train_x, train_x_vis_mask = self._dataset.image_patch_preprocess_tf(train_rgb_patch, train_depth_patch, noisy_Rs, noisy_ts)
        # train_x = self._dataset.wrapped_dataset_augmentation(train_x)[0] 
# 
        # train_y, train_y_vis_mask = self._dataset.image_patch_preprocess_tf(gt_rgb_patch, gt_depth_patch, noisy_Rs, noisy_ts)
# 
        # gt_mask = tf.reduce_any(tf.cast(train_y, dtype=tf.bool), axis=3)
        gt_mask_pixel_num  = tf.reshape(tf.reduce_sum(tf.cast(gt_mask, dtype = tf.float32), axis=[1,2]), (-1,1))
        visable_pixel_num  = tf.reshape(tf.reduce_sum(tf.cast(obj_visable_mask, dtype = tf.float32), axis=[1,2]), (-1,1))
        visable_amount = visable_pixel_num/gt_mask_pixel_num

        if not self._dataset._class_t:
            delta_ts = -delta_ts/self._dataset.max_delta_t_shift
    #   
        return (train_x, train_y, rot_x, rot_y, obj_visable_mask, delta_ts, visable_amount)

class CombinerQueue(object):

    def __init__(self, dataset, num_threads, queue_size, batch_size, test_set_flag=False):
        self._dataset = dataset
        self._num_threads = num_threads
        self._queue_size = queue_size
        self._batch_size = batch_size
        self._render_batch_size = int(batch_size/4)
        self._test_set_flag = test_set_flag

        self._dataset.batch_size = self._render_batch_size

        # shapes = 3*[self._dataset.shape] + [self._dataset.mask_shape]+ [self._dataset.t_shape] + [self._dataset.visable_amount_shape]

        batch_img_shape = [None]+list(self._dataset.shape)
        batch_rot_rgb_shape = [None] + list(self._dataset.rot_raw_img_shape)
        batch_rot_depth_shape = [None] + list(self._dataset.rot_raw_img_shape)[:2]
        batch_mask_shape = [None] + list(self._dataset.mask_shape)
        batch_R_shape = [None] + list(self._dataset.R_shape)
        batch_t_shape = [None]+list(self._dataset.t_shape)
        batch_visable_amount_shape = [None] + list(self._dataset.visable_amount_shape)
        
        # train_x, train_y, rot_rgbs, rot_depths, pred_rgbs, pred_depths, obj_visable_mask, delta_ts, noisy_Rs, visable_amount = self._placeholders
        self._placeholders = [
            tf.compat.v1.placeholder(dtype=tf.uint8, shape=batch_img_shape),
            tf.compat.v1.placeholder(dtype=tf.uint8, shape=batch_img_shape),
            tf.compat.v1.placeholder(dtype=tf.uint8, shape=batch_rot_rgb_shape),
            tf.compat.v1.placeholder(dtype=tf.float32, shape=batch_rot_depth_shape),
            tf.compat.v1.placeholder(dtype=tf.uint8, shape=batch_rot_rgb_shape),
            tf.compat.v1.placeholder(dtype=tf.float32, shape=batch_rot_depth_shape),
            tf.compat.v1.placeholder(dtype=tf.bool, shape=batch_mask_shape),
            tf.compat.v1.placeholder(dtype=tf.bool, shape=batch_mask_shape),
            tf.compat.v1.placeholder(dtype=tf.float32, shape=batch_t_shape),
            tf.compat.v1.placeholder(dtype=tf.float32, shape=batch_R_shape),
        ]

        datatypes = ['uint8']
        shapes = [self._dataset.shape] 
        self._queue = tf.queue.FIFOQueue(self._queue_size, datatypes, shapes=shapes)
        self.x = self._queue.dequeue_up_to(self._batch_size)
        self.enqueue_op = self._queue.enqueue_many(self.map_func())
        self._coordinator = tf.train.Coordinator()
        self._threads = []


    def start(self, session):
        assert len(self._threads) == 0
        tf.compat.v1.train.start_queue_runners(session, self._coordinator)
        for _ in range(self._num_threads):
            thread = threading.Thread(
                        target=CombinerQueue.__run__, 
                        args=(self, session)
                        )
            thread.deamon = True
            thread.start()
            self._threads.append(thread)

    def stop(self, session):
        self._coordinator.request_stop()
        session.run(self._queue.close(cancel_pending_enqueues=True))
        self._coordinator.join(self._threads)
        self._threads[:] = []

    def __run__(self, session):
        while not self._coordinator.should_stop():        
            batch = self._dataset.batch_dynamic_complete_pred(self._render_batch_size, test=self._test_set_flag)
            
            feed_dict = { k:v for k,v in zip( self._placeholders, batch ) }
            try:
                session.run(self.enqueue_op, feed_dict)
                # print 'enqueued something'
            except tf.errors.CancelledError as e:
                print('worker was cancelled')
                pass

    @lazy_property
    def black_image(self):
        return tf.zeros((self._render_batch_size,) + self._dataset.shape[:-1] + (3,), dtype=tf.uint8)

    @tf.function
    def map_func(self):
        train_x, _, rot_rgbs, rot_depths, pred_rgbs, pred_depths, obj_visable_mask, gt_mask, delta_ts, noisy_Rs = self._placeholders
# 
        rot_y = self._dataset.rotation_target_image_preprocess_tf(rot_rgbs, rot_depths, noisy_Rs)

        pred_y = self._dataset.rotation_target_image_preprocess_tf(pred_rgbs, pred_depths, noisy_Rs)
        
        rgb_x = tf.concat((train_x[:,:,:,0:3], self.black_image), axis = -1)

        train_x = tf.stack((rgb_x,train_x, pred_y, rot_y), axis =1)
        train_x = tf.reshape(train_x, (self._batch_size,) + self._dataset.shape)

        return train_x
        
class LSTMQueue(object):

    def __init__(self, dataset, num_threads, queue_size, batch_size, test_set_flag=False):
        self._dataset = dataset
        self._num_threads = num_threads
        self._queue_size = queue_size
        self._batch_size = batch_size
        self._test_set_flag = test_set_flag

        # data from dataset.batch: (batch_x, batch_y, batch_mask, batch_delta_t, visable_amount)

        datatypes = ['float32', 'float32']

        shapes = [self._dataset.x_shape] + [self._dataset.y_shape]

        batch_x_shape = [None]+list(self._dataset.x_shape)
        batch_y_shape = [None] + list(self._dataset.y_shape)
        
        self._placeholders = 2*[
            tf.compat.v1.placeholder(dtype=tf.float32, shape=batch_x_shape),
            tf.compat.v1.placeholder(dtype=tf.float32, shape=batch_y_shape)
        ]
        self._queue = tf.queue.FIFOQueue(self._queue_size, datatypes, shapes=shapes)
        self.x, self.y = self._queue.dequeue_up_to(self._batch_size)

        self.enqueue_op = self._queue.enqueue_many(self._placeholders)

        self._coordinator = tf.train.Coordinator()

        self._threads = []


    def start(self, session):
        assert len(self._threads) == 0
        tf.compat.v1.train.start_queue_runners(session, self._coordinator)
        for _ in range(self._num_threads):
            thread = threading.Thread(
                        target=LSTMQueue.__run__, 
                        args=(self, session)
                        )
            thread.deamon = True
            thread.start()
            self._threads.append(thread)


    def stop(self, session):
        self._coordinator.request_stop()
        session.run(self._queue.close(cancel_pending_enqueues=True))
        self._coordinator.join(self._threads)
        self._threads[:] = []


    def __run__(self, session):
        while not self._coordinator.should_stop():        
            # a= time.time()
            # print 'batching...'
            batch = self._dataset.batch(self._batch_size, eval_data=self._test_set_flag)
            # print 'batch creation time ', time.time()-a
            
            feed_dict = { k:v for k,v in zip( self._placeholders, batch ) }
            try:
                session.run(self.enqueue_op, feed_dict)
                # print 'enqueued something'
            except tf.errors.CancelledError as e:
                print('worker was cancelled')
                pass