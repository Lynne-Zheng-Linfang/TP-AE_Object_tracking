# -*- coding: utf-8 -*-

import tensorflow as tf
import ae_factory as factory
import utils 
import numpy as np

import os
from utils import lazy_property

class RotationAutoEncoder(object):

    def __init__(self, args, input_img, target, visible_amount, is_training=False, phase_2 = False):
        self._input = input_img 
        self._phase_2 = phase_2
        self._target = target
        self._visible_amount = visible_amount
        self._is_training = is_training
        self._encoder = factory.build_encoder(self._input, args)
        self._decoder = factory.build_decoder(target, self._encoder, args, is_training=is_training, visable_amount_target = self._visible_amount, phase_2 = self._phase_2)
    
    @lazy_property
    def loss(self):
        return self._decoder.reconstr_loss
    
    @property
    def code(self):
        return self._encoder.z
    
    @property
    def encoder(self):
        return self._encoder
    
    @property
    def decoder(self):
        return self._decoder
    
    @lazy_property
    def global_step(self):
        return tf.Variable(0, dtype=tf.int64, trainable=False, name='global_step')

#     def __init__(self,experiment_name, dataset_path, args,per_process_gpu_memory_fraction, ckpt_dir, is_training=False):
#         self.graph = tf.Graph()
#         self._ckpt_dir = os.path.join(ckpt_dir,'rotation_estimator') 
#         self.checkpoint_file = os.path.join(self._ckpt_dir,'chkpt') 
#         self._epoch_num = 0
#         self._is_training = is_training
#         config = factory.config_GPU(per_process_gpu_memory_fraction)
#         self.sess = tf.compat.v1.Session(graph=self.graph, config=config)
#         with self.sess.as_default():
#             with self.graph.as_default():
#                 self.training, self.combiner, self.train_op, self.saver, self.x, self.target = factory.build_combiner_architecture(experiment_name, dataset_path, args, is_training= is_training)
#                 chkpt = tf.train.get_checkpoint_state(self._ckpt_dir)
#                 if chkpt and chkpt.model_checkpoint_path:
#                     self.saver.restore(self.sess, chkpt.model_checkpoint_path)
#                 else:
#                     self.sess.run(tf.compat.v1.global_variables_initializer())
#                 if self._is_training:
#                     self.merged_loss_summary = tf.compat.v1.summary.merge_all()
#                     self.summary_writer = tf.compat.v1.summary.FileWriter(self._ckpt_dir, self.sess.graph)
    
#     @property
#     def epoch(self):
#         return self._epoch_num

#     def predict(self, x):
#         if x.ndim == 3:
#             x = np.expand_dims(x, axis=0)
#         code = self.sess.run(self.combiner.combiner_pred,feed_dict={self.x: x})
#         return code 
    
#     def init_epoch_num(self, steps_for_each_epoch = 5000):
#         self._epoch_num = int(self.get_global_step()/steps_for_each_epoch)
#         return self._epoch_num
    
#     def get_global_step(self):
#         return self.sess.run(self.combiner.global_step)
    
#     def train(self, data_generator, steps = 5000, log_write_interval = 100):
#         start = self.get_global_step() % steps
#         bar = utils.progressbar_init('Training Epoch ' + str(self._epoch_num)+':', steps)
#         bar.start()
#         for i in range(steps):
#             x, target = data_generator.train_ingredients_for_combiner()
#             self.sess.run(self.train_op, feed_dict={self.training: True, self.x:x, self.target: target})
#             if i % log_write_interval == 0:
#                 loss = self.sess.run(self.merged_loss_summary, feed_dict={self.training: True, self.x:x, self.target: target})
#                 self.summary_writer.add_summary(loss, int(self._epoch_num*steps+i))
#             bar.update(i)
#         self.saver.save(self.sess, self.checkpoint_file, global_step=self.combiner.global_step)
#         self._epoch_num += 1
#         bar.finish()
