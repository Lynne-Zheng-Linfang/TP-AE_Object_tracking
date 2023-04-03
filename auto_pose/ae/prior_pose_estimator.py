# -*- coding: utf-8 -*-

import tensorflow as tf
import ae_factory as factory
import utils 
import numpy as np
import signal
import cv2
import os
from utils import lazy_property

class PriorPoseEstimator(object):
    def __init__(self, experiment_name, experiment_group, args, dataset_path, lstm_dir, per_process_gpu_memory_fraction, is_training=False):
        self.graph = tf.Graph()
        config = factory.config_GPU(per_process_gpu_memory_fraction)
        self.sess = tf.compat.v1.Session(graph=self.graph, config=config)
        self._ckpt_dir = lstm_dir
        self.checkpoint_file = os.path.join(self._ckpt_dir,'chkpt') 
        self._epoch_num = 0
        self._is_training = is_training
        self._experiment_name = experiment_name
        self._experiment_group = experiment_group
        self._lowest_eval_loss = 100000
        self._eval_stable_times = 0
        self._eval_stale_tolarence = 5
        config = factory.config_GPU(per_process_gpu_memory_fraction)
        self.sess = tf.compat.v1.Session(graph=self.graph, config=config)
        with self.sess.as_default():
            with self.graph.as_default():
                self.dataset, self.module, self.training, self.saver, self.input_x, self.train_op, self.queue, self.test_queue = factory.build_prior_pose_estimator_architecture(args, 
                    self._experiment_name, is_training= self._is_training)
                chkpt = tf.train.get_checkpoint_state(self._ckpt_dir)
                if chkpt and chkpt.model_checkpoint_path:
                    self.saver.restore(self.sess, chkpt.model_checkpoint_path)
                else:
                    self.sess.run(tf.compat.v1.global_variables_initializer())
                if self._is_training:
                    self.dataset.load_dataset()
                    self.merged_loss_summary = tf.compat.v1.summary.merge_all()
                    self.summary_writer = tf.compat.v1.summary.FileWriter(self._ckpt_dir, self.sess.graph)
    

    def queue_start(self):
        self.queue.start(self.sess)
        self.test_queue.start(self.sess)

    def queue_stop(self):
        self.queue.stop(self.sess)
        self.test_queue.stop(self.sess)

    def get_train_x(self):
        train_x = self.sess.run(self.queue.x)
        return train_x

    def get_train_y(self):
        train_y = self.sess.run(self.queue.y)
        return train_y

    def predict(self, x):
        if x.ndim == 3:
            x = np.expand_dims(x, axis=0)
        x = self.dataset.normalize_t(x)
        x = self.dataset.dataset_padding(x, target_len= self.seq_length)
        R, t = self.sess.run(self.module.predict, feed_dict={self.input_x: x, self.training: False})
        t = self.dataset.de_normalize_t(t)
        return (R, t) 

    @property
    def epoch(self):
        return self._epoch_num

    def init_epoch_num(self, steps_for_each_epoch):
        self._epoch_num = int(self.get_global_step()/steps_for_each_epoch)
        return self._epoch_num
    
    def get_global_step(self):
        return self.sess.run(self.module.global_step)
    
    def train(self, train_epoch, steps = 5000, log_write_interval = 100):
        init_epoch_num = self.init_epoch_num(steps)
        if init_epoch_num <= train_epoch:
            self.queue_start()
            for _ in range(init_epoch_num, train_epoch):
                gentle_stop = self.train_one_epoch(steps, log_write_interval)
                if gentle_stop:
                    print('Gentle exit')
                    break
                if self._finish_training():
                    print('Training finished.')
                    break
            if not gentle_stop:
                print('Training finished.')
            self.queue_stop()
            self.summary_writer.close()
        else:
            print('Training finished.')

    
    def train_one_epoch(self, steps, log_write_interval):
        start = self.get_global_step() % steps

        gentle_stop = np.array((1,), dtype=np.bool)
        gentle_stop[0] = False
        def on_ctrl_c(signal, frame):
            gentle_stop[0] = True
        signal.signal(signal.SIGINT, on_ctrl_c)
        bar = utils.progressbar_init('Training Epoch ' + str(self._epoch_num)+':', steps)
        bar.start()
        for i in range(start, steps):
            self.sess.run(self.train_op, feed_dict={self.training: True})
            if i % log_write_interval == 0:
                loss = self.sess.run(self.merged_loss_summary, feed_dict={self.training: True})
                self.summary_writer.add_summary(loss, int(self._epoch_num*steps+i))
            bar.update(i)
            if gentle_stop[0]:
                break
        self.saver.save(self.sess, self.checkpoint_file, global_step=self.module.global_step)
        if not gentle_stop[0]:
            eval_loss = self.evaluation()
            print('Evaluation Loss', self._epoch_num, ':', eval_loss)
            self._update_eval_info(eval_loss)
            self._epoch_num += 1
            
        bar.finish()
        return gentle_stop[0]
    
    def _finish_training(self):
            self._eval_stable_times = 0
    
    def _update_eval_info(self, eval_loss):
        if self._lowest_eval_loss > eval_loss:
            self._lowest_eval_loss = eval_loss
            self._eval_stable_times = 0
        else:
            self._eval_stable_times += 1

    def get_queue_contents(self, test = False):
        if test:
            test_x, test_y = self.dataset.test_batch(10240)
        else:
            test_x, test_y = self.dataset.batch(10240, eval_data=True)
        feed_dict = {
            self.queue.x: test_x, 
            self.queue.y: test_y, 
            self.training: False
            }
        return (test_x, test_y, feed_dict)

    def evaluation(self, test= False):
        test_x, test_y, feed_dict = self.get_queue_contents(test = test)
        if test_x is not None:
            return self.sess.run(self.module.loss,feed_dict=feed_dict)
        else:
            return None
        # self.test_summary_writer.add_summary(loss, int(self._epoch_num))
 
    def test(self):
        losses = []
        while True:
            loss = self.evaluation(test=True)
            if loss is None:
                break
            losses.append(loss)
        print(np.mean(np.array(losses)))

    def get_test_seqs(self, path_id):
        return self.dataset.get_test_path_seqs(path_id)
    
    @lazy_property
    def seq_length(self):
        return self.dataset.seq_length