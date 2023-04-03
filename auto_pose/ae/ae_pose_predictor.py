# -*- coding: utf-8 -*-

import tensorflow as tf
import ae_factory as factory
import numpy as np

from utils import lazy_property

class PosePredictor(object):

    def __init__(self,experiment_name, dataset_path, args,per_process_gpu_memory_fraction, ckpt_dir, is_training=False, combiner_training = False):
        self.graph = tf.Graph()
        config = factory.config_GPU(per_process_gpu_memory_fraction)
        self.sess = tf.compat.v1.Session(graph=self.graph, config=config)
        self._is_training = is_training
        self._combiner_training = combiner_training
        with self.graph.as_default():
            if is_training:
                self.dataset, self.queue, self.training, self.codebook, \
                self.predictor, self.R_encoder, self.R_decoder, \
                self.t_encoder, self.t_decoder, self.ae, self.train_op, self.saver = factory.build_train_architecture(experiment_name, dataset_path, args, is_training=True)
                self.x = self.queue.x
            else:
                self.codebook, self.predictor, self.R_encoder, self.R_decoder,\
                self.t_encoder, self.t_decoder, self.dataset, self.x, self.no_rot_x = factory.build_inference_architecture(experiment_name, dataset_path, args, use_combiner_queue= combiner_training)
                self.saver = tf.compat.v1.train.Saver(save_relative_paths=True)
            # if self._combiner_training:
            #     self.dataset.load_augmentation_images()
        with self.sess.as_default():
            chkpt = tf.train.get_checkpoint_state(ckpt_dir)
            if chkpt and chkpt.model_checkpoint_path:
                self.saver.restore(self.sess, chkpt.model_checkpoint_path)
            else:
                self.sess.run(tf.compat.v1.global_variables_initializer())
    
    def predict(self, x, x_no_rot):
        if x.ndim == 3:
            x = np.expand_dims(x, axis=0)
        delta_t, _, vis_amnt = self.sess.run(self.predictor.all_pred_result,feed_dict={self.x: x, self.no_rot_x: x_no_rot})
        x = x/255
        pred_R, similarity_score, confidence = self.codebook.nearest_rotation(self.sess, x, return_similarity_score=True, return_confidence= True)
        return (vis_amnt, delta_t, pred_R, similarity_score, confidence)
    
    def new_predict(self, rgb_img, depth_img, prior_R, prior_t, K, no_rot_depth = False):
        eval_x = self.prior_info_embedding(rgb_img, depth_img, prior_R, prior_t, K, no_rot_depth = no_rot_depth)
        return self.predict(eval_x, eval_x)

    def prior_info_embedding(self, rgb_img, depth_img, prior_R, prior_t, K, no_rot_depth = False):
        if rgb_img.ndim != 3:
            print('Error: unsupported rgb image type. Input RGB dimention:', rgb_img.ndim, 'Required RGB dimention: 3.')
            exit()
        eval_x, _, no_rot_depth_patch = self.dataset.train_images_preprocess(
            np.expand_dims(rgb_img, axis = 0),
            np.expand_dims(depth_img, axis=0),
            np.expand_dims(prior_R, axis = 0),
            np.expand_dims(prior_t, axis = 0),
            np.expand_dims(K, axis=0),
            rotation_target=False, 
            return_no_rot_patch= True,
            broadcast=True)
        if no_rot_depth:
            eval_x[:,:,:,3:6] = no_rot_depth_patch
        pred_img = self.get_render_image_patch(prior_R, prior_t, K, no_rot_depth = no_rot_depth)
        eval_x = np.concatenate((eval_x, pred_img), axis = 0)
        return eval_x

    @property
    def all_result(self):
        delta_t, _, vis_amnt = self.predictor.all_pred_result
        latent_code = self.encoder.z
        return (delta_t, _, vis_amnt, latent_code)

    
    def re_detection_predict(self, x, x_no_rot):
        if x.ndim == 3:
            x = np.expand_dims(x, axis=0)
        delta_t, _, vis_amnt = self.sess.run(self.predictor.all_pred_result,feed_dict={self.x: x, self.no_rot_x: x_no_rot})
        x = x/255
        x[:,:,:,3:6] = 0
        pred_R, similarity_score, confidence = self.codebook.rgb_nearest_rotation(self.sess, x, return_similarity_score=True, return_confidence= True)
        return (vis_amnt, delta_t, pred_R, similarity_score, confidence)

    def get_rotation_reconstr_image(self, x = None):
        if x is not None:
            if x.ndim == 3:
                x = np.expand_dims(x, axis=0)
            img = self.sess.run(self.R_decoder.x, feed_dict={self.x: x})
        elif self.queue is not None:
            img = self.sess.run(self.R_decoder.x)
        else:
            print('Error: get_rotation_reconstr_image(): Please feed input image for rotation reconstruction')
            exit()
        return (img*255).astype(np.uint8)
    
    def train_ingredients_for_combiner(self, x= None):
        if x is not None:
            if x.ndim == 3:
                x = np.expand_dims(x, axis=0)
            vis_amnt, code = self.sess.run(self.rotation_code_and_vis_amnt, feed_dict={self.x:x})
        else:
            vis_amnt, code = self.sess.run(self.rotation_code_and_vis_amnt)
        sample_num = int(len(code)/4)
        sample = np.concatenate((vis_amnt.reshape((-1,1)), code), axis = 1)
        sample_len = sample.shape[-1]
        sample = sample.reshape((sample_num, -1))
        train_x = sample[:,:-sample_len]
        train_y = sample[:,-(sample_len-1):]
        return (train_x, train_y)
    
    @property
    def rotation_code_and_vis_amnt(self):
        return (self.predictor.vis_amount_pred, self.R_encoder.z)
    
    def queue_start(self):
        self.queue.start(self.sess)

    def queue_stop(self):
        self.queue.stop(self.sess)

    def get_train_x(self):
        train_x = self.sess.run(self.queue.x)
        return train_x
    
    def get_prediction_result_for_combiner(self, x):
        vis_amnt, delta_t, code = self.sess.run(self.code_and_t_and_vis_amnt, feed_dict={self.x: x})
        sample_num = int(len(code)/3)
        code = np.concatenate((vis_amnt.reshape((-1,1)), code), axis = 1)
        code = code.reshape((sample_num, -1))
        return (vis_amnt[1], delta_t[1], code)


    @property
    def code_and_t_and_vis_amnt(self):
        vis_amnt = self.predictor.vis_amount_pred
        delta_t = self.predictor.fine_delta_t_with_code_out
        code = self.R_encoder.z
        return (vis_amnt, delta_t, code)
    
    def rotation_from_code(self, code):
        R = self.codebook.get_rotation_from_code(self.sess, code)
        return R

    @property
    def translation_shift(self):
        import itertools
        delta_ts = np.empty((26,3),dtype=float)
        i=0
        expand_range = int(self.dataset.max_delta_t_shift*2)
        for delta_t in itertools.product(np.arange(-expand_range, expand_range+1, expand_range), repeat=3):
            if (delta_t == np.array([0,0,0])).all():
                continue
            delta_ts[i] = delta_t
            i += 1
        delta_ts = delta_ts.reshape((-1,)+self.dataset.t_shape)
        return delta_ts

    @property
    def rotation_euler_angle_shift(self):
        import itertools
        delta_Rs = np.empty((64,3, 3),dtype=float)
        i=0
        for delta_t in itertools.product(np.arange(-90, 181, 90), repeat=3):
            delta_Rs[i] = transform.euler_matrix(delta_t[0], delta_t[1], delta_t[2])[:3,:3]
            i += 1
        return delta_ts

    def get_render_possbility(self, R):
        img_patch = self.dataset.generate_rotation_image_patch(R, R, target = False)
        img_patch = img_patch/255.
        _, similarity_score = self.codebook.nearest_rotation(self.sess, img_patch, return_similarity_score=True)
        return similarity_score
    
    def get_rotation_render_image(self, R):
        img_patch = self.dataset.generate_rotation_image_patch(R, R, target = False)
        return np.expand_dims(img_patch, axis=0)

    def get_render_image_patch(self, R, t, K, no_rot_depth = False):
        rgb, depth, _, K = self.dataset.generate_synthetic_image_crop(t, R, K)
        img_patch, _, no_rot_depth_patch = self.dataset.train_images_preprocess(
            np.expand_dims(rgb, axis=0),
            np.expand_dims(depth, axis=0),
            np.expand_dims(R, axis=0),
            np.expand_dims(t, axis=0),
            np.expand_dims(K, axis=0),
            return_no_rot_patch= True,
            broadcast = True)
        if img_patch.ndim == 3:
            img_patch = np.expand_dims(img_patch, axis=0)
        if no_rot_depth:
            img_patch[:,:,:,3:6] = no_rot_depth_patch
        return img_patch
    
    def correct_rotation_shift(self, obs_R, est_t):
        z_axis = np.array([0,0,1])
        R_shift = self.dataset.get_rotation_from_vectors(est_t, z_axis)
        new_R = np.matmul(R_shift.T, obs_R)
        return new_R
    
    def init_pred_t(self, shape):
        return (self.dataset.render_t).reshape(shape)
    
    def warming(self):
        #TODO: move warming code here.
        return None
