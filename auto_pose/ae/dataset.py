# -*- coding: utf-8 -*-

import numpy as np
import time
import hashlib
import glob
import os
import progressbar
import cv2
import random
import math

import tensorflow as tf
from pysixd_stuff import transform
from pysixd_stuff import view_sampler
from utils import lazy_property
from pytless import inout
import im_process
import utils 

class Dataset(object):

    def __init__(self, dataset_path, test_ratio, args, **kw):
        self._kw = kw
        self.args = args 
        self.shape = (int(kw['h']), int(kw['w']), int(kw['c']))
        self.mask_shape = (int(kw['h']), int(kw['w']))

        self.noof_training_imgs = int(kw['noof_training_imgs'])
        self.total_training_images = int(self._kw['noof_training_imgs'])
        self.test_set_indices, self.train_set_indices = self.train_test_sepration(self.noof_training_imgs, test_ratio)

        self.total_irr_img_num = int(kw['noof_bg_imgs'])
        self.noof_aug_imgs = int(kw['noof_bg_imgs']) 

        self.test_aug_img_indices = []
        self.train_aug_img_indices = np.random.permutation(self.noof_aug_imgs)

        self.siren = False
        self.rotation_only_train_mode = False
        self.ycb_real = False 
        self.is_training = False

        self._class_t = False

        self.dataset_path = dataset_path

        self.vsd_eval = False

        self.train_x = [] 
        self.train_y = [] 
        self.train_mask = []
        self.train_gt_R = []
        self.train_gt_t = [] 
        self.train_noisy_R = [] 
        self.train_noisy_t = [] 
        self.train_delta_R = [] 
        self.train_delta_t = []

        self.syn_rot_train_x = []
        self.syn_rot_train_y = []

        self.train_bg_x = []

        self.eval_x = []
        self.eval_gt_R = [] 
        self.eval_gt_t = [] 
        self.eval_noisy_R = []
        self.eval_noisy_t =[]
        self.eval_delta_R = []
        self.eval_delta_t = []
        self.eval_no_rot_x = []

        self.bg_rgb_list = [] 
        self.bg_depth_list = [] 
        self.bg_info_list = []
        self.bg_gt_list = [] 
        
        self.test_pose_pred_set_indices =[] 
        self.train_pose_pred_set_indices = [] 
        self.gen_indices = [] 
        self.gt_mask_pixel_num = []

        self.batch_size = 0

        self.real_image_id = 0 
        self.aug_angle = 0

        self.dynamic_mode = False

        self.aug_rgb_images = [] 
        self.aug_depth_images = [] 
        self.with_pred = False
        self.real_rgbs, self.real_depths = [], []
        self.real_Ks, self.real_Rs, self.real_ts = [], [], []
        # self.load_ycb_real_training_images_of_target()

        # self.load_augmentation_images()

    def load_ycb_real_training_images_of_target(self):
        if len(self.real_rgbs) > 0:
            return None
        print('loading ycb real training images')
        obj_id = self.target_obj_id + 1
        total_num = self.total_training_images 
        data_base = self._kw['foreground_base_dir'] 
        gt_glob = os.path.join(data_base, '*', 'gt.yml')
        gt_paths = glob.glob(gt_glob)
        gt_paths.sort()
        init_flag = True
        gt_id = 0
        for gt_path in gt_paths:
            scene_id = int(os.path.basename(os.path.dirname(gt_path)))
            if (obj_id ==18) and (scene_id in [27, 29, 38]):
                continue
            gts = inout.load_gt(gt_path)
            info_path = os.path.join(os.path.dirname(gt_path), 'info.yml')
            # print(gt_path)
            infos = inout.load_info(info_path)
            # for gt in gts[1]:
            #     if gt['obj_id'] == obj_id:
            #         print(gt_path)
            #         continue

            for img_id in gts.keys():
                for gt in gts[img_id]:
                    if int(gt['obj_id']) == obj_id:
                        img_id = int(img_id)
                        K = infos[img_id]['cam_K']
                        R = gt['cam_R_m2c']
                        t = gt['cam_t_m2c']
                        rgb_path = os.path.join(os.path.dirname(gt_path), 'rgb', str(img_id).zfill(6)+'.png')
                        rgb = cv2.imread(rgb_path)
                        depth_path = os.path.join(os.path.dirname(gt_path), 'depth', str(img_id).zfill(6)+'.png')
                        depth_scale = infos[img_id]['depth_scale']
                        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)*depth_scale
                        H, W = depth.shape
                        if not self.check_t(t, K, W, H):
                            continue
                        # ref_rgb, _, _ = self.render_gt_image(W = W, H = H, K=K, R=R, t=t)
                        # overlapping = cv2.addWeighted(ref_rgb, 0.8, rgb, 0.5, 0)
                        # cv2.imshow('ref', overlapping)
                        # cv2.waitKey(1)
                        # continue
                        if init_flag:
                            self.init_real_datas(total_num, rgb.shape, K.shape, R.shape, t.shape)
                            init_flag = False
                        self.real_Ks[gt_id] = K
                        self.real_Rs[gt_id] = R
                        self.real_ts[gt_id] = t 
                        self.real_rgbs[gt_id] = rgb
                        self.real_depths[gt_id] = depth
                        gt_id += 1
                    if gt_id == total_num:
                        break
                if gt_id == total_num:
                    break
            if gt_id == total_num:
                break
        print('Succeccfully loaded', gt_id, 'real training images.')
        self.total_training_images = gt_id 
    
    def load_tless_real_training_images_of_target(self):
        print('Loading tless real training images...')
        obj_id = self.target_obj_id + 1 
        data_base = self._kw['foreground_base_dir'] 
        gt_path = os.path.join(data_base, str(obj_id).zfill(2), 'gt.yml')
        gts = inout.load_gt(gt_path)
        print(gt_path)
        total_num = len(gts) 
        info_path = os.path.join(os.path.dirname(gt_path), 'info.yml')
        infos = inout.load_info(info_path)
        init_flag = True
        for img_id in range(total_num):
            rgb_path = os.path.join(os.path.dirname(gt_path), 'rgb', str(img_id).zfill(4)+'.png')
            rgb = cv2.imread(rgb_path)
            depth_path = os.path.join(os.path.dirname(gt_path), 'depth', str(img_id).zfill(4)+'.png')
            depth_scale = infos[img_id]['depth_scale']
            depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)*depth_scale
            K = infos[img_id]['cam_K']
            R = gts[img_id][0]['cam_R_m2c'] 
            t = gts[img_id][0]['cam_t_m2c'] 
            if init_flag:
                self.init_real_datas(total_num, rgb.shape, K.shape, R.shape, t.shape)
                init_flag = False
            self.real_Ks[img_id] = K 
            self.real_Rs[img_id] = R 
            self.real_ts[img_id] = t 
            self.real_rgbs[img_id] = rgb
            self.real_depths[img_id] = depth
        print('Successfully loaded', len(self.real_Ks), 'real training images.')

    def init_real_datas(self, total_num, rgb_shape, K_shape, R_shape, t_shape):
        self.real_rgbs = np.empty((total_num,)+rgb_shape, dtype=np.uint8)
        self.real_depths = np.empty((total_num,)+rgb_shape[:-1], dtype = np.float32)
        self.real_Ks = np.empty((total_num,)+K_shape, dtype = np.float32)
        self.real_Rs = np.empty((total_num,)+R_shape, dtype = np.float32) 
        self.real_ts = np.empty((total_num,)+t_shape, dtype = np.float32) 

    def get_rotation_from_vectors(self, a, b):
        eps = 1e-7
        a = a.reshape(3,)
        b = b.reshape(3,)
        a_normalized = a/np.linalg.norm(a)
        b_normalized = b/np.linalg.norm(b)
        if np.linalg.norm(a_normalized - b_normalized) <= eps:
            return np.eye(3)
        cosin_theta = np.matmul(a_normalized, b_normalized)
        angle = np.arccos(cosin_theta)
        axis = np.cross(a_normalized, b_normalized)
        rotation = transform.rotation_matrix(angle, axis)[:3,:3]
        return rotation

    def train_test_sepration(self, total_num, test_ratio):
        test_num = int(total_num*test_ratio)
        train_num = total_num - test_num 
        np.random.seed(12)
        shuffled_indices = np.random.permutation(total_num)
        return shuffled_indices[:test_num], shuffled_indices[test_num:]
    
    def add_rotation_shift(self, obs_R, est_t):
        z_axis = np.array([0,0,1])
        obs_R = obs_R.reshape((-1,)+ self.R_shape)
        est_t = est_t.reshape((-1,)+ self.t_shape)
        R_num = len(obs_R)
        R_shifts = np.empty((R_num,)+self.R_shape,dtype=np.float32)
        for i in range(R_num):
            R_shifts[i] = self.get_rotation_from_vectors(est_t[i], z_axis)
        new_Rs = np.matmul(R_shifts, obs_R)
        if R_num == 1:
            return new_Rs[0]
        return new_Rs

    @lazy_property
    def model_diameters(self):
        diameters = []
        model_info_path = self._kw['model_info_path'] 
        model_info = inout.load_info(model_info_path)
        for obj_id in model_info.keys():
            diameters.append(model_info[obj_id]['diameter'])
        return diameters

    @lazy_property
    def t_shape(self):
        return (3,1)
    
    @lazy_property
    def R_shape(self):
        return (3,3)
    
    # TODO: need to be modified so it can be the same as cfg file
    @lazy_property
    def code_shape(self):
        return (128,)

    @lazy_property
    def target_obj_id(self):
        #target_obj_id is aims to be used for renderer, the index of models in renderer starts from 0. 
        return int(self._kw['model_index']) - 1

    @lazy_property
    def bg_types(self):
        if self.is_training:
            bg_types = list(eval(self._kw['background_type']))
        else:
            bg_types = 'synthetic'
        return bg_types

    @lazy_property
    def distractor_obj_ids(self):
        obj_ids= list(range(len(self.model_diameters)))
        obj_ids.remove(self.target_obj_id)
        return obj_ids

    @lazy_property
    def renderer(self):
        from meshrenderer import meshrenderer_phong
        if self.vsd_eval:
            model_paths = glob.glob(self._kw['evaluation_model_path'])
        else:
            model_paths = glob.glob(self._kw['model_path'])

        if model_paths == []:
            print('No model file found in model path! Please check with your model path.')
            exit()
        model_paths.sort()
        renderer = meshrenderer_phong.Renderer(
            model_paths,
            int(self._kw['antialiasing']),
            self.dataset_path,
            float(self._kw['vertex_scale'])
        )
        return renderer
    
    def cad_model_paths(self):
        self.vsd_eval = True
        model_paths = glob.glob(self._kw['evaluation_model_path'])
        if model_paths == []:
            print('No model file found in model path! Please check with your model path.')
            exit()
        model_paths.sort()
        return model_paths
    
    @lazy_property
    def delta_t_class_num(self):
        return int(self._kw['t_class_num']) 

    @lazy_property
    def delta_t_resolution(self):
        return self.max_delta_t_shift*2/(self.delta_t_class_num-1)
    

    def classify_ts(self, ts):
        ts = self.max_delta_t_shift - ts
        classified_t = np.round(ts/self.delta_t_resolution).astype(int)
        # classified_t = np.zeros((len(ts),)+ self.classified_t_shape)
        # for i in range(len(ts)):
        #     t = np.round(ts[i]/self.delta_t_resolution).astype(int)
        #     for j in range(3):
        #         classified_t[i, j, t[j]] = 1
        return classified_t
        
    def get_eval_data_batch(self, batch_size):
        eval_indices = np.random.randint(0, len(self.eval_x), batch_size)
        # eval_indices = np.random.choice(len(self.eval_x), batch_size, replace = False)
        return self.eval_x[eval_indices]/255

    @lazy_property
    def min_visable_amount(self):
        return np.float(self._kw['min_visable_level'])

    @lazy_property
    def viewsphere_for_embedding(self):
        kw = self._kw
        num_cyclo = int(kw['num_cyclo'])
        azimuth_range = (0, 2 * np.pi)
        elev_range = (-0.5 * np.pi, 0.5 * np.pi)
        views, _ = view_sampler.sample_views(
            int(kw['min_n_views']),
            float(kw['radius']),
            azimuth_range,
            elev_range
        )
        Rs = np.empty( (len(views)*num_cyclo, 3, 3))
        i = 0
        for view in views:
            for cyclo in np.linspace(0, 2.*np.pi, num_cyclo):
                rot_z = np.array([[np.cos(-cyclo), -np.sin(-cyclo), 0], [np.sin(-cyclo), np.cos(-cyclo), 0], [0, 0, 1]])
                Rs[i,:,:] = rot_z.dot(view['R'])
                i += 1
        return Rs
    
    #TODO: to be tested
    @lazy_property
    def translation_for_embedding(self):
        import itertools
        t_step_num = int(self._kw['delta_t_step_num'])
        t_step = self.max_delta_t_shift*2/t_step_num
        total_delta_t_num = pow(t_step_num+1, 3)
        delta_ts = np.empty((total_delta_t_num,3),dtype=float)
        i=0
        for delta_t in itertools.product(np.arange(0, self.max_delta_t_shift+t_step, t_step), repeat=3):
            delta_ts[i] = delta_t
            i += 1
        return delta_ts


    def load_training_images(self, dataset_path, args, rotation_training = False, t_only = False):
        current_config_hash = hashlib.md5((str(args.items('Dataset')+args.items('Paths'))).encode('utf-8')).hexdigest()
        if self.use_real_imgs_only:
            data_src_comment = 'real_images'
        else:
            data_src_comment = 'mixed_images'
        current_file_name = os.path.join(dataset_path, 'obj_'+ str(self.target_obj_id+1).zfill(2)+'_'+ data_src_comment + '_' + current_config_hash + '.npz')

        if not os.path.exists(current_file_name):
            self.generate_rotation_dataset_file(current_file_name)
        self.load_dataset_file(current_file_name, rotation_training=rotation_training, t_only = t_only)
        print('loaded %s rotation training images' % (len(self.train_x)))

    @lazy_property
    def use_real_imgs_only(self):
        try:
            use_real_images_only = eval(self._kw['use_real_images_only'])
        except:
            print('Error: Keyword "USE_REAL_IMAGES_ONLY" can not be found in cfg file. Please add this keyword in "Dataset" section. The value of "USE_REAL_IMAGES_ONLY" could be True or False')
            exit(-1)
        return use_real_images_only 

    def generate_rotation_dataset_file(self, current_file_name):
        train_x, train_y, _, _, mask, gt_R, gt_t, delta_R, delta_t, noisy_R, noisy_t, K, gt_mask_pixel_num = self.training_dataset_variables_init(total_pixel_num = True)

        bar = utils.progressbar_init('Generate rotation training dataset: ', self.noof_training_imgs)
        bar.start()

        bg_dir_path = self._kw['background_base_dir']
        self.bg_rgb_list, self.bg_depth_list,_, self.bg_info_list, self.bg_gt_list = self.getRealImagesList(bg_dir_path, img_type='.png')

        obj_dir_path = self._kw['foreground_base_dir']
        self.obj_rgb_list, self.obj_depth_list, self.obj_mask_list, self.obj_info_list, self.obj_gt_list = self.getRealImagesList(obj_dir_path, img_type='.png')

        for i in np.arange(self.noof_training_imgs):
            bar.update(i)

            train_x[i], train_y[i], mask[i], gt_R[i], gt_t[i], delta_R[i], delta_t[i], noisy_R[i], noisy_t[i], K[i], gt_mask_pixel_num[i] = self.generate_train_image_patch_of_target_object()
        bar.finish()

        print('Saving training data...')
        np.savez(
            current_file_name, 
            train_x = train_x, 
            mask_x = mask, 
            train_y = train_y, 
            gt_R= gt_R, 
            gt_t = gt_t, 
            delta_R= delta_R, 
            delta_t = delta_t, 
            noisy_R = noisy_R, 
            noisy_t = noisy_t,
            K = K,
            gt_mask_pixel_num = gt_mask_pixel_num
            ) 
    
    def training_dataset_variables_init(self, total_pixel_num = False):
        train_x = np.empty( (self.noof_training_imgs,) + self.shape, dtype=np.uint8) 
        train_y = np.empty( (self.noof_training_imgs,) + self.shape, dtype=np.uint8) 
        syn_train_y = np.empty( (self.noof_training_imgs,) + self.shape, dtype=np.uint8) 
        rot_train_y = np.empty( (self.noof_training_imgs,) + self.shape, dtype=np.uint8) 
        mask = np.empty( (self.noof_training_imgs,) + self.shape[:2], dtype= bool)
        gt_R = np.empty((self.noof_training_imgs,)+self.R_shape, dtype=np.float)
        gt_t = np.empty((self.noof_training_imgs,)+self.t_shape, dtype=np.float)
        delta_R = np.empty((self.noof_training_imgs,)+self.R_shape, dtype=np.float)
        delta_t = np.empty((self.noof_training_imgs,)+self.t_shape, dtype=np.float)
        noisy_R = np.empty((self.noof_training_imgs,)+self.R_shape, dtype=np.float)
        noisy_t = np.empty((self.noof_training_imgs,)+self.t_shape, dtype=np.float)
        K = np.empty((self.noof_training_imgs,)+self.K_shape, dtype=np.float)
        if total_pixel_num:
            pixel_num = np.empty((self.noof_training_imgs,1), dtype=np.int)
            return train_x, train_y, syn_train_y, rot_train_y, mask, gt_R, gt_t, delta_R, delta_t, noisy_R, noisy_t, K, pixel_num
        return train_x, train_y, syn_train_y, rot_train_y, mask, gt_R, gt_t, delta_R, delta_t, noisy_R, noisy_t, K

    @lazy_property
    def K_shape(self):
        return (3,3)

    @lazy_property
    def no_delta_R_limit(self):
        return eval(self._kw['no_delta_r_limit']) 

    def generate_rotation_train_image_patch(self):
        success_flag = False
        while not success_flag:
            train_rgb, train_depth, visable_mask, gt_rgb, gt_depth, gt_mask, syn_gt_rgb, syn_gt_depth, gt_R, gt_t, K = self.generate_train_image_of_target_object()
            # cv2.imshow('generate_train_image_patch_rgb_gt', gt_rgb)
            # cv2.imshow('generate_train_image_patch_depth_gt', gt_depth)
            _, gt_mask_patch = self.depth_preprocess(gt_depth, gt_R, gt_t, K)
            # cv2.imshow('gt_mask', gt_mask_patch)
            if sum(sum(gt_mask_patch)) == 0:
                continue
            try_times = 0
            while True:
                delta_R, delta_t = self.get_pose_noise()
                if self.no_delta_R_limit:
                    delta_R = transform.random_rotation_matrix()[:3, :3]
                noisy_R = np.matmul(delta_R.transpose(), gt_R)
                noisy_t = gt_t + delta_t

                train_image_patch, depth_visable_mask = self.train_images_preprocess(train_rgb, train_depth, noisy_R, noisy_t, K)

                obj_visable_mask = depth_visable_mask*gt_mask_patch
                if self.getVisableAmount(obj_visable_mask, gt_mask_patch) >= self.min_visable_amount:
                    success_flag = True
                    break
                if try_times > 5:
                    break
                try_times += 1
        syn_image_patch, _ = self.train_images_preprocess(syn_gt_rgb, syn_gt_depth, noisy_R, noisy_t, K)
        return train_image_patch, syn_image_patch, obj_visable_mask, gt_R, gt_t, delta_R, delta_t, noisy_R, noisy_t, K
    
    def generate_train_image_patch_of_target_object(self):
        success_flag = False
        while not success_flag:
            if self.real_img_rot_aug:
                train_rgb, train_depth, visable_mask, syn_gt_rgb, syn_gt_depth, gt_mask, K, gt_R, gt_t = self.get_complete_train_images_aug_rotation()
            else:
                train_rgb, train_depth, visable_mask, syn_gt_rgb, syn_gt_depth, gt_mask, K, gt_R, gt_t = self.get_complete_train_images()
            # train_rgb, train_depth, visable_mask, syn_gt_rgb, syn_gt_depth, gt_mask, K, gt_R, gt_t = self.get_complete_train_images()
            try_times = 0
            while True:
                delta_R, delta_t = self.get_pose_noise()
                noisy_R = np.matmul(delta_R.transpose(), gt_R)
                noisy_t = gt_t + delta_t
                H = train_rgb.shape[0]
                W = train_rgb.shape[1]
                noisy_t = self.keep_t_within_image(noisy_t, K, W, H)

                if try_times > 5:
                    noisy_t = gt_t
                # train_image_patch, depth_visable_mask = self.train_images_preprocess(train_rgb, train_depth, noisy_R, noisy_t, K)
                _, gt_mask_patch = self.depth_preprocess(syn_gt_depth, noisy_R, noisy_t, K)
                if sum(sum(gt_mask_patch)) == 0:
                    continue
                # Note: the depth points was selected by a sphere, so the depth_visable_mask is differetnt with RGB_visable_mask
                _, depth_visable_mask = self.depth_preprocess(train_depth*visable_mask, noisy_R, noisy_t, K)
                obj_visable_mask = depth_visable_mask*gt_mask_patch
                if self.getVisableAmount(obj_visable_mask, gt_mask_patch) >= self.min_visable_amount:
                    success_flag = True
                    break
                if try_times > 5:
                    break
                try_times += 1
        train_image_patch, depth_visable_mask = self.train_images_preprocess(train_rgb, train_depth, noisy_R, noisy_t, K)
        syn_image_patch, _ = self.train_images_preprocess(syn_gt_rgb, syn_gt_depth, noisy_R, noisy_t, K)
        gt_mask_pixel_num = np.sum(gt_mask_patch*1)
        return train_image_patch, syn_image_patch, obj_visable_mask, gt_R, gt_t, delta_R, delta_t, noisy_R, noisy_t, K, gt_mask_pixel_num

    @tf.function
    def train_images_preprocess_tf(self, rgb, depth, R, t, K, rotation_target=False, bg_img = False):
        if rotation_target:
            bbox_enlarge_level = self.rotation_image_bbox_enlarge_level
        elif bg_img:
            bbox_enlarge_level = self.bbox_enlarge_level*random.uniform(1,3)
        else:
            bbox_enlarge_level = self.bbox_enlarge_level
        image_patch = im_process.image_preprocess_tf(rgb, depth, t, R, K, self.target_obj_diameter, bbox_enlarge_level, self.shape[:2])
        return image_patch

    def train_images_preprocess(self, rgb, depth, R, t, K, rotation_target=False, bg_img = False, broadcast = False, crop_hs= None, crop_ws = None, return_no_rot_patch= False):
        if bg_img:
            bg_bbox_enlarge_level = self.bbox_enlarge_level*random.uniform(1,3)
        else:
            bg_bbox_enlarge_level = None
        rgb_patch = self.rgb_preprocess(rgb, t, K, rotation_target=rotation_target, bg_bbox_enlarge_level = bg_bbox_enlarge_level, broadcast = broadcast, crop_hs=crop_hs, crop_ws=crop_ws)
        if return_no_rot_patch:
            depth_patch, depth_visable_mask, no_rot_patch = self.depth_preprocess(depth, R, t, K, rotation_target=rotation_target, bg_bbox_enlarge_level = bg_bbox_enlarge_level, broadcast=broadcast, crop_hs=crop_hs, crop_ws=crop_ws, return_no_rot_patch = return_no_rot_patch)
            image_patch_stack = np.concatenate((rgb_patch, depth_patch), axis=3)
            no_rot_patch_stack = np.concatenate((rgb_patch, no_rot_patch), axis=3)
            return image_patch_stack, depth_visable_mask, no_rot_patch_stack
        else:
            depth_patch, depth_visable_mask = self.depth_preprocess(depth, R, t, K, rotation_target=rotation_target, bg_bbox_enlarge_level = bg_bbox_enlarge_level, broadcast=broadcast, crop_hs=crop_hs, crop_ws=crop_ws, return_no_rot_patch = return_no_rot_patch)
        if broadcast:
            image_patch_stack = np.concatenate((rgb_patch, depth_patch), axis=3)
        else:
            image_patch_stack = np.concatenate((rgb_patch, depth_patch), axis=2)
        return image_patch_stack, depth_visable_mask

    def depth_preprocess(self, depth_image, R, t, K, rotation_target=False, bg_bbox_enlarge_level = None, broadcast = False, crop_hs=None, crop_ws=None, return_no_rot_patch = False):
        if rotation_target:
            bbox_enlarge_level = self.rotation_image_bbox_enlarge_level
        elif bg_bbox_enlarge_level:
            bbox_enlarge_level = bg_bbox_enlarge_level 
        else:
            bbox_enlarge_level = self.bbox_enlarge_level
        if broadcast:
            if return_no_rot_patch:
                depth_patch, visable_mask, no_rot_depth = im_process.depth_image_preprocess_broadcast(depth_image, R, t, K, (self.patch_W, self.patch_H), self.target_obj_diameter, bbox_enlarge_level, crop_hs=crop_hs, crop_ws=crop_ws, return_no_rot_patch = return_no_rot_patch)
                return depth_patch, visable_mask, no_rot_depth
            else:
                depth_patch, visable_mask = im_process.depth_image_preprocess_broadcast(depth_image, R, t, K, (self.patch_W, self.patch_H), self.target_obj_diameter, bbox_enlarge_level, crop_hs=crop_hs, crop_ws=crop_ws)
        else:
            depth_patch, visable_mask = im_process.depth_image_preprocess(depth_image, R, t, K.flatten(), (self.patch_W, self.patch_H), self.target_obj_diameter, bbox_enlarge_level)
        return depth_patch, visable_mask
    
    def rgb_preprocess(self, image, t, K, rotation_target=False,  bg_bbox_enlarge_level = None, broadcast = False, crop_hs=None, crop_ws=None):
        if rotation_target:
            bbox_enlarge_level = self.rotation_image_bbox_enlarge_level
        elif bg_bbox_enlarge_level:
            # bbox_enlarge_level = self.bbox_enlarge_level*random.uniform(1,2)
            bbox_enlarge_level = bg_bbox_enlarge_level 
        else:
            bbox_enlarge_level = self.bbox_enlarge_level
        if broadcast:
            rgb_patch = im_process.rgb_image_preprocess_broadcast(image, t, K, (self.patch_W, self.patch_H ), self.target_obj_diameter, bbox_enlarge_level, crop_hs=crop_hs, crop_ws=crop_ws)
        else:
            rgb_patch = im_process.rgb_image_preprocess(image, t, K.flatten(), (self.patch_W, self.patch_H ), self.target_obj_diameter, bbox_enlarge_level)
        return rgb_patch
    def load_dataset_file(self, current_file_name, rotation_training = False, t_only=False):
        training_data = np.load(current_file_name)
        self.train_x = training_data['train_x'].astype(np.uint8)
        self.train_mask = training_data['mask_x'].astype(np.uint8)
        self.train_gt_R = training_data['gt_R'].astype(np.float32)
        self.train_gt_t = training_data['gt_t'].astype(np.float32)
        self.train_delta_R = training_data['delta_R'].astype(np.float32)
        self.train_delta_t = training_data['delta_t'].astype(np.float32)
        self.train_noisy_R = training_data['noisy_R'].astype(np.float32)
        self.train_noisy_t = training_data['noisy_t'].astype(np.float32)
        self.gt_mask_pixel_num = training_data['gt_mask_pixel_num'].astype(np.float32)
        if rotation_training:
            self.train_y = self.load_rotation_only_target_set(current_file_name, self.train_gt_R, self.train_gt_t, self.train_noisy_R)
        elif t_only:
            self.train_y = self.load_t_only_target_set(current_file_name, self.train_delta_t)
        else:
            self.train_y = training_data['train_y'].astype(np.uint8)

    def load_t_only_target_set(self, current_file_name, delta_t):
        t_only_target_file_name = os.path.join(os.path.dirname(current_file_name),'t_only' + str(self.bbox_enlarge_level) + '_' + os.path.basename(current_file_name))
        if os.path.exists(t_only_target_file_name):
            train_y = np.load(t_only_target_file_name)['train_y']
        else:
            W, H = self.render_dim
            R = np.eye(3)
            rgb, depth, mask = self.render_gt_image(W, H, self.render_K, R, self.render_t)
            train_y = np.empty((len(delta_t),) + self.shape, dtype=np.uint8) 
            img_num = len(delta_t)

            bar = utils.progressbar_init('Generate translation only dataset: ', img_num)
            bar.start()
            for i in range(len(delta_t)):
                noisy_t = self.render_t.reshape((3,1)) + delta_t[i]
                train_y[i], _ = self.train_images_preprocess(rgb, depth, R, noisy_t, self.render_K)
                bar.update(i)
            bar.finish()
            print('Saving translation only target images...')
            np.savez(t_only_target_file_name, train_y=train_y) 
            print('Successfully saved translation only target images.')
        return train_y 

    def load_all_training_data(self, dataset_path, args):
        current_config_hash = hashlib.md5((str(args.items('Dataset')+args.items('Paths'))).encode('utf-8')).hexdigest()
        current_file_name = os.path.join(dataset_path, current_config_hash + '.npz')

        training_data = np.load(current_file_name)
        train_x = training_data['train_x'].astype(np.uint8)
        train_y = training_data['train_y'].astype(np.uint8)
        mask_x = training_data['mask_x']
        gt_R = training_data['gt_R'].astype(np.float)
        gt_t = training_data['gt_t'].astype(np.float)
        delta_R = training_data['delta_R'].astype(np.float)
        delta_t = training_data['delta_t'].astype(np.float)
        noisy_R = training_data['noisy_R'].astype(np.float)
        noisy_t = training_data['noisy_t'].astype(np.float)

        print('loaded %s training images' % (len(train_x)))
        return train_x, train_y, mask_x, gt_R, gt_t, delta_R, delta_t, noisy_R, noisy_t

    @lazy_property
    def render_dim(self):
        return eval(self._kw['render_dims'])

    @lazy_property
    def render_K(self):
        return np.array(eval(self._kw['k']))
    
    @lazy_property
    def render_t(self):
        return np.array([0,0, float(self._kw['radius'])])

    def generate_rotation_image_patch(self, gt_R, noisy_R, target=True):
        # W, H = self.render_dim
        # rgb, depth, _ = self.render_gt_image(W, H, self.render_K, gt_R, self.render_t)
        rgb, depth = self.get_rotation_only_target_crop(gt_R)
        image_patch, _ = self.train_images_preprocess(
            np.expand_dims(rgb, axis = 0),
            np.expand_dims(depth, axis = 0),
            np.expand_dims(noisy_R, axis = 0),
            np.expand_dims(self.rot_ts[0], axis = 0),   
            np.expand_dims(self.rotation_target_render_Ks[0], axis = 0),   
            rotation_target=target, broadcast=True)
        # image_patch, _ = self.train_images_preprocess(rgb, depth, noisy_R, self.render_t, self.render_K)
        return image_patch.squeeze()


    def load_rotation_only_target_set(self, current_file_name, gt_Rs, gt_ts, noisy_Rs):
        rotation_only_target_file_name = os.path.join(os.path.dirname(current_file_name),'rotation_only' + str(self.rotation_image_bbox_enlarge_level) + '_' + os.path.basename(current_file_name))
        if os.path.exists(rotation_only_target_file_name):
            train_y = np.load(rotation_only_target_file_name)['train_y']
        else:
            W, H = self.render_dim
            K = np.array(eval(self._kw['k']))
            render_t = np.array([0,0, float(self._kw['radius'])])
            img_num = len(gt_Rs)
            train_y = np.empty( (self.noof_training_imgs,) + self.shape, dtype=np.uint8 ) 
            bar = utils.progressbar_init('Generate rotation only dataset: ', img_num)

            bar.start()
            for img_id in range(img_num):
                bar.update(img_id)
                rgb, depth, mask = self.render_gt_image(W, H, K, gt_Rs[img_id], render_t)
                train_y[img_id], _ = self.train_images_preprocess(rgb, depth, noisy_Rs[img_id], render_t, K, rotation_target=True)
            bar.finish()
            print('Saving rotation only target images...')
            np.savez(rotation_only_target_file_name, train_y=train_y) 
            print('Successfully saved rotation only target images.')
        return train_y

    def get_rotation_noise(self):
        origin, xaxis, yaxis, zaxis = [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
        # noise_level = float(self._kw['rotation_noise_level'])
        # alpha, beta, gamma = np.random.uniform(-noise_level*math.pi, noise_level*math.pi, (3,1)) 
        # # print(alpha, beta, gamma)
        # Rx = transform.rotation_matrix(alpha, xaxis)
        # Ry = transform.rotation_matrix(beta, yaxis)
        # Rz = transform.rotation_matrix(gamma, zaxis)
        # R = transform.concatenate_matrices(Rx, Ry, Rz)
        # return R[:3, :3]
        return self._get_rotation_noise

    @lazy_property
    def _rot_noises_dataset_path(self):
        return os.path.join(self.dataset_path,'ae_rot_noises_'+str(self._rot_noise_level)+'_'+str(self._rot_noises_num)+'.npz')

    @lazy_property
    def _rot_noises_num(self):
        return int(self._kw['rotation_noise_num'])

    @lazy_property
    def _rot_noises(self):
        if not os.path.exists(self._rot_noises_dataset_path):
            return self._generate_rot_noises_dataset()
        else:
            data = np.load(self._rot_noises_dataset_path)
            return data['Rs']
    
    @lazy_property
    def _rot_noises_indices(self):
        return list(range(self._rot_noises_num))

    def _generate_rot_noises_dataset(self):
        rot_noises = np.empty((self._rot_noises_num, 3, 3))
        bar = utils.progressbar_init('Generating rotation noise dataset: ', self._rot_noises_num)
        bar.start()
        for i in range(self._rot_noises_num):
            rot_noises[i] = self._get_rotation_noise()
            bar.update(i)
        np.savez(self._rot_noises_dataset_path, Rs=rot_noises)
        bar.finish()
        return rot_noises

    def _get_rotation_noise(self):
        noise_level = self._rot_noise_level 
        angle = np.random.uniform(-noise_level*np.pi, noise_level*np.pi)
        axis = np.random.randn(3)
        return transform.rotation_matrix(angle, axis)[:3,:3]

    @lazy_property
    def _rot_noise_level(self):
        return float(self._kw['rotation_noise_level'])

    def render_rot(self, R, t=None ,downSample = 1):
        kw = self._kw
        h, w = self.shape[:2]
        radius = float(kw['radius'])
        W, H = self.render_dim
        K = eval(kw['k'])
        K = np.array(K).reshape(3,3)
        K[:2,:] = K[:2,:] / downSample

        pad_factor = float(kw['pad_factor'])

        t = np.array([0, 0, float(kw['radius'])])

        bgr_y, depth_y = self.renderer.render(
            obj_id=self.target_obj_id,
            W=W/downSample,
            H=H/downSample,
            K=K.copy(),
            R=R,
            t=t,
            near=self.clip_near,
            far=self.clip_far,
            random_light=False
        )
        rgb_patch = self.rgb_preprocess(bgr_y,t, K)
        depth_patch,_ = self.depth_preprocess(depth_y,R,t,K)
        output = np.concatenate((rgb_patch, depth_patch), axis=2).astype(np.uint8)
        return output 

    @lazy_property
    def clip_near(self):
        return float(self._kw['clip_near'])

    @lazy_property
    def clip_far(self):
        return float(self._kw['clip_far'])

    def render_distractors(self, W, H, K, t, distractor_num = 2):
        K = np.array(K).reshape((3,3)) 

        # if self.target_obj_diameter > 80:
        #     distractor_num += 1
        success_flag = False
        while not success_flag:
            render_obj_ids, Rs, ts = self.random_generate_distractors_and_poses(distractor_num, t)
            try:
                rgb, depth, _ = self.renderer.render_many(
                    obj_ids = render_obj_ids,
                    W = W,
                    H = H,
                    Rs = Rs,
                    ts = ts,
                    K = K,
                    near = self.clip_near,
                    far = self.clip_far,
                    random_light = True
                )
            except:
                continue
            success_flag = True
        mask = depth > 0
        return rgb, depth, mask

    def render_gt_image(self, W, H, K, R, t, random_light = False):
        K = np.array(K).reshape((3,3)) 
        R = np.array(R).reshape((3,3)) 
        t = np.array(t).flatten()
        rgb, depth = self.renderer.render(
            obj_id = self.target_obj_id,
            W = W,
            H = H,
            R = R,
            t = t,
            K = K,
            near = self.clip_near,
            far = self.clip_far,
            random_light = random_light 
        )
        mask = depth > 0
        return rgb, depth, mask

    def get_depth_scale(self, images_info, im_id):
        im_info = images_info[im_id]
        depth_scale = im_info['depth_scale']
        return depth_scale 

    def get_complete_train_images(self):
        bg_type = self._choose_bg_type()
        
        #TODO: following 4 lines are used for test, cfg file should be modified after testing, and following 4 lines should be deleted
        if bg_type == 'tless':
            bg_type = 'real'
        if bg_type == 'linemod':
            bg_type = 'synthetic'

        ori_rgb, ori_depth, syn_gt_rgb, syn_gt_depth, gt_mask, K, R, t = self.get_images_of_target_object(bg_type)

        bg_rgb, bg_depth, bg_mask = ori_rgb, ori_depth, gt_mask
        real_background = (random.random() <= 0.7)
        if real_background:
            temp_bg_rgb, temp_bg_depth, temp_bg_mask, success = self.add_real_background(ori_rgb, ori_depth,  K, t, gt_mask)
            if success:
                bg_rgb = temp_bg_rgb
                bg_depth = temp_bg_depth
                bg_mask = temp_bg_mask 

        add_distractors = (random.random() <= 0.9)
        if add_distractors:
            i = 0
            while True:
                distractor_rgb, distractor_depth, _ = self.render_distractors(W = ori_rgb.shape[1], H = ori_rgb.shape[0], K=K, t=t)
                train_rgb, train_depth, visable_mask = self.merge_images(bg_rgb, bg_depth, distractor_rgb, distractor_depth, bg_mask)
                if (self.getVisableAmount(visable_mask, gt_mask) < min(self.min_visable_amount, 1)) and i < 3:
                    continue
                elif i >= 3:
                    train_rgb = bg_rgb.copy()
                    train_depth = bg_depth.copy()
                    visable_mask = bg_mask.copy()
                    break
                else:
                    break
                i += 1
        else:
            train_rgb = bg_rgb.copy()
            train_depth = bg_depth.copy()
            visable_mask = bg_mask.copy()


        return train_rgb, train_depth, visable_mask, syn_gt_rgb, syn_gt_depth, gt_mask, K, R, t
    
    def get_complete_train_images_aug_rotation(self):
        ori_rgb, ori_depth, syn_gt_rgb, syn_gt_depth, gt_mask, K, R, t = self.get_images_of_target_object_rot_aug()

        bg_rgb, bg_depth, bg_mask = ori_rgb, ori_depth, gt_mask
        real_background = (random.random()<=0.7)
        if real_background:
            temp_bg_rgb, temp_bg_depth, temp_bg_mask, success = self.add_real_background(ori_rgb, ori_depth,  K, t, gt_mask)
            if success:
                bg_rgb = temp_bg_rgb
                bg_depth = temp_bg_depth
                bg_mask = temp_bg_mask 

        add_distractors = (random.random()<=0.9) 
        if add_distractors:
            i = 0
            while True:
                distractor_rgb, distractor_depth, _ = self.render_distractors(W = ori_rgb.shape[1], H = ori_rgb.shape[0], K=K, t=t)
                train_rgb, train_depth, visable_mask = self.merge_images(bg_rgb, bg_depth, distractor_rgb, distractor_depth, bg_mask)
                if (self.getVisableAmount(visable_mask, gt_mask) < min(self.min_visable_amount, 1)) and i < 3:
                    continue
                elif i >= 3:
                    train_rgb = bg_rgb.copy()
                    train_depth = bg_depth.copy()
                    visable_mask = bg_mask.copy()
                    break
                else:
                    break
                i += 1
        else:
            train_rgb = bg_rgb.copy()
            train_depth = bg_depth.copy()
            visable_mask = bg_mask.copy()


        return train_rgb, train_depth, visable_mask, syn_gt_rgb, syn_gt_depth, gt_mask, K, R, t
    
    @lazy_property
    def real_img_rot_aug(self):
        try:
            real_img_rot_aug = eval(self._kw['real_image_rotation_augmentation']) 
        except:
            print("can not find keyword 'real_image_rotation_augmentation' in cfg file. Generate images without rotation augmentation")
            real_img_rot_aug = False
        print('real_img_rot_aug', real_img_rot_aug)
        return real_img_rot_aug

    
    def augment_ori_real_image(self, ori_rgb, ori_depth, gt_mask, syn_gt_depth):
        ori_rgb, ori_depth = ori_rgb.copy(), ori_depth.copy()

        combine_depth = (random.random()<=0.3) 
        if combine_depth:
            ori_depth[ori_depth==0] = syn_gt_depth[ori_depth == 0]
        return ori_rgb, ori_depth

    def get_images_of_target_object(self, ori_image_type):
        if ori_image_type == 'real':
            ori_rgb, ori_depth, K, R, t = self.loadTargetImage()

            syn_gt_rgb, syn_gt_depth, gt_mask = self.render_gt_image(W = ori_rgb.shape[1], H = ori_rgb.shape[0], K=K, R=R,t=t)

            ori_rgb, ori_depth = self.augment_ori_real_image(ori_rgb, ori_depth, gt_mask, syn_gt_depth)

        elif ori_image_type == 'synthetic':
            ori_rgb, ori_depth, gt_mask, K, R, t = self.generate_synthetic_image()
            syn_gt_rgb, syn_gt_depth = ori_rgb.copy(), ori_depth.copy()
        else:
            print('Error: Unrecognized background type.')
            exit()
        return ori_rgb, ori_depth, syn_gt_rgb, syn_gt_depth, gt_mask, K, R, t
    
    def get_images_of_target_object_rot_aug(self):
        ori_rgb, ori_depth, K, R, t, all_real_img_loaded = self.load_rotation_augmented_real_target_image()

        if not all_real_img_loaded:
            syn_gt_rgb, syn_gt_depth, gt_mask = self.render_gt_image(W = ori_rgb.shape[1], H = ori_rgb.shape[0], K=K, R=R,t=t)

            ori_rgb, ori_depth = self.augment_ori_real_image(ori_rgb, ori_depth, gt_mask, syn_gt_depth)
        else:
            ori_rgb, ori_depth, gt_mask, K, R, t = self.generate_synthetic_image()
            syn_gt_rgb, syn_gt_depth = ori_rgb.copy(), ori_depth.copy()
        return ori_rgb, ori_depth, syn_gt_rgb, syn_gt_depth, gt_mask, K, R, t
    
    def loadTargetImage(self):
        img_id = random.randint(0,len(self.target_obj_rgb_list) - 1)
        rgb, depth, K, R, t = self.loadImageInfo(self.target_obj_id, img_id)
        return rgb, depth, K, R, t
    
    @lazy_property
    def aug_angle_resolution(self):
        return 5
    
    def load_rotation_augmented_real_target_image(self):
        all_images_loaded = False
        if self.real_image_id > (len(self.target_obj_rgb_list) - 1):
            all_images_loaded = True
            return None, None, None, None, None, all_images_loaded

        rgb, depth, K, R, t = self.loadImageInfo(self.target_obj_id, self.real_image_id)
        if (self.real_image_id <= 71) or (self.real_image_id >= 1224):
            self.real_image_id += 1 
        elif self.aug_angle == 0:
            self.aug_angle += self.aug_angle_resolution
        else:
            rgb, depth, R, t = im_process.rotate_image(rgb, depth, R, t, K, self.aug_angle)
            self.aug_angle += self.aug_angle_resolution
            if self.aug_angle >= 180:
                self.aug_angle = 0
                self.real_image_id += 1 
        return rgb, depth, K, R, t, all_images_loaded


    @lazy_property
    def target_obj_rgb_list(self):
        return self.obj_rgb_list[self.target_obj_id]
    
    @lazy_property
    def target_obj_info(self):
        return self.obj_info_list[self.target_obj_id]

    def generate_synthetic_image(self):
        img_id = random.randint(0,len(self.target_obj_rgb_list) - 1)
        R = transform.random_rotation_matrix()[:3,:3]
        K = self.target_obj_info[img_id]['cam_K']
        W, H = self.render_dim
        t_shift_max= min(self.target_obj_diameter, 30)
        t_shift = np.random.uniform(0, t_shift_max*2, (3,1)) - t_shift_max
        t = np.array(self.render_t).reshape((3,1)) + t_shift
        t = self.keep_t_within_image(t, K, W, H)
        rgb, depth, mask = self.render_gt_image(W = W, H = H, K=K, R=R,t=t, random_light = True)
        return rgb, depth, mask, K, R, t

    def generate_synthetic_image_crop(self, gt_t, gt_R, K,  random_light = False):
        cam_param = im_process.get_intrinsic_params(K)
        topleft_x, topleft_y, crop_w, crop_h = im_process.get_enlarged_bbox(
            cam_param, 
            gt_t, 
            self.target_obj_diameter, 
            enlarge_scale=self.bbox_enlarge_level)

        K_shape = K.shape
        K = K.flatten()
        CX_INDEX = 2
        CY_INDEX = 5
        K[CX_INDEX] -= topleft_x
        K[CY_INDEX] -= topleft_y
        K = K.reshape(K_shape)
        W, H = self.synthetic_image_render_dim
        rgb, depth, mask = self.render_gt_image(W = W, H = H, K=K, R=gt_R,t=gt_t, random_light = random_light)
        return rgb, depth, mask, K
    
    def generate_synthetic_image_crop_new(self, noisy_t, gt_t, gt_R, K,  random_light = False):
        cam_param = im_process.get_intrinsic_params(K)
        topleft_x, topleft_y, crop_w, crop_h = im_process.get_enlarged_bbox(
            cam_param, 
            gt_t, 
            self.target_obj_diameter, 
            enlarge_scale=self.bbox_enlarge_level)

        K_shape = K.shape
        K = K.flatten()
        CX_INDEX = 2
        CY_INDEX = 5
        K[CX_INDEX] -= topleft_x
        K[CY_INDEX] -= topleft_y
        K = K.reshape(K_shape)
        rgb, depth, mask = self.render_gt_image(W = 450, H = 450, K=K, R=gt_R,t=gt_t, random_light = random_light)
        return rgb, depth, mask, K
    
    def get_crops_of_target_object(self, ori_image_type):
        delta_R, delta_t = self.get_pose_noise()
        if ori_image_type == 'real':
            ori_rgb, ori_depth, K, gt_R, gt_t = self.loadTargetImage()
            rotate = (random.random()<=0.9)
            if rotate:
                aug_angle = np.random.randint(0,180)
                ori_rgb, ori_depth, gt_R, gt_t = im_process.rotate_image(ori_rgb, ori_depth, gt_R, gt_t, K, aug_angle)

            noisy_R = np.matmul(delta_R.transpose(), gt_R)
            noisy_t = gt_t + delta_t
            ori_rgb, ori_depth, K = im_process.crop_real_image(ori_rgb, ori_depth, K, self.target_obj_diameter, noisy_t, self.bbox_enlarge_level)

            syn_gt_rgb, syn_gt_depth, gt_mask, gt_K_out = self.generate_synthetic_image_crop(gt_t, gt_R, K, random_light = False)
            # assert (gt_K_out == K).all()
            ori_rgb = np.where(np.expand_dims(gt_mask, axis=2), ori_rgb, 0)
            obj_mask = np.abs(syn_gt_depth - ori_depth) <= 30
            ori_depth = np.where(obj_mask, ori_depth, 0) 

        elif ori_image_type == 'synthetic':
            gt_R = transform.random_rotation_matrix()[:3,:3]
            noisy_R = np.matmul(delta_R.transpose(), gt_R)
            gt_t = self.get_synthetic_noisy_t()
            noisy_t = gt_t + delta_t
            ori_rgb, ori_depth, _, K = self.generate_synthetic_image_crop( gt_t, gt_R, self.render_K, random_light = True)
            syn_gt_rgb, syn_gt_depth, gt_mask, _ = self.generate_synthetic_image_crop(gt_t, gt_R, self.render_K, random_light = False)
        else:
            print('Error: Unrecognized background type.')
            exit()
        return ori_rgb, ori_depth, syn_gt_rgb, syn_gt_depth, gt_mask, K, gt_R, gt_t, noisy_R, noisy_t, delta_R, delta_t
    
    def get_synthetic_noisy_t(self, max_bbox=False):
        t_shift_max= min(self.target_obj_diameter, 50)
        if not max_bbox:
            t_shift = np.random.uniform(-t_shift_max, t_shift_max, self.t_shape)
        else:
            t_shift = np.array([0, 0, -t_shift_max]).reshape(self.t_shape)
        t = np.array(self.render_t).reshape(self.t_shape) + t_shift
        return t
    
    @lazy_property
    def render_K_param(self):
        return im_process.get_intrinsic_params(self.render_K)
    
    @lazy_property
    def synthetic_image_render_dim(self):
        topleft_x, topleft_y, crop_w, crop_h = self.synthetic_image_crop_param
        w = max(crop_w, crop_h)
        h = w
        if 'real' in self.bg_types:
            self.check_real_img_status()
            real_img_h, real_img_w = self.real_rgbs[0].shape[:-1]
            w = max(real_img_w, w)
            h = max(real_img_h, h)
        print('synthetic image render dimension:', w, h)
        return (w, h)

    # @lazy_property
    # def synthetic_image_render_K(self):
    #     K = (self.render_K).flatten()
    #     h, w= self.synthetic_image_render_dim
    #     CX_INDEX = 2
    #     CY_INDEX = 5
    #     K[CX_INDEX] = w/2
    #     K[CY_INDEX] = h/2
    #     K = K.reshape(self.render_K.shape)
    #     print('synthetic image render K:', K)
    #     return K

    @lazy_property
    def synthetic_image_crop_param(self):
        t = self.get_synthetic_noisy_t(max_bbox=True)
        topleft_x, topleft_y, crop_w, crop_h = im_process.get_enlarged_bbox(
            self.render_K_param, 
            t, 
            self.target_obj_diameter, 
            enlarge_scale=self.bbox_enlarge_level)
        return (topleft_x, topleft_y, crop_w, crop_h)
    
    def keep_t_within_image(self, t, K, im_W, im_H):
        new_t = t.copy()
        K = im_process.get_intrinsic_params(K)
        t = t.flatten()
        
        x = int(t[0]*K['fx']/t[2] + K['cx'])
        if x < 0:
            new_t[0] = (-K['cx'])*t[2]/K['fx']
        if x > im_W:
            new_t[0] = (im_W-K['cx'])*t[2]/K['fx']

        y = int(t[1]*K['fy']/t[2] + K['cy'])
        if y < 0:
            new_t[1] = (-K['cy'])*t[2]/K['fy']
        if y > im_H:
            new_t[1] = (im_H - K['cy'])*t[2]/K['fy']
        return new_t 
        
    def check_t(self, t, K, im_W, im_H):
        new_t = t.copy()
        K = im_process.get_intrinsic_params(K)
        t = t.flatten()
        threshold = int(self.target_obj_diameter*K['fx']/t[2])
        x = int(t[0]*K['fx']/t[2] + K['cx'])
        if x < threshold:
            return False
        if x > im_W -threshold :
            return False
        y = int(t[1]*K['fy']/t[2] + K['cy'])
        if y < threshold:
            return False
        if y > im_H - threshold:
            return False
        return True
        
    def getVisableAmount(self, visable_mask, ori_mask):
        ori_pixels = sum(sum(ori_mask))
        if ori_pixels == 0:
            return 0 
        visable_amount = sum(sum(visable_mask))/ori_pixels
        return visable_amount

    @lazy_property
    def patch_H(self):
        return int(self._kw['h'])

    @lazy_property
    def patch_W(self):
        return int(self._kw['w'])

    @lazy_property
    def rotation_image_bbox_enlarge_level(self):
        print('rotation_image_bbox_enlarge_level:', np.float(self._kw['rotation_image_bbox_enlarge_level']) )
        return np.float(self._kw['rotation_image_bbox_enlarge_level']) 


    def get_pose_noise(self):
        # delta_R = self.get_rotation_noise()
        noise_ids = np.random.randint(0, self._rot_noises_num)
        delta_R = self._rot_noises[noise_ids]
        delta_t = np.random.uniform(-self.max_delta_t_shift, self.max_delta_t_shift, (3,1))
        return delta_R, delta_t
    
    def _get_delta_ts(self, len=1):
        if len > 1:
            shape = (len,) + self.t_shape
        else:
            shape = self.t_shape
        delta_t = np.random.uniform(-self.max_delta_t_shift, self.max_delta_t_shift, shape)
        return delta_t

    def merge_with_target_obj(self, rgb, depth, center_t, K):
        ori_rgb = rgb
        ori_depth = depth
        while True:
            R, t = self.generate_render_pose(self.target_obj_id, center_t)
            rendered_rgb, rendered_depth, rendered_mask = self.render_gt_image(W = ori_rgb.shape[1], H = ori_rgb.shape[0], K=K, R=R,t=t, random_light = True)
            rgb, depth, mask = self.merge_images(ori_rgb, ori_depth, rendered_rgb, rendered_depth)
            if self.getVisableAmount(mask, rendered_mask) >= min(self.min_visable_amount+0.1, 1):
                break 
        return rgb, depth, mask, R, t

    def merge_images(self, bg_rgb, bg_depth, render_rgb, render_depth, bg_mask = None, bg_type = None, bg_img_merge = False):
        bg_rgb = bg_rgb.copy()
        bg_depth = bg_depth.copy()

        fg_mask = (render_depth > 0)
        empty_area_mask = fg_mask*(bg_depth==0)
        occ_area_mask = fg_mask*(render_depth-bg_depth < 0)
        render_mask = empty_area_mask.copy()
        render_mask[occ_area_mask] = True
        bg_depth[render_mask] = render_depth[render_mask] 
        bg_rgb[render_mask] = render_rgb[render_mask]
        if bg_mask is not None:
            visable_mask = (render_mask == False)*bg_mask
        else:
            visable_mask = render_mask

        return bg_rgb, bg_depth, visable_mask

    def random_choose_bg_image(self, rgb_paths, depth_paths, img_info, img_gt):
        img_index = random.randint(0, len(rgb_paths)-1) 
        depth_scale = self.get_depth_scale(img_info, img_index) 
        rgb = cv2.imread(rgb_paths[img_index])
        depth = cv2.imread(depth_paths[img_index], cv2.IMREAD_ANYDEPTH)*depth_scale
        R, t = self.get_target_pose(img_gt, img_index)
        K = self.get_intrinsic_parameters(img_info, img_index)
        return rgb, depth, R, t, K

    def get_target_pose(self, images_gt, im_id):
        im_gt = images_gt[im_id]
        R = im_gt[0]['cam_R_m2c']
        t = im_gt[0]['cam_t_m2c']
        return R, t     

    def get_eval_target_pose(self, images_gt, im_id, obj_id, instance_id = 0):
        im_gt = images_gt[im_id]
        found_instance_id = 0
        R = None
        t = None
        for obj_gt in im_gt:
            if float(obj_gt['obj_id']) == obj_id:
                R = obj_gt['cam_R_m2c']
                t = obj_gt['cam_t_m2c']
                if instance_id == 0:
                    break
                elif instance_id == found_instance_id:
                    break
                found_instance_id += 1
        if R is None:
            print('can not find object in scene')
            exit()
        return R, t     

    def get_intrinsic_parameters(self, images_info, im_id):
        im_info = images_info[im_id]
        K = im_info['cam_K']
        return K

    def random_generate_distractors_and_poses(self, distractor_num, centre_t):
        Rs = []
        ts = []
        distractor_ids = []
        for idx in range(distractor_num):
            obj_ids = random.randint(0, len(self.distractor_obj_ids)-1)
            obj_id = self.distractor_obj_ids[obj_ids]
            R_render, t_render = self.generate_render_pose(obj_id, centre_t)
            Rs.append(R_render)
            ts.append(t_render)
            distractor_ids.append(obj_id)
        return distractor_ids, Rs, ts

    def generate_render_pose(self, distractor_id, target_obj_t):
        R = transform.random_rotation_matrix()[:3,:3]
        centre_dis = (self.target_obj_diameter + self.model_diameters[distractor_id])*0.3
        t_shift_max= centre_dis
        t_shift_min= 0 
        t_shift = np.random.uniform(t_shift_min, 2*t_shift_max, (3,1))-t_shift_max
        t = target_obj_t + t_shift 
        return R, t

    def render_embedding_image_batch(self, start, end):
        #TODO: to be tested
        # kw = self._kw
        # h, w = self.shape[:2]
        # azimuth_range = (0, 2 * np.pi)
        # elev_range = (-0.5 * np.pi, 0.5 * np.pi)
        # radius = float(kw['radius'])
        # render_dims = eval(kw['render_dims'])
        # K = eval(kw['k'])
        # K = np.array(K).reshape(3,3)

        # pad_factor = float(kw['pad_factor'])

        # t = np.array([0, 0, float(kw['radius'])])
        # batch = np.empty( (end-start,)+ self.shape)
        obj_bbs = np.empty( (end-start,)+ (4,))

        _, _, rot_w, rot_h = self.rotation_target_crop_param
        batch_size = end - start
        rot_rgbs = np.empty((batch_size,) + (rot_h, rot_w, 3,), dtype=np.uint8) 
        rot_depths = np.empty((batch_size,) + (rot_h, rot_w, ), dtype=np.float32) 
        
        for i, R in enumerate(self.viewsphere_for_embedding[start:end]):
            rot_rgbs[i], rot_depths[i] = self.get_rotation_only_target_crop(R)

        batch, _ = self.train_images_preprocess(
            rot_rgbs, rot_depths, self.viewsphere_for_embedding[start:end],
            self.rot_ts[:batch_size], 
            self.rotation_target_render_Ks[:batch_size], 
            rotation_target=True, broadcast=True)
        # batch, _ = self.train_images_preprocess(rgbs, depths, R, t, K)
        return (batch, obj_bbs)

    def load_syn_rotation_only_dataset(self, dataset_path, args):
        current_config_hash = hashlib.md5(('syn_rotation_only_'+str(self.target_obj_id)+'_'+str(self.rotation_image_bbox_enlarge_level)+'_'+str(self.bbox_enlarge_level)).encode('utf-8')).hexdigest()
        print('synthetic rotation only dataset:', current_config_hash)
        file_name = os.path.join(dataset_path,'syn_rotation_only_' +current_config_hash+'.npz' )
        if os.path.exists(file_name):
            self.syn_rot_train_x = np.load(file_name)['train_x']
            self.syn_rot_train_y = np.load(file_name)['train_y']
        else:
            kw = self._kw
            h, w = self.shape[:2]
            azimuth_range = (0, 2 * np.pi)
            elev_range = (-0.5 * np.pi, 0.5 * np.pi)
            radius = float(kw['radius'])
            render_dims = eval(kw['render_dims'])
            K = eval(kw['k'])
            K = np.array(K).reshape(3,3)

            pad_factor = float(kw['pad_factor'])
            t = np.array([0, 0, float(kw['radius'])])
            syn_rot_image_num = len(self.viewsphere_for_embedding)
            train_batch = np.empty((syn_rot_image_num,)+ self.shape, dtype=np.uint8)
            target_batch = np.empty((syn_rot_image_num,)+ self.shape, dtype=np.uint8)

            bar = utils.progressbar_init('Generate synthetic rotation dataset: ', syn_rot_image_num)
            bar.start()
            for i, R in enumerate(self.viewsphere_for_embedding):
                rgb, depth, mask = self.render_gt_image(render_dims[0], render_dims[1], K.copy(), R, t)
  
                target_batch[i], _ = self.train_images_preprocess(rgb, depth, R, t, K, rotation_target = True)
                train_batch[i], _ = self.train_images_preprocess(rgb, depth, R, t, K)
                bar.update(i)
            bar.finish()
            print('Saving dataset...')
            np.savez(file_name, train_x = train_batch, train_y= target_batch)
            print('Successfule saved dataset.')
            self.syn_rot_train_x = train_batch
            self.syn_rot_train_y = target_batch
        return self.syn_rot_train_x, self.syn_rot_train_y

    def extract_square_patch(self, scene_img, bb_xywh, pad_factor,resize=(128,128),interpolation=cv2.INTER_NEAREST,black_borders=False):

        x, y, w, h = np.array(bb_xywh).astype(np.int32)
        size = int(np.maximum(h, w) * pad_factor)

        left = int(np.maximum(x+w/2-size/2, 0))
        right = int(np.minimum(x+w/2+size/2, scene_img.shape[1]))
        top = int(np.maximum(y+h/2-size/2, 0))
        bottom = int(np.minimum(y+h/2+size/2, scene_img.shape[0]))

        scene_crop = scene_img[top:bottom, left:right].copy()

        if black_borders:
            scene_crop[:(y-top),:] = 0
            scene_crop[(y+h-top):,:] = 0
            scene_crop[:,:(x-left)] = 0
            scene_crop[:,(x+w-left):] = 0

        scene_crop = cv2.resize(scene_crop, resize, interpolation = interpolation)
        return scene_crop

    @property
    def embedding_size(self):
        return len(self.viewsphere_for_embedding)

    @property
    def t_embedding_size(self):
        return len(self.translation_for_embedding)

    @lazy_property
    def _aug(self):
        from imgaug.augmenters import Sequential,SomeOf,OneOf,Sometimes,WithColorspace,WithChannels, \
            Noop,Lambda,AssertLambda,AssertShape,Scale,CropAndPad, \
            Pad,Crop,Fliplr,Flipud,Superpixels,ChangeColorspace, PerspectiveTransform, \
            Grayscale,GaussianBlur,AverageBlur,MedianBlur,Convolve, \
            Sharpen,Emboss,EdgeDetect,DirectedEdgeDetect,Add,AddElementwise, \
            AdditiveGaussianNoise,Multiply,MultiplyElementwise,Dropout, \
            CoarseDropout,Invert,ContrastNormalization,Affine,PiecewiseAffine, \
            ElasticTransformation, SigmoidContrast, RelativeRegularGridVoronoi,\
            UniformVoronoi, JpegCompression, ImpulseNoise
        return eval(self._kw['code'])

    @lazy_property
    def _depth_aug(self):
        from imgaug.augmenters import Sequential,SomeOf,OneOf,Sometimes,WithColorspace,WithChannels, \
            Noop,Lambda,AssertLambda,AssertShape,Scale,CropAndPad, \
            Pad,Crop,Fliplr,Flipud,Superpixels,ChangeColorspace, PerspectiveTransform, \
            Grayscale,GaussianBlur,AverageBlur,MedianBlur,Convolve, \
            Sharpen,Emboss,EdgeDetect,DirectedEdgeDetect,Add,AddElementwise, \
            AdditiveGaussianNoise,Multiply,MultiplyElementwise,Dropout, \
            CoarseDropout,Invert,ContrastNormalization,Affine,PiecewiseAffine, \
            ElasticTransformation, ImpulseNoise
        return eval(self._kw['depth'])        

    @lazy_property
    def _aug_occl(self):
        from imgaug.augmenters import Sequential,SomeOf,OneOf,Sometimes,WithColorspace,WithChannels, \
            Noop,Lambda,AssertLambda,AssertShape,Scale,CropAndPad, \
            Pad,Crop,Fliplr,Flipud,Superpixels,ChangeColorspace, PerspectiveTransform, \
            Grayscale,GaussianBlur,AverageBlur,MedianBlur,Convolve, \
            Sharpen,Emboss,EdgeDetect,DirectedEdgeDetect,Add,AddElementwise, \
            AdditiveGaussianNoise,Multiply,MultiplyElementwise,Dropout, \
            CoarseDropout,Invert,ContrastNormalization,Affine,PiecewiseAffine, \
            ElasticTransformation
        return Sequential([Sometimes(0.7, CoarseDropout( p=0.4, size_percent=0.01) )])

    @lazy_property
    def _aug_crop(self):
        from imgaug.augmenters import Sequential,SomeOf,OneOf,Sometimes,WithColorspace,WithChannels, \
            Noop,Lambda,AssertLambda,AssertShape,Scale,CropAndPad, \
            Pad,Crop,Fliplr,Flipud,Superpixels,ChangeColorspace, PerspectiveTransform, \
            Grayscale,GaussianBlur,AverageBlur,MedianBlur,Convolve, \
            Sharpen,Emboss,EdgeDetect,DirectedEdgeDetect,Add,AddElementwise, \
            AdditiveGaussianNoise,Multiply,MultiplyElementwise,Dropout, \
            CoarseDropout,Invert,ContrastNormalization,Affine,PiecewiseAffine, \
            ElasticTransformation
        return eval(self._kw['crop'])


    @lazy_property
    def random_syn_masks(self):
        import bitarray
        workspace_path = os.environ.get('AE_WORKSPACE_PATH')

        random_syn_masks = bitarray.bitarray()
        with open(os.path.join(workspace_path,'random_tless_masks/arbitrary_syn_masks_1000.bin'), 'r') as fh:
            random_syn_masks.fromfile(fh)
        occlusion_masks = np.fromstring(random_syn_masks.unpack(), dtype=np.bool)
        occlusion_masks = occlusion_masks.reshape(-1,224,224,1).astype(np.float32)
        # print(occlusion_masks.shape)

        occlusion_masks = np.array([cv2.resize(mask,(self.shape[0],self.shape[1]), interpolation = cv2.INTER_NEAREST) for mask in occlusion_masks])
        return occlusion_masks

    def load_eval_images(self, dataset_path, args):
        current_config_hash = hashlib.md5((str(args.items('Evaluation'))+str(self.target_obj_id)+str(self.bbox_enlarge_level)).encode('utf-8')).hexdigest()
        current_file_name = os.path.join(dataset_path, current_config_hash + '.npz')
        print('loading ', current_file_name)

        if os.path.exists(current_file_name):
            eval_data = np.load(current_file_name)
            self.eval_x = eval_data['eval_x']
            self.eval_delta_t = eval_data['delta_t']
            self.eval_delta_R = eval_data['delta_R']
            self.eval_gt_t = eval_data['gt_t']
            self.eval_gt_R = eval_data['gt_R']
            self.eval_noisy_R = eval_data['noisy_R']
            self.eval_no_rot_x = eval_data['no_rot_eval_x']
        else:
            self.eval_x, self.eval_gt_R, self.eval_gt_t, self.eval_delta_R, self.eval_delta_t, self.eval_noisy_R, self.eval_noisy_t, self.eval_no_rot_x = self.get_eval_image_patchs()
            np.savez(current_file_name, eval_x = self.eval_x, gt_R= self.eval_gt_R, gt_t = self.eval_gt_t, delta_R= self.eval_delta_R, delta_t = self.eval_delta_t, noisy_R = self.eval_noisy_R, noisy_t = self.eval_noisy_t, no_rot_eval_x = self.eval_no_rot_x) 
        print('loaded %s evaluation images' % (len(self.eval_x)))
        return self.eval_x, self.eval_delta_t, self.eval_delta_R, self.eval_gt_t, self.eval_gt_R, self.eval_noisy_R, self.eval_no_rot_x

    def get_eval_image_patchs(self):
        img_info = inout.load_info(self._kw['evaluation_images_info_path'])
        img_gt = inout.load_gt(self._kw['evaluation_images_gt_path']) 
        rgb_imgs = self.load_images(self._kw['evaluation_rgb_images_glob']) 
        depth_imgs = self.load_images(self._kw['evaluation_depth_images_glob'], depth=True, depth_scale=0.1) 
        eval_image_num = int(self._kw['evaluation_images_num'])

        eval_x = np.empty( (eval_image_num,) + self.shape, dtype=np.uint8 ) 
        no_rot_x = np.empty( (eval_image_num,) + self.shape, dtype=np.uint8 ) 
        gt_R = np.empty((eval_image_num,)+(3,3), dtype=np.float)
        gt_t = np.empty((eval_image_num,)+(3,1), dtype=np.float)
        delta_R = np.empty((eval_image_num,)+(3,3), dtype=np.float)
        delta_t = np.empty((eval_image_num,)+(3,1), dtype=np.float)
        noisy_R = np.empty((eval_image_num,)+(3,3), dtype=np.float)
        noisy_t = np.empty((eval_image_num,)+(3,1), dtype=np.float)

        loaded_img_num = len(rgb_imgs)
        bar = utils.progressbar_init('Generate evaluation dataset: ', eval_image_num)
        bar.start()
        for i in range(eval_image_num):
            im_idx = np.random.randint(0, loaded_img_num)
            depth_img = depth_imgs[im_idx]
            rgb_img = rgb_imgs[im_idx]
            K = self.get_intrinsic_parameters(img_info, im_idx)
            gt_R[i], gt_t[i] = self.get_eval_target_pose(img_gt, im_idx, self.target_obj_id + 1)

            delta_R[i], delta_t[i] = self.get_pose_noise()
            noisy_R[i] = np.matmul(delta_R[i].transpose(), gt_R[i])
            noisy_t[i] = gt_t[i] + delta_t[i]

            # eval_x[i, :,:, 3:], _= self.depth_preprocess(depth_img, noisy_R[i], noisy_t[i], K)
            # eval_x[i,:,:,0:3] = self.rgb_preprocess(rgb_img, noisy_t[i], K)
            eval_x[i], _ , no_rot_x[i]= self.train_images_preprocess(
                np.expand_dims(rgb_img, axis = 0), 
                np.expand_dims(depth_img, axis=0), 
                np.expand_dims(noisy_R[i], axis = 0),
                np.expand_dims(noisy_t[i], axis = 0), 
                np.expand_dims(K, axis=0), 
                return_no_rot_patch = True,
                broadcast=True)
            bar.update(i)
        bar.finish()

        return eval_x, gt_R, gt_t, delta_R, delta_t, noisy_R, noisy_t, no_rot_x 
        
    def generate_ordered_eval_image_with_no_prediction(self, dataset_path, args):
        current_config_hash = hashlib.md5((str(args.items('Evaluation'))+str(self.target_obj_id)+str(self.bbox_enlarge_level)).encode('utf-8')).hexdigest()
        current_file_name = os.path.join(dataset_path, 'Ordered_'+str(self.target_obj_id+1)+'_'+current_config_hash + '.npz')
        print('loading ', current_file_name)
        if os.path.exists(current_file_name):
            eval_data = np.load(current_file_name)
            self.eval_x = eval_data['eval_x']
            self.eval_noisy_t = eval_data['noisy_t']
            self.eval_noisy_R = eval_data['noisy_R']
            self.eval_gt_t = eval_data['gt_t']
            self.eval_gt_R = eval_data['gt_R']
            self.eval_no_rot_x = eval_data['no_rot_eval_x']
        else:
            img_info = inout.load_info(self._kw['evaluation_images_info_path'])
            img_gt = inout.load_gt(self._kw['evaluation_images_gt_path']) 
            rgb_imgs = self.load_images(self._kw['evaluation_rgb_images_glob']) 
            depth_imgs = self.load_images(self._kw['evaluation_depth_images_glob'], depth=True, depth_scale=0.1) 

            loaded_img_num = len(rgb_imgs)
            eval_x = np.empty( (loaded_img_num,) + self.shape, dtype=np.uint8 ) 
            no_rot_x = np.empty( (loaded_img_num,) + self.shape, dtype=np.uint8 ) 
            gt_R = np.empty((loaded_img_num,)+(3,3), dtype=np.float)
            gt_t = np.empty((loaded_img_num,)+(3,1), dtype=np.float)
            noisy_R = np.empty((loaded_img_num,)+(3,3), dtype=np.float)
            noisy_t = np.empty((loaded_img_num,)+(3,1), dtype=np.float)

            bar = utils.progressbar_init('Generate evaluation dataset: ', loaded_img_num)
            bar.start()
            for i in range(loaded_img_num):
                im_idx=i
                depth_img = depth_imgs[im_idx]
                rgb_img = rgb_imgs[im_idx]
                K = self.get_intrinsic_parameters(img_info, im_idx)
                gt_R[i], gt_t[i] = self.get_eval_target_pose(img_gt, im_idx, self.target_obj_id + 1)

                if (i == 0):
                    noisy_R[i] = gt_R[i]
                    noisy_t[i] = gt_t[i]
                else:
                    noisy_R[i] = gt_R[i-1]
                    noisy_t[i] = gt_t[i-1]

                # eval_x[i, :,:, 3:], _= self.depth_preprocess(depth_img, noisy_R[i], noisy_t[i], K)
                # eval_x[i,:,:,0:3] = self.rgb_preprocess(rgb_img, noisy_t[i], K)

                eval_x[i], _, no_rot_x[i]= self.train_images_preprocess(
                    np.expand_dims(rgb_img, axis = 0), 
                    np.expand_dims(depth_img, axis=0), 
                    np.expand_dims(noisy_R[i], axis = 0),
                    np.expand_dims(noisy_t[i], axis = 0), 
                    np.expand_dims(K, axis=0), 
                    return_no_rot_patch = True,
                    broadcast=True)
                bar.update(i)
            bar.finish()

            print('Saving image...')
            np.savez(current_file_name, eval_x = eval_x, gt_R= gt_R, gt_t = gt_t,  noisy_R = noisy_R, noisy_t = noisy_t, no_rot_eval_x = no_rot_x) 
            self.eval_x = eval_x 
            self.eval_noisy_t = noisy_t 
            self.eval_noisy_R = noisy_R 
            self.eval_gt_t = gt_t 
            self.eval_gt_R = gt_R 
            self.eval_no_rot_x = no_rot_x
        print('Successfully load %s ordered evaluation images.' % len(self.eval_x))
        

    def load_images(self, img_glob, depth=False, depth_scale = 1):
        image = []
        img_paths = glob.glob(img_glob)
        if img_paths == []:
            if depth:
                print('Error: Incorrect evaluation depth image glob!')
            else:
                print('Error: Incorrect evaluation RGB image glob!')
            exit(-1)
        img_paths.sort()
        if depth:
            for img_path in img_paths:
                image.append(cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)*depth_scale)
        else:
            for img_path in img_paths:
                image.append(cv2.imread(img_path))
        return image
        
    @lazy_property
    def crop_mask_template(self):
        return np.ones((self.batch_size,)+self.mask_shape, dtype = np.uint8) 
    
    @lazy_property
    def visable_amount_shape(self):
        return (1,)

    def batch(self, batch_size, test = False):
        return self.batch_dynamic_complete(batch_size, test = test)

    @lazy_property
    def batch_R_placeholder(self):
        return np.empty((self.batch_size,)+(3,3), dtype=np.float)
        
    def batch_dynamic(self, batch_size, test = False, pose_pred= False):

        batch_x, batch_y, batch_mask, batch_delta_t, gt_mask_pixel_num = self.get_train_x_broadcast(batch_size, test = test)

        batch_x, batch_mask = self.augment_train_image(batch_x, batch_mask)

        visable_pixel_num = np.sum(batch_mask, axis=(1,2)).reshape(-1,1)
        visable_amount = visable_pixel_num/gt_mask_pixel_num

        if not self._class_t:
            batch_delta_t = -batch_delta_t/self.max_delta_t_shift

        return (batch_x, batch_y, batch_mask, batch_delta_t, visable_amount)
    
    def batch_dynamic_complete(self, batch_size, test = False):
        batch_x, batch_y, depth_no_rot, rot_y, batch_mask, gt_mask, batch_delta_t, batch_noisy_R= self.get_train_x_broadcast_complete(batch_size, test = test)

        batch_x = np.concatenate((batch_x, depth_no_rot), axis = -1)

        batch_x, batch_mask = self.augment_train_image(batch_x, batch_mask)

        predictor_x = np.concatenate((batch_x[:,:,:,0:3], batch_x[:,:,:,6:]), axis = -1)
        batch_x = batch_x[:,:,:,0:6]

        return (batch_x, batch_y, predictor_x, rot_y, batch_mask, gt_mask, batch_delta_t, batch_noisy_R)
    
    def batch_dynamic_complete_acc(self, batch_size, test = False):
    
        batch_x, batch_y, depth_no_rot, rot_rgbs, rot_depths, batch_mask, gt_mask, batch_delta_t, batch_noisy_R= self.get_train_x_broadcast_complete_acc(batch_size, test = test)

        # batch_x = np.concatenate((batch_x, depth_no_rot), axis = -1)

        batch_x, batch_mask = self.augment_train_image(batch_x, batch_mask)

        # predictor_x = np.concatenate((batch_x[:,:,:,0:3], batch_x[:,:,:,6:]), axis = -1)
        batch_x = batch_x[:,:,:,0:6]

        # return (batch_x, batch_y, predictor_x, rot_rgbs, rot_depths, batch_mask, gt_mask, batch_delta_t, batch_noisy_R)
        return (batch_x, batch_y, batch_x, rot_rgbs, rot_depths, batch_mask, gt_mask, batch_delta_t, batch_noisy_R)
        
    def augment_train_image(self, batch_x, batch_mask):
        cropped_mask = self._aug_crop.augment_images(self.crop_mask_template)
        batch_x = batch_x*np.expand_dims(cropped_mask, axis=3)
        batch_mask = batch_mask * cropped_mask
        batch_x[:,:,:,0:3] = self._aug.augment_images(batch_x[:,:,:,0:3])
        batch_x[:,:,:,3:6] = self._depth_aug.augment_images(batch_x[:,:,:,3:6])
        if batch_x.shape[3] > 6:
            batch_x[:,:,:,6:] = self._depth_aug.augment_images(batch_x[:,:,:,6:])
        return batch_x, batch_mask
    
    def batch_dynamic_acc(self, batch_size, test = False, pose_pred= False):
        train_x, obj_visable_mask, gt_rgb_patch, gt_depth_patch, rot_rgbs, rot_depths, noisy_Rs, noisy_ts, delta_ts= self.get_train_x_broadcast_acc(batch_size, test = test)
        train_x, obj_visable_mask = self.augment_train_image(train_x, obj_visable_mask)
        if not self._class_t:
            delta_ts = -delta_ts/self.max_delta_t_shift

        return (train_x, obj_visable_mask, gt_rgb_patch, gt_depth_patch, rot_rgbs, rot_depths, noisy_Rs, noisy_ts, delta_ts)
    
    def getRealImagesList(self, path, img_type):
        """
        Function: get the list of depth images, rgb images, gt.yml and info.yml
        directors should look like:
            path
                --01:
                    --depth
                    --rgb
                    --mask
                    gt.yml
                    info.yml
                --02:
                    --depth
                    --rgb
                    --mask
                    gt.yml
                    info.yml
                --03:
                    ...
        """
        depth_list = []
        rgb_list = []
        mask_list = []
        gt_list = []
        info_list = []
        img_dirs = os.listdir(path)
        img_dirs.sort()
        for dir_name in img_dirs:
            dir_path = os.path.join(path, dir_name) 
            depth_list.append(self.getImageFilePathList(os.path.join(dir_path, 'depth'), '.png') + self.getImageFilePathList(os.path.join(dir_path, 'depth'), '.jpg')) 
            rgb_list.append(self.getImageFilePathList(os.path.join(dir_path, 'rgb'), '.png') + self.getImageFilePathList(os.path.join(dir_path, 'rgb'), '.jpg'))
            info_list.append(inout.load_info(os.path.join(dir_path, 'info.yml')))
            gt_list.append(inout.load_gt(os.path.join(dir_path, 'gt.yml')))
        # print(info_list[0][0])
        # print(gt_list[0][0][0])
        return rgb_list, depth_list, mask_list, info_list, gt_list

    def getImageFilePathList(self, dir_path, img_type):
        glob_path = dir_path+'/*'+img_type
        img_file_paths = glob.glob(glob_path)
        img_file_paths.sort()
        return img_file_paths

    def load_real_training_images_of_target(self):
        print('Loading tless real training images...')
        obj_id = self.target_obj_id  
        total_num = len(self.obj_rgb_list[obj_id])
        img_id = 1
        rgb_shape = (cv2.imread(self.obj_rgb_list[obj_id][img_id])).shape
        rgbs = np.empty((total_num,)+rgb_shape, dtype=np.uint8)
        depths = np.empty((total_num,)+rgb_shape[:-1], dtype = np.float32)
        K_shape = (self.obj_info_list[obj_id][img_id]['cam_K']).shape
        Ks = np.empty((total_num,)+K_shape, dtype = np.float32)
        R_shape = (self.obj_gt_list[obj_id][img_id][0]['cam_R_m2c']).shape
        t_shape = (self.obj_gt_list[obj_id][img_id][0]['cam_t_m2c']).shape
        Rs = np.empty((total_num,)+R_shape, dtype = np.float32) 
        ts = np.empty((total_num,)+t_shape, dtype = np.float32) 
        print(K_shape)
        print(R_shape)
        print(t_shape)
        for img_id in range(total_num):
            rgbs[img_id] = cv2.imread(self.obj_rgb_list[obj_id][img_id])
            depth_scale = self.obj_info_list[obj_id][img_id]['depth_scale']
            depths[img_id] = cv2.imread(self.obj_depth_list[obj_id][img_id], cv2.IMREAD_ANYDEPTH)*depth_scale
            Ks[img_id] = self.obj_info_list[obj_id][img_id]['cam_K']
            Rs[img_id] = self.obj_gt_list[obj_id][img_id][0]['cam_R_m2c'] 
            ts[img_id] = self.obj_gt_list[obj_id][img_id][0]['cam_t_m2c'] 
        self.real_rgbs, self.real_depths = rgbs, depths
        self.real_Ks, self.real_Rs, self.real_ts = Ks, Rs, ts
        print('Successfully loaded', len(self.Ks), 'real training images.')

    def loadImageInfo(self, obj_id, img_id, bg = False):
        if not bg:
            rgb = cv2.imread(self.obj_rgb_list[obj_id][img_id])
            depth_scale = self.obj_info_list[obj_id][img_id]['depth_scale']
            depth = cv2.imread(self.obj_depth_list[obj_id][img_id], cv2.IMREAD_ANYDEPTH)*depth_scale
            K = self.obj_info_list[obj_id][img_id]['cam_K']
            R = self.obj_gt_list[obj_id][img_id][0]['cam_R_m2c'] 
            t = self.obj_gt_list[obj_id][img_id][0]['cam_t_m2c'] 
        else:
            rgb = cv2.imread(self.bg_rgb_list[obj_id][img_id])
            depth_scale = self.bg_info_list[obj_id][img_id]['depth_scale']
            depth = cv2.imread(self.bg_depth_list[obj_id][img_id], cv2.IMREAD_ANYDEPTH)*depth_scale
            K = self.bg_info_list[obj_id][img_id]['cam_K']
            R = self.bg_gt_list[obj_id][img_id][0]['cam_R_m2c'] 
            t = self.bg_gt_list[obj_id][img_id][0]['cam_t_m2c'] 
        return rgb, depth, K, R, t

    def add_real_background(self, obj_rgb, obj_depth, obj_K, obj_t, gt_mask):
        success = False
        i = 0
        while not success:
            scene_id = random.randint(0, len(self.bg_rgb_list)-1)
            img_id = random.randint(0, len(self.bg_rgb_list[scene_id])-1)
            # random.choice(range(len(self.bg_rgb_list[scene_id])))
            # img_id = random.choice(range(len(self.bg_rgb_list[scene_id])))
            bg_rgb, bg_depth, bg_K, _, bg_t = self.loadImageInfo(scene_id, img_id, bg=True)
            _, delta_t = self.get_pose_noise()
            if delta_t[2] < 0:
                delta_t[2] = -delta_t[2]
            noisy_t = obj_t + delta_t*1.2
            bg_rgb, bg_depth, _ = self.moveObjImage(bg_rgb, bg_depth, bg_t, noisy_t, bg_K, obj_K, (obj_rgb.shape[1], obj_rgb.shape[0]))
            result_rgb, result_depth, fg_visable_mask = self.merge_images(bg_rgb, bg_depth, obj_rgb, obj_depth)
            visable_mask = fg_visable_mask*gt_mask
            if self.getVisableAmount(visable_mask, gt_mask) >= min(self.min_visable_amount + 0.1, 1):
                success = True
                break
            if i >= 2:
                break
            i += 1
        return result_rgb, result_depth, visable_mask, success

    def moveObjImage(self, rgb, depth, ori_t, new_t, ori_K, new_K, new_img_shape, mask = None, mask_move = False):
        ori_t = np.array(ori_t).flatten()
        new_t = np.array(new_t).flatten()
        ori_cam_params = im_process.get_intrinsic_params(np.array(ori_K).flatten())
        new_cam_params = im_process.get_intrinsic_params(np.array(new_K).flatten())
        center_aligned_rgb = self.alignImageCenter(rgb, ori_cam_params, new_cam_params, new_img_shape) 
        center_aligned_depth = self.alignImageCenter(depth, ori_cam_params, new_cam_params, new_img_shape) 
        if mask_move:
            center_aligned_mask = self.alignImageCenter(mask.astype(np.uint8), ori_cam_params, new_cam_params, new_img_shape) 
        depth_scale = new_t[2]/ori_t[2] 
        x_scale = ori_t[2]/new_t[2]*new_cam_params['fx']/ori_cam_params['fx']
        y_scale = ori_t[2]/new_t[2]*new_cam_params['fy']/ori_cam_params['fy']
        new_rgb = cv2.resize(center_aligned_rgb, (0,0), fx=x_scale, fy=y_scale, interpolation=cv2.INTER_NEAREST)
        new_depth = cv2.resize(center_aligned_depth, (0, 0), fx=x_scale, fy=y_scale, interpolation=cv2.INTER_NEAREST)
        if mask_move:
            new_mask = cv2.resize(center_aligned_mask, (0, 0), fx=x_scale, fy=y_scale, interpolation=cv2.INTER_NEAREST)
        new_depth *= depth_scale

        center_shift_x = float(new_cam_params['cx']*(1-x_scale))
        center_shift_y = float(new_cam_params['cy']*(1-y_scale))
        x_shift = float((new_t[0] - ori_t[0])/new_t[2]*new_cam_params['fx']) + center_shift_x
        y_shift = float((new_t[1] - ori_t[1])/new_t[2]*new_cam_params['fy']) + center_shift_y
        translation_matrix = np.array([
            [1, 0, x_shift],
            [0, 1, y_shift]
        ])
        new_rgb= cv2.warpAffine(new_rgb, translation_matrix, new_img_shape)
        new_depth= cv2.warpAffine(new_depth, translation_matrix, new_img_shape)
        if mask_move:
            new_mask= cv2.warpAffine(new_mask, translation_matrix, new_img_shape).astype(bool)
        else:
            new_mask = None
        return new_rgb, new_depth, new_mask

    def alignImageCenter(self, ori_img, ori_cam_params, new_cam_params, new_img_shape):
        translation_matrix = np.array([
            [1, 0, new_cam_params['cx'] - ori_cam_params['cx']],
            [0, 1, new_cam_params['cy'] - ori_cam_params['cy']]
        ])
        new_img = cv2.warpAffine(ori_img, translation_matrix, new_img_shape)
        return new_img

    @lazy_property
    def target_obj_diameter(self):
        return self.model_diameters[self.target_obj_id]
    
    @lazy_property
    def max_delta_t_shift(self):
        if eval(self._kw['dynamic_bbox_enlarge']):
            delta_t_shift = np.float(self._kw['max_translation_shift'])
        else:
            delta_t_shift = np.float(self._kw['bbox_enlarge_level'])*self.target_obj_diameter/(2*np.sqrt(3))
        print('max_delta_t_shift', delta_t_shift)
        return delta_t_shift 

    @lazy_property
    def bbox_enlarge_level(self):
        if eval(self._kw['dynamic_bbox_enlarge']):
            max_delta_t_range = np.float(self._kw['max_translation_shift'])
            # In order to get at least half of the target object in the cropped patch, the bbox enlarge level have a minimum value.
            min_enlarge_level = max_delta_t_range*np.sqrt(3)*2/self.target_obj_diameter
            bbox_enlarge_level = max(np.float(self._kw['min_bbox_enlarge_level']), min_enlarge_level)
        else:
            bbox_enlarge_level = np.float(self._kw['bbox_enlarge_level'])
        print('Bbox enlarge level', bbox_enlarge_level)
        return bbox_enlarge_level 

    def get_bg_image_patchs(self, current_file_name):
        bg_dir_path = self._kw['background_base_dir']
        self.bg_rgb_list, self.bg_depth_list,_, self.bg_info_list, self.bg_gt_list= self.getRealImagesList(bg_dir_path, img_type='.png')

        bg_x = np.empty( (self.total_irr_img_num,) + self.shape, dtype=np.uint8 ) 
        gt_R = np.empty((self.total_irr_img_num,)+(3,3), dtype=np.float)
        gt_t = np.empty((self.total_irr_img_num,)+(3,1), dtype=np.float)
        delta_R = np.empty((self.total_irr_img_num,)+(3,3), dtype=np.float)
        delta_t = np.empty((self.total_irr_img_num,)+(3,1), dtype=np.float)
        noisy_R = np.empty((self.total_irr_img_num,)+(3,3), dtype=np.float)
        noisy_t = np.empty((self.total_irr_img_num,)+(3,1), dtype=np.float)

        scene_num = len(self.bg_rgb_list)

        bar = utils.progressbar_init('Generate background dataset: ', self.total_irr_img_num)
        bar.start()
        for i in range(self.total_irr_img_num):
            # while True:
            scene_id = random.randint(0, len(scene_num)-1)
            im_idx = random.randint(0, len(self.bg_rgb_list[scene_id])-1)
            depth_scale = self.bg_info_list[scene_id][im_idx]['depth_scale']
            depth_img = cv2.imread(self.bg_depth_list[scene_id][im_idx], cv2.IMREAD_ANYDEPTH)*depth_scale
            rgb_img = cv2.imread(self.bg_rgb_list[scene_id][im_idx])
            K = self.bg_info_list[scene_id][im_idx]['cam_K']
            gt_R[i] = self.bg_gt_list[scene_id][im_idx][0]['cam_R_m2c'] 
            t = self.bg_gt_list[scene_id][im_idx][0]['cam_t_m2c'] 
            gt_t[i] = np.array(t).reshape((3,1))

            H, W = depth_img.shape

            _, delta_t[i] = self.get_pose_noise()
            delta_R[i] = transform.random_rotation_matrix()[:3,:3]
            noisy_R[i] = np.matmul(delta_R[i].transpose(), gt_R[i])
            noisy_t[i] = gt_t[i] + delta_t[i]
            noisy_t[i] = self.keep_t_within_image(noisy_t[i], K, W, H)

            synthetic = (random.random()<=0.8)
            if synthetic:
                rendered_rgb, rendered_depth, render_mask = self.render_distractors(W, H, np.array(K).reshape((3,3)), gt_t[i], distractor_num=3)
                rgb_img, depth_img, _ = self.merge_images(rgb_img, depth_img, rendered_rgb, rendered_depth)
            bg_x[i], depth_vis_amount = self.train_images_preprocess(rgb_img, depth_img, noisy_R[i], noisy_t[i], K, bg_img = True)
            bar.update(i)
        bar.finish()
        print('Saving background dataset...')
        np.savez(current_file_name, bg_x = bg_x, gt_R= gt_R, gt_t = gt_t, delta_R= delta_R, delta_t = delta_t, noisy_R = noisy_R, noisy_t = noisy_t) 
        print('Successfully saved background dataset.')
        self.train_bg_x = bg_x 


    @lazy_property
    def train_img_fan_out(self):
        return int(self._kw['train_img_fan_out'])

    def get_images_of_target_object_dynamic(self, ori_image_type):
        if ori_image_type == 'real':
            ori_rgb, ori_depth, K, R, t = self.loadTargetImage()
            
            rotate = (random.random()<=0.9)
            if rotate:
                aug_angle = np.random.randint(0,180)
                ori_rgb, ori_depth, R, t = im_process.rotate_image(ori_rgb, ori_depth, R, t, K, aug_angle)
            syn_gt_rgb, syn_gt_depth, gt_mask = self.render_gt_image(W = ori_rgb.shape[1], H = ori_rgb.shape[0], K=K, R=R,t=t)
            ori_rgb = np.where(np.expand_dims(gt_mask, axis=2), ori_rgb, 0)
            obj_mask = np.abs(syn_gt_depth - ori_depth) <= 30
            ori_depth = np.where(obj_mask, ori_depth, 0) 

        elif ori_image_type == 'synthetic':
            ori_rgb, ori_depth, gt_mask, K, R, t = self.generate_synthetic_image()
            syn_gt_rgb, syn_gt_depth = ori_rgb.copy(), ori_depth.copy()
        else:
            print('Error: Unrecognized background type.')
            exit()
        return ori_rgb, ori_depth, syn_gt_rgb, syn_gt_depth, gt_mask, K, R, t
    
    
    def get_rotation_only_target_patch(self, gt_R, noisy_R):
        W, H = self.render_dim
        rgb, depth, _ = self.render_gt_image(W, H, self.render_K, gt_R, self.render_t)
        image_patch, _ = self.train_images_preprocess(rgb, depth, noisy_R, self.render_t, self.render_K, rotation_target=True)
        return image_patch 

    def get_rotation_only_target_image(self, gt_R):
        W, H = self.render_dim
        rgb, depth, _= self.render_gt_image(W, H, self.render_K, gt_R, self.render_t)
        return rgb, depth 

    @lazy_property
    def rotation_target_crop_param(self):
        cam_param = im_process.get_intrinsic_params(self.render_K)
        topleft_x, topleft_y, crop_w, crop_h = im_process.get_enlarged_bbox(
            cam_param, 
            self.render_t, 
            self.target_obj_diameter, 
            enlarge_scale=self.rotation_image_bbox_enlarge_level)
        print('rotation target crop dim:', topleft_x, topleft_y, crop_w, crop_h)
        return (topleft_x, topleft_y, crop_w, crop_h)
    
    @lazy_property
    def rotation_target_render_K(self):
        K = (self.render_K).flatten()
        topleft_x, topleft_y, _, _ = self.rotation_target_crop_param
        CX_INDEX = 2
        CY_INDEX = 5
        K[CX_INDEX] -= topleft_x
        K[CY_INDEX] -= topleft_y
        K = K.reshape(self.render_K.shape)
        return K
        
    def get_rotation_only_target_crop(self, gt_R):
        _, _, W, H = self.rotation_target_crop_param
        rgb, depth, _= self.render_gt_image(W, H, self.rotation_target_render_K, gt_R, self.render_t)
        return rgb, depth 
    
    def rotation_target_crop_check(self, number):
        for _ in range(number): 
            R = transform.random_rotation_matrix()[:3,:3]
            rgb, depth = self.get_rotation_only_target_crop(R)
            cv2.imshow('rgb', rgb)
            cv2.waitKey(5000)

    @lazy_property
    def rot_Ks(self):
        rot_Ks = np.tile(np.expand_dims(self.render_K, axis=0), (self.batch_size, 1))
        return rot_Ks
    
    @lazy_property
    def rot_ts(self) :
        rot_ts = np.tile(self.render_t.reshape(-1, 3,1), (self.batch_size, 1, 1))
        return rot_ts 

    def get_train_x_broadcast(self, batch_size, test = False):
        W, H = self.render_dim
        train_rgb_images = np.zeros((batch_size,) + (H, W, 3,), dtype=np.uint8) 
        train_depth_images = np.zeros((batch_size,) + (H, W,), dtype=np.float32) 
        syn_gt_rgbs = np.zeros((batch_size,) + (H, W, 3,), dtype=np.uint8) 
        syn_gt_depths = np.zeros((batch_size,) + (H, W,), dtype=np.float32) 
        if self.rotation_only_train_mode:
            rot_rgbs = np.empty((batch_size,) + (H, W, 3,), dtype=np.uint8) 
            rot_depths = np.empty((batch_size,) + (H, W, ), dtype=np.float32) 
        noisy_Rs = np.empty((batch_size,)+self.R_shape, dtype=np.float)
        noisy_ts = np.empty((batch_size,)+self.t_shape, dtype=np.float)
        delta_ts = np.empty((batch_size,)+self.t_shape, dtype=np.float)
        Ks = np.empty((batch_size,)+self.K_shape, dtype=np.float)

        for i in range(batch_size):  
            bg_type = self._choose_bg_type()

            if bg_type == 'tless':
                bg_type = 'real'
            if bg_type == 'linemod':
                bg_type = 'synthetic'

            train_rgb, train_depth, syn_gt_rgb, syn_gt_depth, _, Ks[i], gt_R, gt_t = self.get_images_of_target_object_dynamic(bg_type)
            if self.rotation_only_train_mode:
                rot_rgbs[i], rot_depths[i] = self.get_rotation_only_target_image(gt_R)
            h, w = train_rgb.shape[0:2]
            train_rgb_images[i, 0:h, 0:w] = train_rgb
            train_depth_images[i, 0:h, 0:w] = train_depth
            syn_gt_rgbs[i, 0:h, 0:w] = syn_gt_rgb
            syn_gt_depths[i, 0:h, 0:w] = syn_gt_depth
            delta_R, delta_ts[i] = self.get_pose_noise()
            noisy_Rs[i] = np.matmul(delta_R.transpose(), gt_R)
            noisy_ts[i] = gt_t + delta_ts[i]
            noisy_ts[i] = self.keep_t_within_image(noisy_ts[i], Ks[i], w, h)

        aug_rgb_patch, aug_depth_patch = self.get_aug_patchs(batch_size, test = test)
        train_x, obj_visable_mask = self.train_images_preprocess_broadcast_and_aug(train_rgb_images, train_depth_images, aug_rgb_patch, aug_depth_patch, noisy_Rs, noisy_ts, Ks)
        train_y, _ = self.train_images_preprocess(syn_gt_rgbs, syn_gt_depths, noisy_Rs, noisy_ts, Ks, broadcast = True)
        gt_mask = (train_y.astype(bool)).any(axis=3)
        pixel_num = np.sum(gt_mask.reshape(batch_size, -1)*1, axis = 1).reshape(batch_size, 1)
        if self.rotation_only_train_mode:
            train_y, _ = self.train_images_preprocess(rot_rgbs, rot_depths, noisy_Rs, self.rot_ts, self.rot_Ks, rotation_target=True, broadcast=True)
        return train_x, train_y, obj_visable_mask, delta_ts, pixel_num

    def get_train_x_broadcast_complete(self, batch_size, test = False):
        W, H = self.synthetic_image_render_dim

        train_rgb_images = np.zeros((batch_size,) + (H, W, 3,), dtype=np.uint8) 
        train_depth_images = np.zeros((batch_size,) + (H, W,), dtype=np.float32) 

        gt_rgb_images = np.zeros((batch_size,) + (400, 400, 3,), dtype=np.uint8) 
        gt_depth_images = np.zeros((batch_size,) + (400, 400,), dtype=np.float32) 
        gt_Ks = np.empty((batch_size,)+self.K_shape, dtype=np.float32)

        noisy_Rs = np.empty((batch_size,)+self.R_shape, dtype=np.float32)
        noisy_ts = np.empty((batch_size,)+self.t_shape, dtype=np.float32)
        gt_ts = np.empty((batch_size,)+self.t_shape, dtype=np.float32)
        delta_ts = np.empty((batch_size,)+self.t_shape, dtype=np.float32)
        Ks = np.empty((batch_size,)+self.K_shape, dtype=np.float32)

        for i in range(batch_size):  
            bg_type = self._choose_bg_type()

            train_rgb, train_depth, Ks[i], gt_R, gt_ts[i], noisy_Rs[i], noisy_ts[i], delta_R, delta_ts[i], gt_rgb_images[i], gt_depth_images[i], gt_Ks[i] = self.get_crops_of_target_object_no_gt(bg_type)
            h, w = train_rgb.shape[0:2]
            train_rgb_images[i, 0:h, 0:w] = train_rgb
            train_depth_images[i, 0:h, 0:w] = train_depth 
        
        # # One of the training sample doesn't add noise
        noisy_Rs[i] = gt_R
        noisy_ts, delta_ts = self.random_denoise_one_translation(noisy_ts, delta_ts)

        syn_gt_y, _ = self.train_images_preprocess(gt_rgb_images, gt_depth_images, noisy_Rs, noisy_ts, gt_Ks, broadcast = True)
        gt_mask = (syn_gt_y.astype(bool)).any(axis=3)

        aug_rgb_patch, aug_depth_patch = self.get_aug_patchs(batch_size, test = test)
        # aug_rgb_patch, aug_depth_patch = self.add_whole_black_augmentation_image(aug_rgb_patch, aug_depth_patch)
        train_x, obj_visable_mask, depth_no_rot = self.train_images_preprocess_broadcast_and_aug(train_rgb_images, train_depth_images, aug_rgb_patch, aug_depth_patch, noisy_Rs, noisy_ts, Ks, gt_mask = gt_mask)

        obj_visable_mask *= gt_mask.squeeze()

        rot_y, _ = self.train_images_preprocess(gt_rgb_images, gt_depth_images, noisy_Rs, gt_ts, gt_Ks, rotation_target=True, broadcast=True)

        return (train_x, syn_gt_y, depth_no_rot, rot_y, obj_visable_mask, gt_mask, delta_ts, noisy_Rs)

    @lazy_property
    def _bg_types_num(self):
        return len(self.bg_types)
    
    def _choose_bg_type(self):
        if len(self.bg_types) == 2:
            bg_type = 'real' if (random.random() < 0.3) else 'synthetic'
            return bg_type
        else:
            return self.bg_types[0]

    def get_train_x_broadcast_complete_acc(self, batch_size, test = False):
        W, H = self.synthetic_image_render_dim

        train_rgb_images = np.zeros((batch_size,) + (H, W, 3,), dtype=np.uint8) 
        train_depth_images = np.zeros((batch_size,) + (H, W,), dtype=np.float32) 

        gt_rgb_images = np.zeros((batch_size,) + (H, W, 3,), dtype=np.uint8) 
        gt_depth_images = np.zeros((batch_size,) + (H, W,), dtype=np.float32) 
        gt_Ks = np.empty((batch_size,)+self.K_shape, dtype=np.float32)

        _, _, rot_w, rot_h = self.rotation_target_crop_param
        rot_rgbs = np.empty((batch_size,) + (rot_h, rot_w, 3,), dtype=np.uint8) 
        rot_depths = np.empty((batch_size,) + (rot_h, rot_w, ), dtype=np.float32) 
        rot_noisy_Rs = np.empty((batch_size,)+self.R_shape, dtype=np.float32)

        delta_Rs = self._rot_noises[np.random.randint(0, self._rot_noises_num, batch_size)]
        gt_Rs = np.empty((batch_size,)+self.R_shape, dtype=np.float32)
        delta_ts = self._get_delta_ts(batch_size)
        gt_ts = np.empty((batch_size,)+self.t_shape, dtype=np.float32)
        Ks = np.empty((batch_size,)+self.K_shape, dtype=np.float32)

        for i in range(batch_size):  
            bg_type = self._choose_bg_type()

            rgb, depth, Ks[i], gt_Rs[i], gt_ts[i], gt_rgb_images[i], gt_depth_images[i], gt_Ks[i] = self.get_crops_of_target_object_no_noises(bg_type)
            true_h, true_w = rgb.shape[:-1]
            train_rgb_images[i, :true_h, :true_w] = rgb
            train_depth_images[i, :true_h, :true_w] = depth
            rot_gt_R = self.add_rotation_shift(gt_Rs[i], gt_ts[i]) 
            rot_rgbs[i], rot_depths[i] = self.get_rotation_only_target_crop(rot_gt_R)

        noisy_Rs = np.matmul(delta_Rs.transpose(0,2,1), gt_Rs)
        rot_noisy_Rs = self.add_rotation_shift(noisy_Rs, gt_ts) 
        noisy_ts = gt_ts + delta_ts
        
        # One of the training sample doesn't add noise
        noisy_Rs[i] = gt_Rs[i]
        rot_noisy_Rs[i] = rot_gt_R
        noisy_ts, delta_ts = self.random_denoise_one_translation(noisy_ts, delta_ts)

        syn_gt_y, _ = self.train_images_preprocess(gt_rgb_images, gt_depth_images, noisy_Rs, noisy_ts, gt_Ks, broadcast = True)
        gt_mask = (syn_gt_y.astype(bool)).any(axis=3)

        rot_rgbs = self.rotation_target_image_preprocess(rot_rgbs, rot_depths, rot_noisy_Rs)

        aug_rgb_patch, aug_depth_patch = self.get_aug_patchs(batch_size, test = test)

        if self.ycb_real:
            match_mask = np.abs(train_depth_images - gt_depth_images)<20
            gt_depth_images *= match_mask

        train_x, obj_visable_mask, depth_no_rot = self.train_images_preprocess_broadcast_and_aug(train_rgb_images, train_depth_images, aug_rgb_patch, aug_depth_patch, noisy_Rs, noisy_ts, Ks, gt_mask = gt_mask)

        obj_visable_mask *= gt_mask.squeeze()
        if self._gray_bg:
            rgb_mask = (rot_rgbs == [0,0,0,0,0,0]).all(axis=-1)
            rot_rgbs[rgb_mask] = self.gray_bg_color 
            syn_gt_mask = (syn_gt_y== [0,0,0,0,0,0]).all(axis=-1)
            syn_gt_y[syn_gt_mask] = self.gray_bg_color 

        return (train_x, syn_gt_y, depth_no_rot, rot_rgbs, rot_depths, obj_visable_mask, gt_mask, delta_ts, rot_noisy_Rs)

    @lazy_property
    def _gray_bg(self):
        try:
            gray_bg = eval(self._kw['gray_bg'])
        except:
            gray_bg = False
        return gray_bg 

    @lazy_property
    def gray_bg_color(self):
        return np.array([127,127,127,127,127,127])

    def random_denoise_one_translation(self, noisy_ts, delta_ts):
        i = random.randint(0,len(noisy_ts)-1)
        noisy_ts[i] -= delta_ts[i]
        delta_ts[i] = 0
        return noisy_ts, delta_ts

    def get_crops_of_target_object_no_gt(self, ori_image_type):
        delta_R, delta_t = self.get_pose_noise()
        if ori_image_type == 'real':
            ori_rgb, ori_depth, K, gt_R, gt_t = self.loadTargetImage()
            rotate = (random.random()<=0.9)
            if rotate:
                aug_angle = np.random.randint(0,180)
                ori_rgb, ori_depth, gt_R, gt_t = im_process.rotate_image(ori_rgb, ori_depth, gt_R, gt_t, K, aug_angle)
            noisy_R = np.matmul(delta_R.transpose(), gt_R)
            noisy_t = gt_t + delta_t

        elif ori_image_type == 'synthetic':
            gt_R = transform.random_rotation_matrix()[:3,:3]
            noisy_R = np.matmul(delta_R.transpose(), gt_R)
            gt_t = self.get_synthetic_noisy_t()
            noisy_t = gt_t + delta_t
            ori_rgb, ori_depth, _, K = self.generate_synthetic_image_crop( gt_t, gt_R, self.render_K, random_light = True)
        else:
            print('Error: Unrecognized background type.')
            exit()
        K = K.reshape(self.K_shape)
        gt_rgb, gt_depth, _, gt_K = self.generate_synthetic_image_crop(gt_t, gt_R, K.copy(), random_light=False)
        return ori_rgb, ori_depth, K, gt_R, gt_t, noisy_R, noisy_t, delta_R, delta_t, gt_rgb, gt_depth, gt_K

    def check_real_img_status(self):
        if len(self.real_rgbs) == 0:
            if 'real' in self.bg_types:

                # obj_dir_path = self._kw['foreground_base_dir']
                if not self.ycb_real:
                    self.load_tless_real_training_images_of_target()
                    # self.obj_rgb_list, self.obj_depth_list, self.obj_mask_list, self.obj_info_list, self.obj_gt_list = self.getRealImagesList(obj_dir_path, img_type='.png')
                    # self.load_real_training_images_of_target()
                    self.total_training_images = len(self.real_Ks)
                else:
                    self.load_ycb_real_training_images_of_target()

    def get_crops_of_target_object_no_noises(self, ori_image_type):
        if ori_image_type == 'real':
            self.check_real_img_status()
            im_id = np.random.randint(0, self.total_training_images)
            ori_rgb, ori_depth = self.real_rgbs[im_id], self.real_depths[im_id]
            K, gt_R, gt_t = self.real_Ks[im_id], self.real_Rs[im_id], self.real_ts[im_id]
            rotate = (random.random() < 0.8)
            if rotate:
                aug_angle = np.random.randint(0,180)
                ori_rgb, ori_depth, gt_R, gt_t = im_process.rotate_image(ori_rgb, ori_depth, gt_R, gt_t, K, aug_angle)

        elif ori_image_type == 'synthetic':
            gt_R = transform.random_rotation_matrix()[:3,:3]
            gt_t = self.get_synthetic_noisy_t()
            ori_rgb, ori_depth, _, K = self.generate_synthetic_image_crop( gt_t, gt_R, self.render_K, random_light = True)
        else:
            print('Error: Unrecognized background type.')
            exit()
        K = K.reshape(self.K_shape)
        gt_rgb, gt_depth, _, gt_K = self.generate_synthetic_image_crop(gt_t, gt_R, K.copy(), random_light=False)
        return ori_rgb, ori_depth, K, gt_R, gt_t, gt_rgb, gt_depth, gt_K

    @lazy_property
    def rotation_target_render_Ks(self):
        rot_Ks = np.tile(np.expand_dims(self.rotation_target_render_K, axis=0), (self.batch_size, 1))
        return rot_Ks
        
    def get_train_x_broadcast_acc(self, batch_size, test = False):
        patch_shape = self.shape[:2]

        train_rgb_patch = np.empty((batch_size,) + patch_shape + (3,), dtype=np.uint8) 
        train_depth_patch = np.empty((batch_size,) + patch_shape + (3,), dtype=np.float32) 

        gt_rgb_patch = np.empty((batch_size,) + patch_shape + (3,), dtype=np.uint8) 
        gt_depth_patch = np.empty((batch_size,) + patch_shape + (3,), dtype=np.float32) 

        _, _, rot_w, rot_h = self.rotation_target_crop_param
        rot_rgbs = np.empty((batch_size,) + (rot_h, rot_w, 3,), dtype=np.uint8) 
        rot_depths = np.empty((batch_size,) + (rot_h, rot_w, ), dtype=np.float32) 

        noisy_Rs = np.empty((batch_size,)+self.R_shape, dtype=np.float32)
        noisy_ts = np.empty((batch_size,)+self.t_shape, dtype=np.float32)
        delta_ts = np.empty((batch_size,)+self.t_shape, dtype=np.float32)

        for i in range(batch_size):  
            img_type = self._choose_bg_type()
            train_rgb_crop, train_depth_crop, gt_rgb_crop, gt_depth_crop, gt_mask, K, gt_R, gt_t, noisy_Rs[i], noisy_ts[i], delta_R, delta_ts[i] = self.get_crops_of_target_object(img_type)
            rot_rgbs[i], rot_depths[i] = self.get_rotation_only_target_crop(gt_R)
            train_rgb_patch[i], train_depth_patch[i] = im_process.to_3D_patch(train_rgb_crop, train_depth_crop, K, patch_shape)
            gt_rgb_patch[i], gt_depth_patch[i] = im_process.to_3D_patch(gt_rgb_crop, gt_depth_crop, K, patch_shape)
            
        aug_rgb_patch, aug_depth_patch = self.get_aug_patchs(batch_size, test = test)
        train_x, obj_visable_mask =self.train_patchs_preprocess_broadcast_and_aug(train_rgb_patch, train_depth_patch, aug_rgb_patch, aug_depth_patch, noisy_Rs, noisy_ts)       
        
        return (train_x, obj_visable_mask, gt_rgb_patch, gt_depth_patch, rot_rgbs, rot_depths, noisy_Rs, noisy_ts, delta_ts)
    
    @tf.function
    def image_patch_preprocess_tf(self, rgb, depth, Rs, ts):
        depth_patch, depth_vis_mask = im_process.depth_patch_preprocess_broadcast_tf(depth, Rs, ts, self.target_obj_diameter, self.bbox_enlarge_level)
        patch = tf.concat((rgb, depth_patch), axis=3)
        return patch, depth_vis_mask 

    @tf.function
    def rotation_target_image_preprocess_tf(self, rgbs, depths, Rs):
        K = tf.expand_dims(tf.cast(self.rotation_target_render_K, dtype=tf.float32), axis=0)
        ts = tf.expand_dims(tf.cast(self.render_t, dtype=tf.float32), axis=0)
        ts = tf.reshape(ts, (-1,) + self.t_shape)
        depths = im_process.rotation_depth_process_tf(depths, K, Rs, ts, self.target_obj_diameter, self.rotation_image_bbox_enlarge_level,self.shape[:2])
        rgbs = tf.image.resize(rgbs, self.shape[:2], method='nearest')
        rot_patch = tf.concat((rgbs, depths), axis = -1)
        return rot_patch
 
    def rotation_target_image_preprocess(self, rgbs, depths, Rs):
        depths = im_process.rotation_depth_process(depths, self.rotation_target_render_Ks, Rs, self.rot_ts, self.target_obj_diameter, self.rotation_image_bbox_enlarge_level,self.shape[:2])
        rgbs = im_process.batch_images_resize(rgbs, self.shape[:2])
        patch = np.concatenate((rgbs, depths), axis = -1)
        return patch

    @lazy_property
    def rot_raw_img_shape(self):
        _,_, w, h = self.rotation_target_crop_param
        return (h,w,3)

    def get_aug_patchs(self, batch_size, test = False):
        rand_idcs = np.random.randint(0, self.noof_aug_imgs, batch_size)
        if len(self.aug_rgb_images) == 0:
            print('Error please load augmentation images dataset first.')
            exit()
        depth_shift = np.random.uniform(0,50, (batch_size,1,1))
        depth = self.aug_depth_images[rand_idcs] 
        depth[:,:,:,-1] += depth_shift
        return self.aug_rgb_images[rand_idcs], depth 

    def train_images_preprocess_broadcast_and_aug(self, rgb, depth, aug_rgb_patch, aug_depth_patch, R, t, K, gt_mask= None, seg_rate=None):
        rgb_patch = self.rgb_preprocess(rgb, t, K, broadcast=True)
        depth_patch, aug_vis_mask, depth_patch_no_rot = self.depth_preprocess_broadcast_and_aug(depth, rgb_patch, aug_depth_patch, R, t, K, gt_mask = gt_mask, seg_rate = seg_rate)
        rgb_patch[1:] = np.where(aug_vis_mask[1:], rgb_patch[1:], aug_rgb_patch[1:])
        rgb_patch[0] = rgb_patch[0]*aug_vis_mask[0]
        image_patch_stack = np.concatenate((rgb_patch, depth_patch), axis=3)
        return image_patch_stack, aug_vis_mask.squeeze(), depth_patch_no_rot

    def train_patchs_preprocess_broadcast_and_aug(self, rgb_patch, depth, aug_rgb_patch, aug_depth_patch, R, t):
        depth_patch, aug_vis_mask = im_process.depth_patch_preprocess_broadcast_and_aug(depth, rgb_patch, aug_depth_patch, R, t, self.shape[:2], self.target_obj_diameter, self.bbox_enlarge_level)
        rgb_patch = np.where(aug_vis_mask, rgb_patch, aug_rgb_patch)
        image_patch_stack = np.concatenate((rgb_patch, depth_patch), axis=3)
        return image_patch_stack, aug_vis_mask.squeeze() 

    def depth_preprocess_broadcast_and_aug(self, depth_image, rgb_patch, aug_depth_patch, R, t, K, gt_mask = None, seg_rate = None):
        depth_patch, visable_mask, depth_no_rot = im_process.depth_image_preprocess_broadcast_and_aug(
            depth_image, rgb_patch, aug_depth_patch, 
            R, t, K, 
            (self.patch_W, self.patch_H), 
            self.target_obj_diameter, 
            self.bbox_enlarge_level,
            gt_mask = gt_mask, seg_rate= seg_rate
            )
        return depth_patch, visable_mask, depth_no_rot
    
    def crop_and_resize_train_image(self, rgb, depth, K, t, rotation_target = False, bg_img = False):
        if rotation_target:
            bbox_enlarge_level = self.rotation_image_bbox_enlarge_level
        elif bg_img:
            bg_bbox_enlarge_level = self.bbox_enlarge_level*random.uniform(1,3)
        else:
            bbox_enlarge_level = self.bbox_enlarge_level
        image_patch = im_process.crop_and_resize(rgb, depth, K, self.target_obj_diameter, t, bbox_enlarge_level, self.shape[:2])
        return image_patch 
    
    def depth_patch_preprocess_broadcast(self, depth_patchs, Rs, ts, rotation_target= False):
        if rotation_target:
            bbox_enlarge_level = self.rotation_image_bbox_enlarge_level
        else:
            bbox_enlarge_level = self.bbox_enlarge_level
        patchs, vis_mask = im_process.depth_patch_preprocess_broadcast(depth_patchs, Rs, ts,  self.target_obj_diameter, bbox_enlarge_level)
        return patchs, vis_mask

    def load_augmentation_images(self):
        if len(self.aug_rgb_images) > 0:
            print('Augmentation images has been loaded.')
            return
        current_config_hash = hashlib.md5((str('aug_images')+str(self.target_obj_id)+ str(self.noof_aug_imgs)).encode('utf-8')).hexdigest()
        current_file_name = os.path.join(self.dataset_path, current_config_hash + '.npz')
        print('Augmentation dataset file name:', current_file_name)
        if not os.path.exists(current_file_name):
            bg_dir_path = self._kw['background_base_dir']
            self.bg_rgb_list, self.bg_depth_list,_, self.bg_info_list, self.bg_gt_list = self.getRealImagesList(bg_dir_path, img_type='.png')
            self.generate_augmentation_dataset_file(current_file_name)
        else:
            aug_data = np.load(current_file_name)
            self.aug_rgb_images = aug_data['aug_rgb'].astype(np.uint8) 
            self.aug_depth_images = aug_data['aug_depth'].astype(np.float)
        print('loaded %s augmentation images' % (len(self.aug_rgb_images)))

    def generate_augmentation_dataset_file(self, current_file_name):
        rgb_patch, depth_patch = self.augmentation_dataset_variables_init()

        bar = utils.progressbar_init('Generate augmentation dataset: ', self.noof_aug_imgs)

        bar.start()
        for i in np.arange(self.noof_aug_imgs):
            bar.update(i)
            rgb_patch[i], depth_patch[i] = self.generate_augmentation_image_patch()
        bar.finish()

        print('Saving augmentation dataset...')
        np.savez(
            current_file_name, 
            aug_rgb = rgb_patch, 
            aug_depth = depth_patch, 
        )
        print('Successfully saved augmentation dataset')
        exit()
        # self.aug_rgb_images = rgb_patch
        # self.aug_depth_images = depth_patch
    
    def augmentation_dataset_variables_init(self):
        rgb_patch = np.empty( (self.noof_aug_imgs,) + self.shape[:2] + (3,), dtype=np.uint8) 
        depth_patch = np.empty( (self.noof_aug_imgs,) + self.shape[:2] + (3,), dtype=np.float) 
        return rgb_patch, depth_patch 
    
    def generate_augmentation_image_patch(self):
        rgb, depth, K, t = self.random_generate_augmentation_image()
        patch = self.crop_and_resize_train_image(rgb, depth, K, t)
        rgb = patch[:,:, 0:3].astype(np.uint8)
        depth = patch[:,:, 3:] - t.reshape((1,3))
        return rgb, depth 
    
    def random_generate_augmentation_image(self):
        bg_rgb, bg_depth, K, R, t = self.random_get_one_background_image()
        H, W = bg_rgb.shape[:2]
        distractor_centre = self.get_distractor_centre(t) 
        distractor_centre = self.keep_t_within_image(distractor_centre, K, W, H)
        distractor_rgb, distractor_depth, _ = self.render_distractors(W = W, H = H, K=K, t=distractor_centre)
        rgb, depth, _ = self.merge_images(bg_rgb, bg_depth, distractor_rgb, distractor_depth)
        return (rgb, depth, K, distractor_centre) 
    
    def get_distractor_centre(self, t):
        max_shift = min(100, self.target_obj_diameter)
        x_y_shift = np.random.uniform(-max_shift, max_shift, (2,1))
        z_shift = np.random.uniform(-1.2*max_shift, -0.5*max_shift)
        t[:2] = t[:2] + x_y_shift
        t[2] += z_shift
        return t

    
    def random_get_one_background_image(self):
        scene_num = len(self.bg_rgb_list)
        scene_id = random.randint(0, scene_num-1)
        im_idx = random.randint(0, len(self.bg_rgb_list[scene_id]) -1)
        depth_scale = self.bg_info_list[scene_id][im_idx]['depth_scale']
        depth_img = cv2.imread(self.bg_depth_list[scene_id][im_idx], cv2.IMREAD_ANYDEPTH)*depth_scale
        rgb_img = cv2.imread(self.bg_rgb_list[scene_id][im_idx])
        K = self.bg_info_list[scene_id][im_idx]['cam_K']
        obj_id = random.randint(0, len(self.bg_gt_list[scene_id][im_idx])-1)
        R = self.bg_gt_list[scene_id][im_idx][obj_id]['cam_R_m2c'] 
        t = self.bg_gt_list[scene_id][im_idx][obj_id]['cam_t_m2c'] 
        t = np.array(t).reshape((3,1))
        return (rgb_img, depth_img, K, R, t)
