# -*- coding: utf-8 -*-
import numpy as np

import tensorflow as tf

from utils import lazy_property
from pysixd_stuff import transform
from sixd_rot import rot_6d
from operator import itemgetter
import random
import time
import os
import utils
# from render_image import RenderImage

class PriorPoseDataset(object):

    def __init__(self, **kw):
        self._kw = kw
        self._eval_dataset = []
        self._train_dataset = []
        self._train_data_num = 0
        self._eval_data_num = 0
        self._test_dataset = []
        self._test_data_num = 0
        self._test_seqs_set = []
        self._test_seqs_start_id = 0

    @lazy_property
    def _rot_noises_dataset_path(self):
        return self._kw['lstm_rotation_noise_dataset_path']

    @lazy_property
    def _rot_noises_num(self):
        return int(self._kw['lstm_rotation_noise_num'])

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

    @lazy_property
    def _lower_bound(self):
        return float(self._kw['lstm_t_lower_bound'])

    @lazy_property
    def _upper_bound(self):
        return float(self._kw['lstm_t_upper_bound'])
    
    @lazy_property
    def _target_len(self):
        return int(self._kw['lstm_target_length'])
    
    @lazy_property
    def _input_time_steps(self):
        return int(self._kw['lstm_input_time_steps']) 
    
    @lazy_property
    def seq_length(self):
        return self._input_time_steps
    
    @lazy_property
    def _max_delta_t_shift(self):
        return float(self._kw['lstm_max_delta_t_shift'])
    
    @lazy_property
    def _rot_noise_level(self):
        return float(self._kw['lstm_rotation_noise_level'])
    
    @lazy_property
    def _eval_ratio(self):
        return float(self._kw['lstm_test_ratio'])
    
    @lazy_property
    def _dataset_path(self):
        return self._kw['lstm_dataset_path']
    
    @lazy_property
    def _test_dataset_path(self):
        return self._kw['lstm_test_dataset_path']

    @lazy_property
    def x_shape(self):
        return (self._input_time_steps, 4, 4)
    
    @lazy_property
    def y_shape(self):
        if self._target_len > 1:
            return (self._target_len, 4, 4)
        else:
            return (4,4)

    def load_dataset(self, test = False):
        if test:
            dataset_path = self._test_dataset_path
        else:
            dataset_path = self._dataset_path
        print('Loading dataset:', dataset_path)
        import json
        with open(dataset_path) as json_file:
            data = json.load(json_file)
            ori_dataset = []
            for i in range(len(data)):
                path = np.array(data[str(i)])
                path = self._normalize_translation(path)
                ori_dataset.append(path)
        if not test:
            self._train_dataset, self._eval_dataset = self._sep_train_eval_dataset(ori_dataset)
            self._train_data_num = len(self._train_dataset)
            self._eval_data_num = len(self._eval_dataset)
        else:
            self._test_dataset = ori_dataset
            self._test_data_num = len(ori_dataset)
        print('Successfully loaded', len(ori_dataset), 'pose paths.')

    def _sep_train_eval_dataset(self, dataset):
        eval_indices, train_indices = self.get_train_eval_indices(len(dataset))
        train_dataset = itemgetter(*train_indices)(dataset) 
        eval_dataset = itemgetter(*eval_indices)(dataset)
        return train_dataset, eval_dataset 

    def data_generator(self, batch_size, eval_data=False):
        while True:
            paths = self._random_paths(batch_size, eval_data = eval_data)
            seqs = []
            for path in paths:
                seq = self._choose_timesteps(path)
                if not eval_data:
                    seq = self._augment_sequence(seq)
                seqs.append(seq)
            seqs = self.dataset_padding(seqs, target_len= self._input_time_steps+self._target_len)
            x = seqs[:, :-self._target_len, :, :]
            y = seqs[:, -self._target_len:, :, :]
            if self._target_len == 1:
                y = y.reshape((-1,4,4))
            yield x, y
            
    def batch(self, batch_size, eval_data=False):
        paths = self._random_paths(batch_size, eval_data = eval_data)
        seqs = []
        for path in paths:
            seq = self._choose_timesteps(path)
            if not eval_data:
                seq = self._augment_sequence(seq)
            seqs.append(seq)
        seqs = self.dataset_padding(seqs, target_len=self._input_time_steps+self._target_len)
        x = seqs[:, :-self._target_len]
        y = seqs[:, -self._target_len:]
        if self._target_len == 1:
            y = y.reshape((-1,4,4))
        return x, y
    
    def test_batch(self, batch_size):
        self._generate_test_time_steps()
        if self._test_seqs_start_id == len(self._test_seqs_set):
            return (None, None)
        start = self._test_seqs_start_id 
        end = min(self._test_seqs_start_id+batch_size, len(self._test_seqs_set))
        x = self._test_seqs_set[start:end, :-self._target_len]
        y = self._test_seqs_set[start:end, -self._target_len]
        if self._target_len == 1:
            y = y.reshape((-1,4,4))
        self._test_seqs_start_id = end
        return (x,y)

    def normalize_t(self, path):
        return self._normalize_translation(path)

    def de_normalize(self, path):
        return self._de_normalize(path)
    
    def de_normalize_t(self, t):
        t = t.copy()
        t[:,:2] *= self._upper_bound
        t[:,2] = (t[:,2]+1)*self._upper_bound/2
        return t

    def _normalize_translation(self, path):
        if path.ndim == 2:
            path = np.expand_dims(path, axis=0)
        path = path.copy()
        if path.ndim == 3:
            path[:,:2,3] = path[:,:2,3]/self._upper_bound
            path[:,2,3] = path[:,2,3]/(self._upper_bound)*2 -1 
        else:
            path[:,:,:2,3] = path[:, :,:2,3]/self._upper_bound
            path[:,:, 2,3] = path[:, :, 2,3]/(self._upper_bound)*2 -1 
        return path

    def _de_normalize(self, path):
        if path.ndim == 2:
            path = np.expand_dims(path, axis=0)
        path = path.copy()
        path[:,:2,3] = path[:,:2,3]*self._upper_bound
        path[:,2,3] = (path[:,2,3]+1)*self._upper_bound/2  
        return path
    
    def _augment_sequence(self, seq):
        flags = np.random.randint(0, 2, 9)
        if flags[0]:
            seq = self._reverse_rot(seq)
        if flags[1]:
            seq = self._reverse_translation(seq)
        if flags[2]:
            seq = self._add_rotation_shift(seq)
        if flags[3]:
            seq = self._add_translation_shift(seq)
        if flags[4]:
            seq = self._swap_axis(seq)
        if flags[5]:
            seq = self._fix_one_axis(seq)
        if flags[6]:
            seq = self._add_small_noises(seq)
        if flags[7]:
            seq = self._delete_elements(seq)
        if flags[8]:
            seq = self._swap_elements(seq)
        seq[:,:3,3] = np.clip(seq[:, :3, 3], -1, 1) 
        return seq
    
    def _delete_elements(self, seq):
        seq_len = len(seq)
        if seq_len > 5:
            delte_num = np.random.randint(1, 3)
            delte_id = np.random.randint(0, seq_len, (1, delte_num)) 
            new_seq = np.delete(seq, delte_id, axis=0)
        else:
            new_seq = seq
        return new_seq

    def _swap_elements(self, seq):
        seq = seq.copy()
        if len(seq) > 4:
            swap_id = np.random.randint(0, len(seq)-2) 
            temp = seq[swap_id].copy()
            seq[swap_id] = seq[swap_id+1]
            seq[swap_id + 1] = temp
        return seq

    def _add_small_noises(self, seq):
        seq_len = len(seq)
        new_seq = seq.copy()
        noise_ids = np.random.randint(0, self._rot_noises_num, seq_len)
        rot_noises = self._rot_noises[noise_ids]
        new_seq[:,:3,:3] = np.matmul(rot_noises, seq[:,:3,:3])
        new_seq[:,:3, 3] += self._get_translation_noise(length=seq_len)
        return new_seq

    def _get_empty_seq(self, len, dim=4):
        return np.tile(np.eye(dim), [len,1,1])

    def _get_pose_noise(self):
        pose = self._get_rotation_noise()
        max_delta_t_range = self._max_delta_t_shift
        delta_t = (np.random.uniform(0, max_delta_t_range*2, (3,)) - max_delta_t_range)/self._upper_bound
        delta_t[2] = delta_t[2]*2
        pose[:3,3] = delta_t
        return pose 
    
    def _get_translation_noise(self, length=None):
        shape = (length, 3) if length is not None else (3,)
        max_delta_t_range = self._max_delta_t_shift
        delta_t = (np.random.uniform(0, max_delta_t_range*2, shape) - max_delta_t_range)/self._upper_bound
        if length is None:
            delta_t[2] = delta_t[2]*2
        else:
            delta_t[:,2] = delta_t[:, 2]*2
        return delta_t

    def _get_rotation_noise(self):
        noise_level = self._rot_noise_level 
        angle = np.random.uniform(-noise_level*np.pi, noise_level*np.pi)
        axis = np.random.rand(3)
        return transform.rotation_matrix(angle, axis)[:3,:3]

    def _swap_axis(self, seq):
        seq.copy()
        axis = np.random.randint(0,3,2)
        temp = seq[:, axis[0],3].copy()
        seq[:,axis[0], 3] = seq[:, axis[1], 3]
        seq[:,axis[1],3] = temp
        return seq

    def _fix_one_axis(self, seq):
        seq.copy()
        fix = np.random.rand(1)
        axis = np.random.randint(0,3,1)
        seq[:,axis,3] = fix
        return seq

    def _add_rotation_shift(self, seq):
        seq = seq.copy()
        rot = seq[:,:3,:3].copy()
        rot_shift = np.expand_dims(transform.random_rotation_matrix()[:3,:3].T, axis=0)
        seq[:,:3,:3] = np.matmul(rot_shift, rot) 
        return seq

    def _add_translation_shift(self, seq):
        xy_shift = np.random.rand(2)
        z_shift = np.random.rand(1)
        seq[:,:2,3] = seq[:, :2, 3] + xy_shift
        seq[:,2,3] = seq[:, 2, 3] + z_shift
        return seq 

    def _choose_timesteps(self, path):
        total_num = len(path)
        max_skip_num = min(int(total_num/10), 5)
        skip_step = random.randint(1,max_skip_num)
        begin = random.randint(0, int((total_num-self._target_len-1)/skip_step))
        # Note: the sequence num includes the target number
        seq_num = random.randint(2, self._input_time_steps+self._target_len)
        end = min(begin+seq_num*skip_step, total_num)
        out = path[begin: end:skip_step].copy()
        if (len(out) <= 2) or (out.shape == (4, 4)):
            out = np.tile(out,(2, 1, 1))
        return out
    
    def _generate_test_time_steps(self, path_id = None):
        if len(self._test_seqs_set) > 0:
            return 
        if self._test_dataset == []:
            self.load_dataset(test=True)
        seqs = []
        for path in self._test_dataset:
            seqs += self._generate_path_seqs(path)
        self._test_seqs_set = self.dataset_padding(seqs, self._input_time_steps+self._target_len)
    
    def _generate_path_seqs(self, path):
        seqs = []
        for end in range(1, len(path)+1):
            start = max(end-self._input_time_steps - self._target_len, 0)
            seq = path[start:end]
            if len(seq) <= 2:
                seq = np.tile(path[0],(2,1,1))
            seqs.append(seq)
        return seqs

    def _reverse_rot(self, seq):
        seq[:,:3,:3] = seq[:,:3,:3][::-1]
        return seq 
    
    def _reverse_translation(self, seq):
        seq[:,:3,3] = seq[:,:3,3][::-1]
        return seq

    def _random_paths(self, batch_size, eval_data = False):
        if eval_data:
            path_ids = np.random.randint(0, self._eval_data_num, batch_size)
            return itemgetter(*path_ids)(self._eval_dataset) 
        else:
            path_ids = np.random.randint(0, self._train_data_num, batch_size)
            return itemgetter(*path_ids)(self._train_dataset) 

    def _get_longest_len(self, data):
        if len(data) == 0:
            print('Error: _get_longest_len: empty data')
            exit()
        max_len = 0
        for path in data:
            if len(path) > max_len:
                max_len = len(path)
        return max_len

    def get_data_from_indices(self, data, indices):
        paths = []
        for index in indices:
            path = np.array(data[str(index)])
            paths.append(path)
        return paths
    
    def get_train_eval_indices(self, total_data_num):
        eval_num = int(total_data_num*self._eval_ratio)
        train_num = total_data_num - eval_num 
        np.random.seed(12)
        shuffled_indices = np.random.permutation(total_data_num)
        return shuffled_indices[:eval_num], shuffled_indices[eval_num:]
    
    @lazy_property
    def identical_sequence(self):
        return np.tile(np.eye(4), [self._input_time_steps + self._target_len, 1, 1])

    def sequence_padding(self, path):
        out_path = self.identical_sequence
        out_path[-len(path):] = np.array(path)
        return out_path

    def dataset_padding(self, data, target_len = None):
        if data[0].ndim == 2:
            new_data = []
            new_data.appen(data) 
            data = new_data
        length = self._get_longest_len(data) if target_len is None else target_len
        data_num = len(data)
        new_data = np.tile(np.eye(4), (data_num, length, 1, 1))
        for i in range(data_num):
            path = data[i].copy()
            new_data[i, -len(path):] = path
        return new_data 
    
    def get_test_path_seqs(self, path_id = 0):
        if self._test_dataset == []:
            self.load_dataset(test=True)
        path = self._test_dataset[path_id]
        seqs = self._generate_path_seqs(path)
        seqs = self.dataset_padding(seqs)
        x = seqs[:, :-self._target_len]
        y = seqs[:, -self._target_len:]
        if self._target_len == 1:
            y = y.reshape((-1,4,4))
        return (x,y)