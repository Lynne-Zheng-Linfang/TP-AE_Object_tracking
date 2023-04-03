 # -*- coding: utf-8 -*-
import os
import configparser
import argparse
import numpy as np
import signal
import shutil
import cv2
import time


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import progressbar
import tensorflow as tf

from . import ae_factory as factory
from . import utils as u
from .combiner_module import CombinerModule
from .ae_pose_predictor import PosePredictor
# import faulthandler
# faulthandler.enable()

def main():
    workspace_path = os.environ.get('AE_WORKSPACE_PATH')

    if workspace_path is None:
        print('Please define a workspace path:\n')
        print('export AE_WORKSPACE_PATH=/path/to/workspace\n')
        exit(-1)

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name")
    parser.add_argument("-d", action='store_true', default=False)
    parser.add_argument("-gen", action='store_true', default=False)
    parser.add_argument("-r", action="store_true", default=False)
    parser.add_argument("-fine", action="store_true", default=False)
    parser.add_argument("-tdis", action="store_true", default=False)
    parser.add_argument('-cl', action="store_true", default=False, help='delta t classification mode') 
    parser.add_argument('-dynamic', action="store_true", default=False, help='generate train data online') 
    parser.add_argument('-mask', action="store_true", default=False, help='delta t classification mode') 
    parser.add_argument('-tonly', action="store_true", default=False, help='translation only mode') 
    parser.add_argument('-with_pred', action="store_true", default=False, help='train network with predicted image') 
    parser.add_argument('-epoch', type=int, default = 50) 
    arguments = parser.parse_args()

    full_name = arguments.experiment_name.split('/')

    experiment_name = full_name.pop()
    experiment_group = full_name.pop() if len(full_name) > 0 else ''

    debug_mode = arguments.d
    generate_data = arguments.gen
    rotation_only_mode = arguments.r
    fine_tune_mode= arguments.fine
    classication_mode = arguments.cl
    t_distribution_mode = arguments.tdis
    mask_check = arguments.mask
    dynamic_mode = arguments.dynamic
    train_epoch = arguments.epoch

    cfg_file_path, checkpoint_file, ckpt_dir, train_fig_dir, dataset_path, test_summary_dir = u.file_paths_init(workspace_path, experiment_name, experiment_group, is_training= True)

    args = configparser.ConfigParser()
    args.read(cfg_file_path)

    per_process_gpu_memory_fraction = 0.4
    predictor = PosePredictor(experiment_name, dataset_path, args, per_process_gpu_memory_fraction, ckpt_dir, is_training=False, combiner_training = True)

    combiner = CombinerModule(experiment_name, dataset_path, args,per_process_gpu_memory_fraction, ckpt_dir, is_training=True)

    # predictor.queue_start()
    # for _ in range(20):
    #     train_x = predictor.get_rotation_reconstr_image()
    #     iter_num = int(len(train_x)/4)
    #     for i in range(iter_num) :
    #         cv2.imshow('rgb_only_rgb', train_x[i*4+0,:,:,0:3])
    #         cv2.imshow('rgb_only_depth', train_x[i*4+0,:,:,3:6])
    #         cv2.imshow('train_rgb', train_x[i*4+1,:,:,0:3])
    #         cv2.imshow('train_depth', train_x[i*4+1,:,:,3:6])
    #         cv2.imshow('esti_rgb', train_x[i*4+2,:,:,0:3])
    #         cv2.imshow('esti_depth', train_x[i*4+2,:,:,3:6])
    #         cv2.imshow('tar_rgb', train_x[i*4+3,:,:,0:3])
    #         cv2.imshow('tar_depth', train_x[i*4+3,:,:,3:6])
    #         cv2.waitKey(5000)
    # predictor.queue_stop()
    # exit()

    predictor.queue_start()
    init_epoch_num = combiner.init_epoch_num()
    if init_epoch_num <= train_epoch:
        for _ in range(init_epoch_num, train_epoch):
            combiner.train(predictor, log_write_interval=20)
    predictor.queue_stop()
    print('Finished training combiner')
if __name__ == '__main__':
    main()
