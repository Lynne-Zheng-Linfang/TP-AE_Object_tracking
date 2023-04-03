 # -*- coding: utf-8 -*-
import os
import configparser
import argparse
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import ae_factory as factory
import utils as u
from prior_pose_estimator import PriorPoseEstimator 

def main():
    workspace_path = os.environ.get('AE_WORKSPACE_PATH')

    if workspace_path is None:
        print('Please define a workspace path:\n')
        print('export AE_WORKSPACE_PATH=/path/to/workspace\n')
        exit(-1)

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name")
    parser.add_argument('-epoch', type=int, default = 30, help='training epochs, default is 50') 
    parser.add_argument('-steps', type=int, default = 5000, help='training steps, default is 1000') 
    parser.add_argument("-test", action='store_true', default=False)
    parser.add_argument("-gpu", type=float, default = 0.5, help='gpu fraction, default is 0.5')
    arguments = parser.parse_args()

    full_name = arguments.experiment_name.split('/')

    experiment_name = full_name.pop()
    experiment_group = full_name.pop() if len(full_name) > 0 else ''

    train_epoch = arguments.epoch
    train_steps = arguments.steps
    is_training = not(arguments.test)

    cfg_file_path, checkpoint_file, ckpt_dir, train_fig_dir, dataset_path, test_summary_dir = u.file_paths_init(workspace_path, experiment_name, experiment_group, is_training= True)

    print(cfg_file_path)
    args = configparser.ConfigParser()
    args.read(cfg_file_path)
    print(args.get('Paths', 'LSTM_DATASET_PATH'))

    per_process_gpu_memory_fraction = arguments.gpu
    model = PriorPoseEstimator(experiment_name, experiment_group, args, dataset_path, ckpt_dir, per_process_gpu_memory_fraction, is_training=is_training)

    if is_training:
        model.train(train_epoch, train_steps)
    else:
        x, y = model.get_test_seqs(path_id = 0)
        Rs, ts = model.predict(x)
        np.savez('/home/linfang/Documents/Code/AAE_tracker/ws1/experiments/lstm/poses.npz', ts=ts, Rs=Rs)
        # model.test()


if __name__ == '__main__':
    main()
