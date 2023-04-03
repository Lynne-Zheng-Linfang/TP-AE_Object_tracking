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

from auto_pose.ae import factory, utils
from auto_pose.ae.combiner_module import CombinerModule
from auto_pose.ae.ae_pose_predictor import PosePredictor
# import faulthandler
# faulthandler.enable()

workspace_path = os.environ.get('AE_WORKSPACE_PATH')

if workspace_path is None:
    print('Please define a workspace path:\n')
    print('export AE_WORKSPACE_PATH=/path/to/workspace\n')
    exit(-1)
main_work_folder = os.path.dirname(workspace_path)

parser = argparse.ArgumentParser()
parser.add_argument("experiment_name")
parser.add_argument("-d", action='store_true', default=False)
parser.add_argument('-dynamic', action="store_true", default=False, help='generate train data online') 
arguments = parser.parse_args()

full_name = arguments.experiment_name.split('/')

experiment_name = full_name.pop()
experiment_group = full_name.pop() if len(full_name) > 0 else ''

debug_mode = arguments.d
dynamic_mode = arguments.dynamic

cfg_file_path, checkpoint_file, ckpt_dir, train_fig_dir, dataset_path, test_summary_dir = utils.file_paths_init(workspace_path, experiment_name, experiment_group, is_training= True)

args = configparser.ConfigParser()
args.read(cfg_file_path)

per_process_gpu_memory_fraction = 0.4
predictor = PosePredictor(experiment_name, dataset_path, args, per_process_gpu_memory_fraction, ckpt_dir)

combiner = CombinerModule(experiment_name, dataset_path, args,per_process_gpu_memory_fraction, ckpt_dir)

img_info, img_gt, rgb_imgs, depth_imgs, obj_id = utils.load_evaluation_infos(args, predictor.dataset) 
height, width, channels= rgb_imgs[0].shape

obj_gt_id = -1
for i, obj_gt in enumerate(img_gt[0]):
    if obj_gt['obj_id'] == obj_id:
         obj_gt_id = i
         break
if obj_gt_id == -1:
    print('Error: can not find ground truth informations in gt file.')
    exit(-1)

eval_num = len(rgb_imgs)
out_R = np.empty((eval_num,)+(3,3), dtype=np.float)
dRs = np.empty((eval_num,)+(3,3), dtype=np.float)
out_t = np.empty((eval_num,)+(3,1), dtype=np.float)
gt_t = np.empty((eval_num,)+(3,1), dtype=np.float)
dts = np.empty((eval_num,)+(3,1), dtype=np.float)
vis_amnt = np.empty((eval_num,), dtype=np.float)

for i in range(eval_num):
    K = predictor.dataset.get_intrinsic_parameters(img_info, i)
    if i == 0:
        pred_R, pred_t= predictor.dataset.get_eval_target_pose(img_gt, i, obj_id)
    else:
        pred_R = out_R[i-1]
        pred_t = out_t[i-1]
        if i >= 2:
            dRs[i-2] = np.matmul(out_R[i-2].transpose(), out_R[i-1])
            dts[i-2] = out_t[i-2] - out_t[i-1]
            pred_R = utils.pred_next_rotation(dRs, pred_R, i, average_num = 0)
            pred_t = utils.pred_next_translation(dts, pred_t, i, average_num = 0)
    for _ in range(1):
        eval_x, _ = predictor.dataset.train_images_preprocess(rgb_imgs[i], depth_imgs[i], pred_R, pred_t, K)

        pred_image = predictor.dataset.generate_rotation_image_patch(pred_R, pred_R)

        rgb_x = eval_x.copy()
        rgb_x[:,:,3:6] = 0

        eval_x = np.stack((rgb_x, eval_x, pred_image), axis = 0)

        if eval_x.ndim == 3:
            eval_x = np.expand_dims(eval_x, 0)
        
        visable_amount, delta_t, code_x = predictor.get_prediction_result_for_combiner(eval_x)

        if vis_amnt[i] >= 0.1:
            pred_t = pred_t + delta_t.reshape((3,1))*predictor.dataset.max_delta_t_shift
        code = combiner.predict(code_x)
        pred_R = predictor.rotation_from_code(code)
        out_t[i] = pred_t
        out_R[i] = pred_R
        _, gt_t[i]= predictor.dataset.get_eval_target_pose(img_gt, i, obj_id)
np.savez(main_work_folder+'/pose_prediction.npz', R=out_R, t= gt_t)
print('finished')
