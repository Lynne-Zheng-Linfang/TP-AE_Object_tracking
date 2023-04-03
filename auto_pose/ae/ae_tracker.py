# -*- coding: utf-8 -*-

import cv2
import tensorflow as tf
import numpy as np
import glob
import os
import time
import argparse
import configparser
import hashlib
import random
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import sys
from pytless import inout
from pysixd_stuff import transform
import utils
import ae_factory as factory
import im_process
from ae_pose_predictor import PosePredictor
from sixd_toolkit.pysixd import pose_error
from prior_pose_estimator import PriorPoseEstimator

parser = argparse.ArgumentParser()
parser.add_argument("experiment_name")
parser.add_argument("-lstm", action='store_true', default=False)
parser.add_argument("-GRU", action='store_true', default=False)
parser.add_argument("-reinit", action='store_true', default=False)
parser.add_argument("-cosy", action='store_true', default=False)
parser.add_argument("-v", type=float, default = 0.5, help='re-initial visible amount')
parser.add_argument("-o", action='store_true', default=False)
parser.add_argument("-len", type=int, default = 15, help='re-initial visible amount')
parser.add_argument("-inst", type=int,  help='instance_id')
parser.add_argument("-scene", type=int, help='scene_id')
parser.add_argument("-r_conf", action='store_true', default=False)

arguments = parser.parse_args()
full_name = arguments.experiment_name.split('/')
experiment_name = full_name.pop()
experiment_group = full_name.pop() if len(full_name) > 0 else ''
USE_LSTM = arguments.lstm
no_re_init = not arguments.reinit
RE_INIT_VIS_AMNT = arguments.v
OUTPUT = arguments.o
seq_length = arguments.len
SCENE_ID = arguments.scene 
INSTANCE_ID = arguments.inst 
USE_R_CONFI = arguments.r_conf
USE_GRU = arguments.GRU
USE_COSY_INIT = arguments.cosy

workspace_path = os.environ.get('AE_WORKSPACE_PATH')
if workspace_path == None:
    print('Please define a workspace path:\n')
    print('export AE_WORKSPACE_PATH=/path/to/workspace\n')
    exit(-1)

main_work_folder = os.path.dirname(workspace_path)
cfg_file_path, checkpoint_file, ckpt_dir, dataset_path = utils.file_paths_init(workspace_path, experiment_name, experiment_group)

args = configparser.ConfigParser()
args.read(cfg_file_path)

print('#'*20+' INFO '+'#'*20)
print(experiment_group+'/'+experiment_name)
print('Use LSTM:', USE_LSTM)
print('Re-init:', arguments.reinit)
print('Re-init vis amount:', RE_INIT_VIS_AMNT)
print('Scene ID:', SCENE_ID)
print('Instance ID:', INSTANCE_ID)
print('#'*46)

per_process_gpu_memory_fraction = 0.6
pose_predictor = PosePredictor(experiment_name, dataset_path, args, per_process_gpu_memory_fraction, ckpt_dir)

if USE_LSTM:
    # TODO: exp_name and group name should be get from the cfg file of pose predictor
    lstm_exp_group = 'GRU' if USE_GRU else 'lstm' 
    lstm_exp_name = 'GRU_01' if USE_GRU else 'lstm_5_low_upper_bound'
    lstm_cfg_file_path, lstm_checkpoint_file, lstm_ckpt_dir, lstm_dataset_path = utils.file_paths_init(workspace_path, lstm_exp_name, lstm_exp_group)

    lstm_args = configparser.ConfigParser()
    lstm_args.read(lstm_cfg_file_path)
    per_process_gpu_memory_fraction = 0.2
    prior_pose_generator = PriorPoseEstimator(lstm_exp_name, lstm_exp_group, lstm_args, lstm_dataset_path, lstm_ckpt_dir, per_process_gpu_memory_fraction, is_training=False)

img_info, img_gt, rgb_imgs, depth_imgs, obj_id = utils.load_evaluation_infos_by_scene(args, pose_predictor.dataset, SCENE_ID) 
height, width, channels= rgb_imgs[0].shape
instance_id = INSTANCE_ID

reinit_poses = img_gt
if USE_COSY_INIT:
    reinit_file = '/home/linfang/Documents/Code/cosypose/local_data/results/tless-siso-n_views=1--684390594/poses/{}.yml'
    reinit_poses = inout.load_gt(reinit_file.format(str(SCENE_ID).zfill(2)))

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
gt_Rs = np.empty((eval_num,)+(3,3), dtype=np.float)
no_rot_crt_Rs = np.empty((eval_num,)+(3,3), dtype=np.float)
dRs = np.empty((eval_num,)+(3,3), dtype=np.float)
out_t = np.empty((eval_num,)+(3,1), dtype=np.float)
gt_ts = np.empty((eval_num,)+(3,1), dtype=np.float)
dts = np.empty((eval_num,)+(3,1), dtype=np.float)
vis_amnt = np.empty((eval_num,), dtype=np.float)
confidences = np.empty((eval_num,), dtype=np.float)
gt_Ks = np.empty((eval_num,)+(3,3), dtype=np.float)
init_flags = np.empty((eval_num,), dtype=bool)
if USE_LSTM:
    result_poses = np.tile(np.eye(4), (eval_num,1,1))

EXTENSION_IMG_NUM = 26
average_num = seq_length
re_init_times = 0
pred_step = 0
init_flag = False
next_init_flag = False
for i in range(eval_num):
    if i == 1:
        start_time = time.time()
    K = pose_predictor.dataset.get_intrinsic_parameters(img_info, i)
    if (i == 0) or init_flag:
        pred_R, pred_t= pose_predictor.dataset.get_eval_target_pose(reinit_poses, i, obj_id, instance_id = instance_id)
        pred_step = 0
    else:
        pred_R = out_R[i-1]
        pred_t = out_t[i-1]
        if pred_step >= 2:
            if not USE_LSTM:
                dRs[pred_step-2] = np.matmul(out_R[i-2].transpose(), out_R[i-1])
                dts[pred_step-2] = out_t[i-1] - out_t[i-2]
                pred_R = utils.pred_next_rotation(dRs, pred_R, pred_step, average_num = average_num)
                pred_t = utils.pred_next_translation(dts, pred_t, pred_step, average_num = average_num)
                pred_t = im_process.keep_t_within_image(pred_t, K, width, height)
            else:
                start = max(0, pred_step - seq_length) 
                pred_R, pred_t = prior_pose_generator.predict(result_poses[start:pred_step])
                pred_t = pred_t.reshape((3,1))
                pred_t = im_process.keep_t_within_image(pred_t, K, width, height)
    if pred_t[2] < 200:
        pred_t[2] =200

    eval_x, _, _= pose_predictor.dataset.train_images_preprocess(
        np.expand_dims(rgb_imgs[i], axis = 0), 
        np.expand_dims(depth_imgs[i], axis=0), 
        np.expand_dims(pred_R, axis = 0),
        np.expand_dims(pred_t, axis = 0), 
        np.expand_dims(K, axis=0), 
        rotation_target=False, 
        return_no_rot_patch= True,
        broadcast=True)

    pred_img = pose_predictor.get_render_image_patch(pred_R, pred_t, K)
    eval_x = np.concatenate((eval_x, pred_img), axis = 0)
    eval_x_no_rot = eval_x.copy()

    if eval_x.ndim == 3:
        eval_x = np.expand_dims(eval_x, 0)

    vis_amounts, delta_ts, pred_Rs, similarity_scores, confidence = pose_predictor.predict(eval_x, eval_x_no_rot)

    vis_amnt[i] = vis_amounts[0]
    confidences[i] = confidence[0]
    R_pass = (confidence[0]> 0.6 ) if USE_R_CONFI else True 
    if no_re_init:
        if (vis_amounts[0] >= 0.1): 
            pred_t = pred_t + delta_ts[0].reshape((3,1))*pose_predictor.dataset.max_delta_t_shift
 
            similarity_score = vis_amounts[0]*similarity_scores[0] + (1-vis_amounts[0])*similarity_scores[1]

            top_k = 1
            unsorted_max_idcs = np.argpartition(-similarity_score, top_k)[:top_k]
            idcs = unsorted_max_idcs[np.argsort(-similarity_score[unsorted_max_idcs])]
            pred_R = pose_predictor.dataset.viewsphere_for_embedding[idcs]
    else:
        if init_flag or (i == 0):
            if (vis_amounts[0] >= RE_INIT_VIS_AMNT) and R_pass:
                re_init_times += 1
                next_init_flag = False
            else:
                pred_step = 0
                next_init_flag = True
        else:
            if (vis_amounts[0] < 0.1) or (not R_pass):
                # Do re-initialization 
                init_flag = True
                pred_R, pred_t= pose_predictor.dataset.get_eval_target_pose(reinit_poses,i,obj_id, instance_id = instance_id)

                pred_img = pose_predictor.get_render_image_patch(pred_R, pred_t, K)
                eval_x = np.concatenate((eval_x, pred_img), axis = 0)
                eval_x_no_rot = eval_x.copy()

                if eval_x.ndim == 3:
                    eval_x = np.expand_dims(eval_x, 0)

                vis_amounts, delta_ts, pred_Rs, similarity_scores, confidence = pose_predictor.predict(eval_x, eval_x_no_rot)

                vis_amnt[i] = vis_amounts[0]
                confidences[i] = confidence[0]
                pred_step = 0
                R_pass = (confidence[0]> 0.6 ) if USE_R_CONFI else True 
                if (vis_amounts[0] < 0.1) or (not R_pass):
                    next_init_flag = True
                else:
                    next_init_flag = False 
            else:
                next_init_flag = False

        pred_t = pred_t + delta_ts[0].reshape((3,1))*pose_predictor.dataset.max_delta_t_shift
 
        similarity_score = vis_amounts[0]*similarity_scores[0] + (1-vis_amounts[0])*similarity_scores[1]

        top_k = 1
        unsorted_max_idcs = np.argpartition(-similarity_score, top_k)[:top_k]
        idcs = unsorted_max_idcs[np.argsort(-similarity_score[unsorted_max_idcs])]
        pred_R = pose_predictor.dataset.viewsphere_for_embedding[idcs]
    no_rot_crt_Rs[i] = pred_R
    pred_R = pose_predictor.correct_rotation_shift(pred_R, pred_t)
    out_t[i] = pred_t
    out_R[i] = pred_R
    gt_Rs[i], gt_ts[i] = pose_predictor.dataset.get_eval_target_pose(img_gt,i,obj_id, instance_id=instance_id)
    gt_Ks[i] = K
    init_flags[i] = init_flag 
    if USE_LSTM:
        result_poses[pred_step,:3,:3] = pred_R
        result_poses[pred_step,:3, 3] = pred_t.reshape(3,)
    pred_step += 1
    init_flag = init_flag if no_re_init else next_init_flag
runtime = (time.time() - start_time)/(eval_num)
print(runtime)
print(re_init_times)
result_file_name = 'no_reinit_' if no_re_init else 'reinit_'
result_file_name += (str(int(RE_INIT_VIS_AMNT*100)) + '_')
if USE_LSTM:
    result_file_name += 'GRU_' if USE_GRU else 'lstm_'
else:
    result_file_name += 'const_vel_'
result_file_name += str(seq_length)
result_file_name += ('_'+str(USE_R_CONFI))
if USE_COSY_INIT:
    result_file_name += ('_cosy')
result_file_name += '.txt'
result_file_name = os.path.join(workspace_path, 'est_results', result_file_name)
np.savez(main_work_folder+'/pose_prediction.npz', R=out_R, t= out_t, vis_amnt=vis_amnt, confidence = confidences, gt_ts = gt_ts, gt_Rs = gt_Rs, gt_Ks = gt_Ks, lost_flags = init_flags, no_rot_crt_Rs = no_rot_crt_Rs, file_name = result_file_name, scene_id = SCENE_ID, instance_id = INSTANCE_ID)
print('finished')


if OUTPUT:
    # headline = '#'*20+' INFO '+'#'*20 +'\n'
    # group_info = experiment_group+'/'+experiment_name + '\n'
    # lstm_info = 'Use LSTM: '+ str(USE_LSTM) + '\n'
    # reinit_info = 'Re-init: '+ str(arguments.reinit) + '\n'
    # vis_info = 'Re-init vis amount: '+ str(RE_INIT_VIS_AMNT) + '\n'
    # scene_info = 'Scene ID: '+ str(SCENE_ID) + '\n'
    # ins_info = 'Instance ID: '+ str(INSTANCE_ID) + '\n'
    # runtime_info = 'runtime: ' + str(runtime) + '\n'
    # re_init_times_info = 'reinit times: ' + str(re_init_times) + '\n'
    # endline = '#'*46 + '\n'
    # file_name = 'performace.txt'
    # with open(os.path.join('/home/linfang/Documents/Code/AAE_tracker/tests/performance',file_name), 'a+') as f:
    #     f.write(headline)
    #     f.write(group_info)
    #     f.write(lstm_info)
    #     f.write(reinit_info)
    #     f.write(vis_info)
    #     f.write(scene_info)
    #     f.write(ins_info)
    #     f.write(runtime_info)
    #     f.write(re_init_times_info)
    #     f.write(endline)

    if not os.path.exists(result_file_name):
        # Sixd number
        # Sixd correct number
        # Sixd no reinit-count num
        # Sixd no reinit count correct num
        # BOP number
        # BOP correct number
        # BOP no reinit-count num
        # BOP no reinit count correct num
        # results of no rot correction
        # Sixd correct number 
        # Sixd no reinit count correct num
        # BOP correct number
        # BOP no reinit count correct num
        with open(result_file_name, 'w+') as f:
            f.write('Obj_ID, Scene_ID, Instance_ID, Re-init Count, Runtime, Total Image Num, Sixd Valid Num, Sixd Correct Num, Sixd Valid Num No-reinit Count, Sixd Correct Num No-reinit Count, BOP Valid Num, BOP Correct Num, BOP Valid Num No-reinit Count, BOP Correct Num No-reinit Count, Sixd No Rot Correct Num, Sixd No Rot Correct Num No-reinit Count, BOP No Rot Correct Num, BOP No Rot Correct Num No-reinit Count, \n')
    with open(result_file_name, 'a+') as f:
        out_info = str(obj_id)+','+ str(SCENE_ID) +',' + str(INSTANCE_ID) + ',' + str(re_init_times) + ',' + str(runtime) + ',' + str(eval_num) + ',' 
        f.write(out_info)