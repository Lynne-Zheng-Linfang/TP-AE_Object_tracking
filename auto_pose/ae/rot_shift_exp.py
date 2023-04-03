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
        
def calculate_vsd_error(gt_R, gt_t, est_R, est_t,depth_test, K, model, pose_predictor, vsd_delta, vsd_tau, type='step'):
    W = depth_test.shape[1] 
    H = depth_test.shape[0]
    _, depth_gt, _ = pose_predictor.dataset.render_gt_image(W, H, K, gt_R, gt_t, random_light = False)
    _, depth_est, _ = pose_predictor.dataset.render_gt_image(W, H, K, est_R, est_t, random_light = False)
    error, vis_amnt = pose_error.vsd(est_R, est_t, gt_R, gt_t, depth_gt, depth_est, depth_test, model, K, vsd_delta, vsd_tau, cost_type='step')
    return error, vis_amnt



parser = argparse.ArgumentParser()
parser.add_argument("experiment_name")
parser.add_argument("-o", action='store_true', default=False)
arguments = parser.parse_args()
full_name = arguments.experiment_name.split('/')
experiment_name = full_name.pop()
experiment_group = full_name.pop() if len(full_name) > 0 else ''
OUTPUT = arguments.o

workspace_path = os.environ.get('AE_WORKSPACE_PATH')
if workspace_path == None:
    print('Please define a workspace path:\n')
    print('export AE_WORKSPACE_PATH=/path/to/workspace\n')
    exit(-1)

main_work_folder = os.path.dirname(workspace_path)
cfg_file_path, checkpoint_file, ckpt_dir, dataset_path = utils.file_paths_init(workspace_path, experiment_name, experiment_group)

args = configparser.ConfigParser()
args.read(cfg_file_path)

per_process_gpu_memory_fraction = 0.9
pose_predictor = PosePredictor(experiment_name, dataset_path, args, per_process_gpu_memory_fraction, ckpt_dir)


data = np.load(main_work_folder+'/pose_prediction.npz')
gt_Rs=data['gt_Rs'] 
gt_ts=data['gt_ts'] 
gt_Ks=data['gt_Ks']
est_Rs = data['R']
est_ts = data['t']
vis_amnts = data['vis_amnt']
confidences = data['confidence']
lost_flags = data['lost_flags']
no_rot_crt_Rs = data['no_rot_crt_Rs']
result_file_name = str(data['file_name'])
SCENE_ID = data['scene_id']
instance_id = data['instance_id']

img_info, img_gt, rgb_imgs, depth_imgs, obj_id = utils.load_evaluation_infos_by_scene(args, pose_predictor.dataset, SCENE_ID) 
# img_info, img_gt, rgb_imgs, depth_imgs, obj_id, instance_id = utils.load_evaluation_infos(args, pose_predictor.dataset) 
height, width, channels= rgb_imgs[0].shape

obj_gt_id = -1
vis_amount_index = 0
instance_id_found = 0
for i, obj_gt in enumerate(img_gt[0]):
    if obj_gt['obj_id'] == obj_id:
        if instance_id == instance_id_found:
            obj_gt_id = i
            break
        instance_id_found += 1
    vis_amount_index += 1
if obj_gt_id == -1:
    print('Error: can not find ground truth informations in gt file.')
    exit(-1)
ground_truth_vis_amount = utils.get_gt_vis_amount(args, vis_amount_index, scene_id = SCENE_ID)
model_paths = pose_predictor.dataset.cad_model_paths()
model = inout.load_ply(model_paths[obj_id-1])
error_type = 'vsd_step' # 'vsd', 'adi', 'add', 'cou', 're', 'te'
# VSD parameters
vsd_delta = 15
vsd_tau = 20


eval_num = len(gt_Ks)
acc = np.empty((eval_num, ), dtype = np.float32)
errors = np.empty((eval_num, ), dtype = np.float32)
gt_vis_amnts = np.empty_like(vis_amnts)
vis_amnts_error = np.empty_like(vis_amnts)
img_id = 0
bop_errors =[]
sixd_errors = []
sixd_errors_no_reinit_count_errors = []
bop_errors_no_reinit_count_errors = []
for i in range(eval_num):
    _, gt_vis_amnts[i] = calculate_vsd_error(gt_Rs[i], gt_ts[i], est_Rs[i], est_ts[i], depth_imgs[i], gt_Ks[i], model, pose_predictor, vsd_delta, vsd_tau, type='step')
    if error_type == 'vsd_tlinear':
        errors[i], gt_vis_amnts[i] = calculate_vsd_error(gt_Rs[i], gt_ts[i], est_Rs[i], est_ts[i], depth_imgs[i], gt_Ks[i], model, pose_predictor, vsd_delta, vsd_tau, type='tlinear')
        # vis_amnts_error[i] = np.abs(gt_vis_amnts[i] - vis_amnts[i])
        vis_amnts_error[i] = np.abs(ground_truth_vis_amount[i] - vis_amnts[i])
    elif error_type == 'vsd_step':
        errors[i], gt_vis_amnts[i] = calculate_vsd_error(gt_Rs[i], gt_ts[i], est_Rs[i], est_ts[i], depth_imgs[i], gt_Ks[i], model, pose_predictor, vsd_delta, vsd_tau, type='step')
    elif error_type == 're':
        errors[i] = pose_error.re(est_Rs[i],gt_Rs[i])
    elif error_type == 'add':
        errors[i] = pose_error.add(est_Rs[i], est_ts[i], gt_Rs[i], gt_ts[i], model)
    elif error_type == 'adi':
        errors[i] = pose_error.adi(est_Rs[i], est_ts[i], gt_Rs[i], gt_ts[i], model)
    elif error_type == 'te':
        errors[i] = pose_error.re(est_ts[i],gt_ts[i])
    else:
        print('Error: Wrong error type.')
    
    if gt_vis_amnts[i] >= 0.1:
        sixd_errors.append(errors[i])
        if not lost_flags[i]:
            sixd_errors_no_reinit_count_errors.append(errors[i])

    if ground_truth_vis_amount[i] >= 0.1:
        bop_errors.append(errors[i])
        if not lost_flags[i]:
            bop_errors_no_reinit_count_errors.append(errors[i])
    
if error_type == 'vsd_tlinear' or error_type == 'vsd_step':
    sixd_acc = np.mean(np.array(sixd_errors) < 0.3)
    print('sixd_acc:', sixd_acc, len(sixd_errors), sum(np.array(sixd_errors) < 0.3))
    sixd_acc_no_reinit_count = np.mean(np.array(sixd_errors_no_reinit_count_errors) < 0.3)
    print('sixd_acc_no_re_init_count:', sixd_acc_no_reinit_count, len(sixd_errors_no_reinit_count_errors), sum(np.array(sixd_errors_no_reinit_count_errors) < 0.3))
    bop_acc = np.mean(np.array(bop_errors) < 0.3)
    print('bop_acc:', bop_acc, len(bop_errors), sum(np.array(bop_errors) < 0.3))
    bop_acc_no_reinit_count = np.mean(np.array(bop_errors_no_reinit_count_errors) < 0.3)
    print('bop_acc_no_re_init_count:', bop_acc_no_reinit_count, len(bop_errors_no_reinit_count_errors), sum(np.array(bop_errors_no_reinit_count_errors) < 0.3))
elif error_type == 're':
    acc = np.mean(errors[:i] < 5)
    correct_num = np.sum(errors[:i] < 5)
elif error_type == 'add' or error_type == 'adi':
    acc = np.mean(errors[:i] < 5)
    correct_num = np.sum(errors[:i] < 5)

if OUTPUT:
    sixd_acc_comment = 'sixd_acc: '+ str(sixd_acc)+'\t'+ str(len(sixd_errors))+'   \t' + str( sum(np.array(sixd_errors) < 0.3))+'\n'

    sixd_acc_no_re_init_comment = 'sixd_acc_no_re_init_count: '+ \
        str(sixd_acc_no_reinit_count)+ '\t' + \
        str(len(sixd_errors_no_reinit_count_errors)) + '\t' + \
        str(sum(np.array(sixd_errors_no_reinit_count_errors) < 0.3)) + '\n'

    bop_acc_comment = 'bop_acc: '+ str(bop_acc)+'\t'+ str(len(bop_errors))+'   \t' + str( sum(np.array(bop_errors) < 0.3))+'\n'

    bop_acc_no_re_init_comment = 'bop_acc_no_re_init_count: '+ \
        str(bop_acc_no_reinit_count)+ '\t' + \
        str(len(bop_errors_no_reinit_count_errors)) + '\t' + \
        str(sum(np.array(bop_errors_no_reinit_count_errors) < 0.3)) + '\n'

    file_name = 'performace.txt'
    with open(os.path.join('/home/linfang/Documents/Code/AAE_tracker/tests/performance',file_name), 'a+') as f:
        f.write('### With Rotation Shift Correction ### \n')
        f.write(sixd_acc_comment)
        f.write(sixd_acc_no_re_init_comment)
        f.write(bop_acc_comment)
        f.write(bop_acc_no_re_init_comment)

    with open(result_file_name, 'a+') as f:
        # Sixd number
        # Sixd correct number
        # Sixd no reinit-count num
        # Sixd no reinit count correct num
        # BOP number
        # BOP correct number
        # BOP no reinit-count num
        # BOP no reinit count correct num
        out_info = str(len(sixd_errors))+','+ str(sum(np.array(sixd_errors) < 0.3)) +',' + str(len(sixd_errors_no_reinit_count_errors)) + ',' + str(sum(np.array(sixd_errors_no_reinit_count_errors) < 0.3)) + ',' + str(len(bop_errors)) + ',' + str(sum(np.array(bop_errors) < 0.3)) + ',' + str(len(bop_errors_no_reinit_count_errors)) + ',' + str(sum(np.array(bop_errors_no_reinit_count_errors) < 0.3)) + ','  
        f.write(out_info)

print('No rotation correction:')
est_Rs = no_rot_crt_Rs
sixd_errors = []
bop_errors = []
sixd_errors_no_reinit_count_errors = []
bop_errors_no_reinit_count_errors = []
for i in range(eval_num):
    _, gt_vis_amnts[i] = calculate_vsd_error(gt_Rs[i], gt_ts[i], est_Rs[i], est_ts[i], depth_imgs[i], gt_Ks[i], model, pose_predictor, vsd_delta, vsd_tau, type='step')
    if error_type == 'vsd_tlinear':
        errors[i], gt_vis_amnts[i] = calculate_vsd_error(gt_Rs[i], gt_ts[i], est_Rs[i], est_ts[i], depth_imgs[i], gt_Ks[i], model, pose_predictor, vsd_delta, vsd_tau, type='tlinear')
        # vis_amnts_error[i] = np.abs(gt_vis_amnts[i] - vis_amnts[i])
        vis_amnts_error[i] = np.abs(ground_truth_vis_amount[i] - vis_amnts[i])
    elif error_type == 'vsd_step':
        errors[i], gt_vis_amnts[i] = calculate_vsd_error(gt_Rs[i], gt_ts[i], est_Rs[i], est_ts[i], depth_imgs[i], gt_Ks[i], model, pose_predictor, vsd_delta, vsd_tau, type='step')
    elif error_type == 're':
        errors[i] = pose_error.re(est_Rs[i],gt_Rs[i])
    elif error_type == 'add':
        errors[i] = pose_error.add(est_Rs[i], est_ts[i], gt_Rs[i], gt_ts[i], model)
    elif error_type == 'adi':
        errors[i] = pose_error.adi(est_Rs[i], est_ts[i], gt_Rs[i], gt_ts[i], model)
    elif error_type == 'te':
        errors[i] = pose_error.re(est_ts[i],gt_ts[i])
    else:
        print('Error: Wrong error type.')
    
    if gt_vis_amnts[i] >= 0.1:
        sixd_errors.append(errors[i])
        if not lost_flags[i]:
            sixd_errors_no_reinit_count_errors.append(errors[i])

    if ground_truth_vis_amount[i] >= 0.1:
        bop_errors.append(errors[i])
        if not lost_flags[i]:
            bop_errors_no_reinit_count_errors.append(errors[i])
    
if error_type == 'vsd_tlinear' or error_type == 'vsd_step':
    sixd_acc = np.mean(np.array(sixd_errors) < 0.3)
    print('sixd_acc:', sixd_acc, len(sixd_errors), sum(np.array(sixd_errors) < 0.3))
    sixd_acc_no_reinit_count = np.mean(np.array(sixd_errors_no_reinit_count_errors) < 0.3)
    print('sixd_acc_no_re_init_count:', sixd_acc_no_reinit_count, len(sixd_errors_no_reinit_count_errors), sum(np.array(sixd_errors_no_reinit_count_errors) < 0.3))
    bop_acc = np.mean(np.array(bop_errors) < 0.3)
    print('bop_acc:', bop_acc, len(bop_errors), sum(np.array(bop_errors) < 0.3))
    bop_acc_no_reinit_count = np.mean(np.array(bop_errors_no_reinit_count_errors) < 0.3)
    print('bop_acc_no_re_init_count:', bop_acc_no_reinit_count, len(bop_errors_no_reinit_count_errors), sum(np.array(bop_errors_no_reinit_count_errors) < 0.3))
    # acc = np.mean(errors[:i] < 0.3)
    # correct_num = np.sum(errors[:i] < 0.3)
elif error_type == 're':
    acc = np.mean(errors[:i] < 5)
    correct_num = np.sum(errors[:i] < 5)
elif error_type == 'add' or error_type == 'adi':
    acc = np.mean(errors[:i] < 5)
    correct_num = np.sum(errors[:i] < 5)
print('finished')

if OUTPUT:
    # sixd_acc_comment = 'sixd_acc: '+ str(sixd_acc)+'\t'+ str(len(sixd_errors))+'   \t' + str( sum(np.array(sixd_errors) < 0.3))+'\n'

    # sixd_acc_no_re_init_comment = 'sixd_acc_no_re_init_count: '+ \
    #     str(sixd_acc_no_reinit_count)+ '\t' + \
    #     str(len(sixd_errors_no_reinit_count_errors)) + '\t' + \
    #     str(sum(np.array(sixd_errors_no_reinit_count_errors) < 0.3)) + '\n'

    # bop_acc_comment = 'bop_acc: '+ str(bop_acc)+'\t'+ str(len(bop_errors))+'   \t' + str( sum(np.array(bop_errors) < 0.3))+'\n'

    # bop_acc_no_re_init_comment = 'bop_acc_no_re_init_count: '+ \
    #     str(bop_acc_no_reinit_count)+ '\t' + \
    #     str(len(bop_errors_no_reinit_count_errors)) + '\t' + \
    #     str(sum(np.array(bop_errors_no_reinit_count_errors) < 0.3)) + '\n'

    # file_name = 'performace.txt'
    # with open(os.path.join('/home/linfang/Documents/Code/AAE_tracker/tests/performance',file_name), 'a+') as f:
    #     f.write('### No Rotation Shift Correction ### \n')
    #     f.write(sixd_acc_comment)
    #     f.write(sixd_acc_no_re_init_comment)
    #     f.write(bop_acc_comment)
    #     f.write(bop_acc_no_re_init_comment)

    with open(result_file_name, 'a+') as f:
        # results of no rot correction
        # Sixd correct number 
        # Sixd no reinit count correct num
        # BOP correct number
        # BOP no reinit count correct num
        out_info = str(sum(np.array(sixd_errors) < 0.3)) +',' + str(sum(np.array(sixd_errors_no_reinit_count_errors) < 0.3)) + ',' + str(sum(np.array(bop_errors) < 0.3)) + ',' + str(sum(np.array(bop_errors_no_reinit_count_errors) < 0.3)) + ', \n'
        f.write(out_info)
