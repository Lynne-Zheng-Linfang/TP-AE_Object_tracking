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
from sixd_challenge_toolkit.pysixd import inout as sixd_inout
# from eval import icp_utils 

def load_previous_results(path):
    if not os.path.exists(path):
        res = {}
        res['ests'] = []
    else:
        res = sixd_inout.load_results_sixd17(path)
    return res

def store_result(path, score, R, t, runtime):
    result = {}
    result['score'] = score
    result['R'] = R
    result['t'] = t
    result['runtime'] = runtime
    res = load_previous_results(path)
    res['ests'].append(result)
    sixd_inout.save_results_sixd17(path, res, runtime)

def check_init_status(t):
    return (t != np.zeros(t.shape)).any()

parser = argparse.ArgumentParser()
parser.add_argument("experiment_name")
parser.add_argument("-lstm", action='store_true', default=False)
parser.add_argument("-GRU", action='store_true', default=False)
parser.add_argument("-reinit", action='store_true', default=False)
parser.add_argument("-cosy", action='store_true', default=False)
parser.add_argument("-v", type=float, default = 0.5, help='re-initial visible amount')
parser.add_argument("-o", action='store_true', default=False)
parser.add_argument("-len", type=int, default = 15, help='re-initial visible amount')
parser.add_argument("-noise", action='store_true', default = False)
parser.add_argument("-pure", action='store_true', default = False)
parser.add_argument("-dist", action='store_true', default = False)
parser.add_argument("-inst", type=int,  help='instance_id')
parser.add_argument("-scene", type=int, help='scene_id')
parser.add_argument("-r_conf", action='store_true', default=False)
parser.add_argument("-rtrack", action='store_true', default=False)
parser.add_argument("-YCB", action='store_true', default=False)
parser.add_argument("-refine", action='store_true', default=False)
parser.add_argument("-ronly", action='store_true', default=False)
parser.add_argument("-tonly", action='store_true', default=False)
parser.add_argument("-icp", action='store_true', default=False)
parser.add_argument("-res_dir", type=str, default='PriAAE_tless_primesens', help='folder name for storing results')
parser.add_argument("-posecnn", action='store_true', default=False)
parser.add_argument("-ori", action='store_true', default=False)

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
RES_DIR = arguments.res_dir 
USE_NOISE_VERSION = arguments.noise
DISTIBUTION = arguments.dist
PURE = arguments.pure
ROTATION_TRACK_ONLY = arguments.rtrack
REFINE_MODE = arguments.refine
YCB = arguments.YCB
R_ONLY = arguments.ronly
T_ONLY = arguments.tonly
ICP = arguments.icp
POSECNN = arguments.posecnn
ORI = arguments.ori

workspace_path = os.environ.get('AE_WORKSPACE_PATH')
if workspace_path == None:
    print('Please define a workspace path:\n')
    print('export AE_WORKSPACE_PATH=/path/to/workspace\n')
    exit(-1)

mid_dir_name = ('NewVisReinit-'+str(int(RE_INIT_VIS_AMNT*100)).zfill(3)) if arguments.reinit else 'Noreinit'
if USE_LSTM and USE_GRU:
    mid_dir_name += ('-GRU-'+str(seq_length).zfill(2))
if USE_LSTM and not USE_GRU:
    mid_dir_name += ('-LSTM'+str(seq_length).zfill(2))
if not USE_LSTM:
    mid_dir_name += '-constV'
if USE_COSY_INIT:
    mid_dir_name += '-cosy' 
elif POSECNN:
    mid_dir_name += '-posecnn' 
else:
    mid_dir_name += '-gt'

# mid_dir_name += '-cosy' if USE_COSY_INIT else '-gt'
mid_dir_name += '-rpass' if USE_R_CONFI else '-no-rpass'
mid_dir_name += '-dist' if DISTIBUTION else '-single'
mid_dir_name += '-noise' if USE_NOISE_VERSION else '-clean'
if PURE:
    mid_dir_name += '-pure' 
if ROTATION_TRACK_ONLY:
    mid_dir_name += '-rtrack' 
if REFINE_MODE:
    mid_dir_name += '-refine' 
if R_ONLY:
    mid_dir_name += '-ronly' 
if T_ONLY:
    mid_dir_name += '-tonly' 
if ICP:
    mid_dir_name += '-icp' 
if ORI:
    mid_dir_name += '-ori' 

mid_dir_name += '_real_syn_check'
dataset_name = 'tless' if not YCB else 'YCB_V'
vis_result_dir_path = os.path.join(workspace_path,'test_results', dataset_name, mid_dir_name,'vis_res', RES_DIR, str(SCENE_ID).zfill(2))

# if not os.path.exists(vis_result_dir_path):
#     os.makedirs(vis_result_dir_path)

vis_R_result_dir_path = os.path.join(workspace_path, 'test_results',  dataset_name, mid_dir_name, 'vis_r_res', RES_DIR, str(SCENE_ID).zfill(2))
if not os.path.exists(vis_R_result_dir_path):
    os.makedirs(vis_R_result_dir_path)

# vis_result_no_crt_dir_path = os.path.join(workspace_path,'test_results',  dataset_name, mid_dir_name,'vis_res_no_crt', RES_DIR, str(SCENE_ID).zfill(2))
# if not os.path.exists(vis_result_no_crt_dir_path):
#     os.makedirs(vis_result_no_crt_dir_path)

vis_R_result_no_crt_dir_path = os.path.join(workspace_path, 'test_results',  dataset_name, mid_dir_name, 'vis_r_res_no_crt', RES_DIR, str(SCENE_ID).zfill(2))
if not os.path.exists(vis_R_result_no_crt_dir_path):
    os.makedirs(vis_R_result_no_crt_dir_path)

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
print('Noise Version:', USE_NOISE_VERSION)
print('CosyPose:', USE_COSY_INIT)
print('R_tracking:', ROTATION_TRACK_ONLY)
print('#'*46)

per_process_gpu_memory_fraction = 0.7
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

img_info, img_gt, rgb_imgs, depth_imgs, obj_id = utils.load_evaluation_infos_by_scene(args, pose_predictor.dataset, SCENE_ID, ycb_v = YCB) 
height, width, channels= rgb_imgs[0].shape
instance_id = INSTANCE_ID

start_id = list(img_info.keys())[0]

reinit_poses = img_gt
if (POSECNN or USE_COSY_INIT) and (not ROTATION_TRACK_ONLY):
    if YCB and USE_COSY_INIT:
        cosy_result_base = '/home/linfang/Documents/Code/cosypose/local_data/results/bop-synt+real--150755/dataset=ycbv/poses'
    elif POSECNN:
        cosy_result_base = '/home/linfang/Documents/Code/cosypose/local_data/results/ycbv-n_views=5--8073381555/poses'
        # cosy_result_base = '/home/linfang/Documents/Code/cosypose/local_data/results/bop-synt+real--150755/dataset=ycbv/poses'
    else:
        if ORI:
            cosy_result_base = '/home/linfang/Documents/Code/cosypose/local_data/results/tless-siso-n_views=1--684390594/poses'
            print('come here')
        else:
            cosy_result_base = '/home/linfang/Documents/Code/cosypose/local_data/results/bop-synt+real--5896/dataset=tless/poses'
    reinit_file = cosy_result_base
    if DISTIBUTION:
        reinit_file += '_dist'
    if USE_NOISE_VERSION:
        reinit_file += '_noise'
    if PURE:
        reinit_file += '_no_gt'
    reinit_file = os.path.join(reinit_file, '{}.yml')
    reinit_poses = inout.load_gt(reinit_file.format(str(SCENE_ID).zfill(2)))

obj_gt_id = -1
for i, obj_gt in enumerate(img_gt[start_id]):
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

average_num = seq_length
re_init_times = 0
pred_step = 0
init_flag = False
next_init_flag = False

model_path = os.path.join(os.path.dirname(args.get('Evaluation', 'EVALUATION_MODEL_PATH')), 'obj_'+str(obj_id).zfill(2)+'.ply')
print(model_path)
# model_path = '/home/linfang/Documents/Dataset/T_LESS/t-less_kinect/models_cad/obj_01.ply'
# icp_renderer = icp_utils.SynRenderer(model_path, pose_predictor.dataset.target_obj_diameter, pose_predictor.dataset.bbox_enlarge_level-0.2) if ICP else None
#######Lots of lazy_properties need to be initialized during warming######
warming = True
if warming:
    i = 0
    K = pose_predictor.dataset.get_intrinsic_parameters(img_info, start_id)
    pred_R, pred_t= pose_predictor.dataset.get_eval_target_pose(img_gt, start_id, obj_id, instance_id = instance_id)
    if USE_LSTM:
        start = max(0, pred_step - seq_length) 
        _, _ = prior_pose_generator.predict(result_poses[start:2])
        pred_t = pred_t.reshape((3,1))
        _ = im_process.keep_t_within_image(pred_t, K, width, height)
    vis_amounts, delta_ts, pred_Rs, similarity_scores, confidence = pose_predictor.new_predict(rgb_imgs[i], depth_imgs[i], pred_R, pred_t, K, no_rot_depth = False)
    pred_t = pred_t + delta_ts[0].reshape((3,1))*pose_predictor.dataset.max_delta_t_shift
    similarity_score = vis_amounts[0]*similarity_scores[0] + (1-vis_amounts[0])*similarity_scores[1]
    top_k = 1
    unsorted_max_idcs = np.argpartition(-similarity_score, top_k)[:top_k]
    idcs = unsorted_max_idcs[np.argsort(-similarity_score[unsorted_max_idcs])]
    pred_R = pose_predictor.dataset.viewsphere_for_embedding[idcs]
    pred_R = pose_predictor.correct_rotation_shift(pred_R, pred_t)
################################################################################
average_num = seq_length
re_init_times = 0
pred_step = 0
init_flag = False
next_init_flag = False
init_failed = False
init_ok = False

for img_id in img_info.keys():
    i = (img_id - 1) if YCB else img_id
    K = pose_predictor.dataset.get_intrinsic_parameters(img_info, img_id)

    start_time = time.time()
    if (i == 0) or init_flag:
        pred_R, pred_t= pose_predictor.dataset.get_eval_target_pose(reinit_poses, img_id, obj_id, instance_id = instance_id)
        init_ok = check_init_status(pred_t)
        if init_ok:
            pred_step = 0
        else:
            if i > 0:
                pred_R, pred_t = out_R[i-1], out_t[i-1]
            else:
                pred_t = pose_predictor.init_pred_t(pred_t.shape)
    else:
        pred_R = out_R[i-1]
        pred_t = out_t[i-1]

    if pred_step >= 2:
        if not USE_LSTM:
            dRs[pred_step-2] = np.matmul(out_R[i-2].transpose(), out_R[i-1])
            dts[pred_step-2] = out_t[i-1] - out_t[i-2]
            pred_R = utils.pred_next_rotation(dRs, pred_R, pred_step, average_num = average_num)
            pred_t = utils.pred_next_translation(dts, pred_t, pred_step, average_num = average_num)
        else:
            start = max(0, pred_step - seq_length) 
            pred_R, pred_t = prior_pose_generator.predict(result_poses[start:pred_step])
            pred_t = pred_t.reshape((3,1))
    if pred_t[2] < 200:
        pred_t[2] =200
    pred_t = im_process.keep_t_within_image(pred_t, K, width, height)
    vis_amounts, delta_ts, pred_Rs, similarity_scores, confidence = pose_predictor.new_predict(rgb_imgs[i], depth_imgs[i], pred_R, pred_t, K, no_rot_depth = False)
    vis_amnt[i] = vis_amounts[0]
    confidences[i] = confidence[0]
    R_pass = (confidence[0]> 0.6 ) if USE_R_CONFI else True 
    if no_re_init:
        if (vis_amounts[0] >= RE_INIT_VIS_AMNT): 
            pred_t = pred_t + delta_ts[0].reshape((3,1))*pose_predictor.dataset.max_delta_t_shift
 
            if False: 
                similarity_score = similarity_scores[0]
            else:
                similarity_score = vis_amounts[0]*similarity_scores[0] + (1-vis_amounts[0])*similarity_scores[1]

            top_k = 1
            unsorted_max_idcs = np.argpartition(-similarity_score, top_k)[:top_k]
            idcs = unsorted_max_idcs[np.argsort(-similarity_score[unsorted_max_idcs])]
            pred_R = pose_predictor.dataset.viewsphere_for_embedding[idcs]
    else:
        if init_flag or (i == 0):
            re_init_times += 1
            if (vis_amounts[0] >= RE_INIT_VIS_AMNT) and R_pass:
                next_init_flag = False
            else:
                pred_step = 0
                next_init_flag = True
        else:
            if (vis_amounts[0] < RE_INIT_VIS_AMNT) or (not R_pass):
                # Do re-initialization 
                init_flag = True
                init_R, init_t= pose_predictor.dataset.get_eval_target_pose(reinit_poses,img_id,obj_id, instance_id = instance_id)
                init_ok = check_init_status(init_t)
                if init_ok:
                    pred_R, pred_t = init_R, init_t
                    vis_amounts, delta_ts, pred_Rs, similarity_scores, confidence = pose_predictor.new_predict(rgb_imgs[i], depth_imgs[i], pred_R, pred_t, K, no_rot_depth = False)
                    vis_amnt[i] = vis_amounts[0]
                    confidences[i] = confidence[0]
                    pred_step = 0
                    R_pass = (confidence[0]> 0.6 ) if USE_R_CONFI else True 
                    if (vis_amounts[0] < RE_INIT_VIS_AMNT) or (not R_pass):
                        next_init_flag = True
                    else:
                        next_init_flag = False 
                        re_init_times += 1
            else:
                next_init_flag = False

        pred_t = pred_t + delta_ts[0].reshape((3,1))*pose_predictor.dataset.max_delta_t_shift
 
        if False:
            similarity_score = similarity_scores[0]
        else:
            similarity_score = vis_amounts[0]*similarity_scores[0] + (1-vis_amounts[0])*similarity_scores[1]

        top_k = 1
        unsorted_max_idcs = np.argpartition(-similarity_score, top_k)[:top_k]
        idcs = unsorted_max_idcs[np.argsort(-similarity_score[unsorted_max_idcs])]
        pred_R = pose_predictor.dataset.viewsphere_for_embedding[idcs]
    if not init_ok:
        next_init_flag = True

    no_rot_crt_Rs[i] = pred_R
    pred_R = pose_predictor.correct_rotation_shift(pred_R, pred_t)
    if False:
        # Rs_est_old, ts_est_old = pred_R.copy(), pred_t.copy()
        # R_est_refined, t_est_refined = pose_predictor.dataset.get_eval_target_pose(reinit_poses,img_id,obj_id, instance_id=instance_id)
        # gt_Rs[i], gt_ts[i] = pose_predictor.dataset.get_eval_target_pose(img_gt,img_id,obj_id, instance_id=instance_id)
        # R_est_refined, t_est_refined = icp_utils.icp_refinement(depth_imgs[i], icp_renderer, pred_R, pred_t.reshape((3,)), K.copy(), max_mean_dist_factor=5.0)
        H, W = depth_imgs[i].shape
        # ori_rgb, _, _ = pose_predictor.dataset.render_gt_image(W = W, H = H, K=K, R=pred_R, t=pred_t)
        # ref_rgb, _, _ = pose_predictor.dataset.render_gt_image(W = W, H = H, K=K, R=R_est_refined, t=t_est_refined)
        # if R_ONLY:
        #     t_est_refined[2] = pred_t[2]
        # else:
        #     t_est_refined = pred_t
        # pred_R = R_est_refined
        # pred_t =t_est_refined
        ref_rgb, _, _ = pose_predictor.dataset.render_gt_image(W = W, H = H, K=K, R=pred_R, t=pred_t)

        overlapping = cv2.addWeighted(ref_rgb, 0.8, rgb_imgs[i], 0.5, 0)
        # cv2.imshow('ori', ori_rgb)
        cv2.imshow('ref', overlapping)
        # cv2.imshow('in', rgb_imgs[i])
        cv2.waitKey(1)
            # print(t_est_refined, R_est_refined)
        # pred_R, pred_t = R_est_refined, t_est_refined.reshape((3,1))
        # pred_t= t_est_refined.reshape((3,1))
        
    init_R, init_t= pose_predictor.dataset.get_eval_target_pose(reinit_poses,img_id,obj_id, instance_id = instance_id)
    if R_ONLY:
        # pred_t[:2] = init_t[:2]
        pred_t = init_t
    if T_ONLY:
        pred_R = init_R
    gt_Rs[i], gt_ts[i] = pose_predictor.dataset.get_eval_target_pose(img_gt,img_id,obj_id, instance_id=instance_id)
    out_t[i] = pred_t if not ROTATION_TRACK_ONLY else gt_ts[i] 
    out_R[i] = pred_R
    if REFINE_MODE and (not init_ok):
        out_t[i] = np.zeros(pred_t.shape)
        out_R[i] = np.eye(3)
    gt_Ks[i] = K
    init_flags[i] = init_flag 
    if USE_LSTM:
        result_poses[pred_step,:3,:3] = pred_R
        result_poses[pred_step,:3, 3] = pred_t.reshape(3,)
    pred_step += 1
    init_flag = init_flag if no_re_init else next_init_flag
    if REFINE_MODE:
        init_flag = True
    runtime = (time.time() - start_time)

    score = float(vis_amounts[0] + confidence[0])
    vis_R_res_path = os.path.join(vis_R_result_dir_path, str(img_id).zfill(4)+'_'+str(obj_id).zfill(2)+'.yml')
    store_result(vis_R_res_path, score, out_R[i], out_t[i], runtime)
    vis_R_res_no_crt_path = os.path.join(vis_R_result_no_crt_dir_path, str(img_id).zfill(4)+'_'+str(obj_id).zfill(2)+'.yml')
    store_result(vis_R_res_no_crt_path, score, no_rot_crt_Rs[i], out_t[i], runtime)

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
if DISTIBUTION:
    result_file_name += ('_dist')
if USE_NOISE_VERSION:
    result_file_name += ('_noise')

result_file_name += '.txt'
result_file_name = os.path.join(workspace_path, 'est_results', result_file_name)
# np.savez(main_work_folder+'/pose_prediction.npz', R=out_R, t= out_t, vis_amnt=vis_amnt, confidence = confidences, gt_ts = gt_ts, gt_Rs = gt_Rs, gt_Ks = gt_Ks, lost_flags = init_flags, no_rot_crt_Rs = no_rot_crt_Rs, file_name = result_file_name, scene_id = SCENE_ID, instance_id = INSTANCE_ID)
print('finished')


if OUTPUT:
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
        out_info = str(obj_id)+','+ str(SCENE_ID) +',' + str(INSTANCE_ID) + ',' + str(re_init_times) + ',' + str(runtime) + ',' + str(eval_num) + '\n' 
        f.write(out_info)