import cv2
import tensorflow as tf
import numpy as np
import glob
import os
import time
import argparse
import configparser
from sklearn.metrics.pairwise import cosine_similarity

from auto_pose.ae import factory, utils
from auto_pose.ae.pytless import inout
from auto_pose.ae.pysixd_stuff import transform

def get_render_possbility(sess, dataset, codebook, R, eval_x):
    img_patch = dataset.generate_rotation_image_patch(R, R)
    img_patch = img_patch/255.
    img_patch[:,:, 0:3] = 0 
    _, similarity_score = codebook.nearest_rotation(sess, img_patch, return_similarity_score=True)
    return similarity_score

def get_render_depth(dataset, R):
    img_patch = dataset.generate_rotation_image_patch(R, R)
    img_patch = img_patch/255.
    return img_patch[:,:, 3:] 

def get_ave_R(scores, Rs):
    possibility = scores/sum(scores)
    # print(possibility)
    sum_R = np.zeros((3,3))
    for i, R in enumerate(Rs):
        sum_R += R*possibility[i]
    u, _, v = np.linalg.svd(sum_R.transpose())
    if np.linalg.det(sum_R.transpose()) > 0:
        ave_R = v.transpose() @ u.transpose()
    else:
        H = np.eye(3)
        H[2,2] = -1
        ave_R = v.transpose() @ H @ u.transpose()
    return ave_R

def pred_next_rotation(dRs, ori_R, current_index, average_num = 0):
    current_index -= 1
    if average_num == 0:
        return ori_R
    else:
        start = max(0, current_index - average_num)
        end = current_index
        weights = np.ones(end-start)
        # print(weights)

        ave_dR = get_ave_R(weights, dRs[start:end])
        # print(ave_dR)
        # if current_index == 4:
        #     exit()
        pred_R = np.matmul(ave_dR, ori_R)
        return pred_R 

def get_quaternion_list():
    rot_num = len(dataset.viewsphere_for_embedding)
    out = np.empty((rot_num, 4), dtype=np.float64)
    for i in range(rot_num):
        temp_matrix = np.eye(4)
        temp_matrix[0:3, 0:3] = dataset.viewsphere_for_embedding[i]
        out[i] = transform.quaternion_from_matrix(temp_matrix)
    return out

def get_rotation_similarity_score(rot_list, new_rot):
    temp_matrix = np.eye(4)
    temp_matrix[0:3, 0:3] = new_rot 
    rot_quaternion = transform.quaternion_from_matrix(temp_matrix).reshape((1,-1))
    out = cosine_similarity(rot_list, rot_quaternion)
    return out
        
parser = argparse.ArgumentParser()
parser.add_argument("experiment_name")
arguments = parser.parse_args()
full_name = arguments.experiment_name.split('/')
experiment_name = full_name.pop()
experiment_group = full_name.pop() if len(full_name) > 0 else ''

workspace_path = os.environ.get('AE_WORKSPACE_PATH')
if workspace_path == None:
    print('Please define a workspace path:\n')
    print('export AE_WORKSPACE_PATH=/path/to/workspace\n')
    exit(-1)
main_work_folder = os.path.dirname(workspace_path)
log_dir = utils.get_log_dir(workspace_path,experiment_name,experiment_group)
ckpt_dir = utils.get_checkpoint_dir(log_dir)
dataset_path = utils.get_dataset_path(workspace_path)

cfg_file_path = utils.get_config_file_path(workspace_path, experiment_name, experiment_group)
if not os.path.exists(cfg_file_path):
    print('Could not find config file:\n')
    print('{}\n'.format(cfg_file_path))
    exit(-1)
args = configparser.ConfigParser()
args.read(cfg_file_path)

# codebook, dataset = factory.build_codebook_from_name(experiment_name, experiment_group, return_dataset=True)
codebook, predictor, R_encoder, R_decoder, t_encoder, t_decoder, dataset, input_x,_ = factory.build_inference_architecture(experiment_name, dataset_path, args)
# dataset.generate_ordered_eval_image_with_no_prediction()

eval_with_pred = eval(args.get('Training', 'WITH_PRED_IMAGE'))
img_info = inout.load_info(dataset._kw['evaluation_images_info_path'])
img_gt = inout.load_gt(dataset._kw['evaluation_images_gt_path']) 
rgb_imgs = dataset.load_images(dataset._kw['evaluation_rgb_images_glob']) 
depth_imgs = dataset.load_images(dataset._kw['evaluation_depth_images_glob'], depth=True, depth_scale=0.1) 
height, width, layers = rgb_imgs[0].shape

obj_id = dataset.target_obj_id + 1

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
gt_t = np.empty((eval_num,)+(3,1), dtype=np.float)

rot_list = get_quaternion_list()

per_process_gpu_memory_fraction = 0.9
config = factory.config_GPU(per_process_gpu_memory_fraction)
with tf.compat.v1.Session(config=config) as sess:

    factory.restore_checkpoint(sess, tf.compat.v1.train.Saver(), ckpt_dir)
    top_k = 10
    for i in range(eval_num):
        K = dataset.get_intrinsic_parameters(img_info, i)
        if i == 0:
            pred_R, gt_t[i]= dataset.get_eval_target_pose(img_gt, i, obj_id)
            eval_noisy_t = gt_t[i]
        elif i == 1:
            pred_R = out_R[i-1]
            _, gt_t[i]= dataset.get_eval_target_pose(img_gt, i, obj_id)
            # pred_R, _= dataset.get_eval_target_pose(img_gt, i-1, obj_id)
            eval_noisy_t = gt_t[i-1]
        else:
            _, gt_t[i]= dataset.get_eval_target_pose(img_gt, i, obj_id)
            pred_R = out_R[i-1]
            # pred_R, _= dataset.get_eval_target_pose(img_gt, i-1, obj_id)
            # last_R, _= dataset.get_eval_target_pose(img_gt, i-2, obj_id)
            # dRs[i-2] = np.matmul(last_R.transpose(), pred_R)
            eval_noisy_t = gt_t[i-1]
            dRs[i-2] = np.matmul(out_R[i-2].transpose(), out_R[i-1])
            pred_R = pred_next_rotation(dRs, pred_R, i, average_num = 4)
            # pred_R = np.matmul(dRs[i-1], out_R[i-1])
        for _ in range(1):
            depth_patch, _= dataset.depth_preprocess(depth_imgs[i], pred_R, eval_noisy_t, K) # Use gt_t of last frame and Rotation estimation result as prediction
            rgb_patch = dataset.rgb_preprocess(rgb_imgs[i], eval_noisy_t, K)
            eval_x = np.concatenate((rgb_patch, depth_patch), axis=2)
            if eval_with_pred:
                img_patch = dataset.generate_rotation_image_patch(pred_R, pred_R)
                eval_x = np.concatenate((eval_x, img_patch), axis=-1)
            eval_x = eval_x/255

            R, similarity_score_1 = codebook.nearest_rotation(sess, eval_x, return_similarity_score=True)

            render_possbility = get_render_possbility(sess, dataset, codebook, pred_R, eval_x)
 
            similarity_score = similarity_score_1 + render_possbility*0.2

            if True:
                top_k = 1
                unsorted_max_idcs = np.argpartition(-similarity_score.squeeze(), top_k)[:top_k]
                idcs = unsorted_max_idcs[np.argsort(-similarity_score.squeeze()[unsorted_max_idcs])]
                pred_Rs = dataset.viewsphere_for_embedding[idcs]
                # scores = similarity_score.squeeze()[idcs]
            else:
                top_k = 100
                unsorted_max_idcs = np.argpartition(-similarity_score.squeeze(), top_k)[:top_k]
                idcs = unsorted_max_idcs[np.argsort(-similarity_score.squeeze()[unsorted_max_idcs])]
                pred_Rs = dataset.viewsphere_for_embedding[idcs]
                pred_scores = pred_similarity_score[idcs]
                scores_ori = similarity_score.squeeze()[idcs]
                scores = scores_ori*(pred_scores+1)
                scores_ori = similarity_score.squeeze()[idcs]
            
                top_k = 3
                unsorted_max_idcs = np.argpartition(-scores.squeeze(), top_k)[:top_k]
                idcs = unsorted_max_idcs[np.argsort(-scores.squeeze()[unsorted_max_idcs])]
                pred_Rs = pred_Rs[idcs]
                scores = scores_ori[idcs]
            if top_k ==1:
                pred_R = pred_Rs
            else:
                pred_R = get_ave_R(scores, pred_Rs)


            # pred_R = dataset.viewsphere_for_embedding[idcs].squeeze()
        # out_R[i] = dataset.viewsphere_for_embedding[idcs].squeeze()
        out_R[i] = pred_R

    np.savez(main_work_folder+'/rotation_prediction.npz', R=out_R, t= gt_t)
    print('finished')