
import os
import numpy as np
import functools
import cv2
import progressbar
import shutil
from pytless import inout
from pysixd_stuff import transform
from sklearn.metrics.pairwise import cosine_similarity

# https://danijar.com/structuring-your-tensorflow-models/
def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator

def batch_iteration_indices(N, batch_size):
    end = int(np.ceil(float(N) / float(batch_size)))
    for i in range(end):
        a = i*batch_size
        e = i*batch_size+batch_size
        e = e if e <= N else N
        yield (a, e)

def get_dataset_path(workspace_path):
    return os.path.join(
        workspace_path, 
        'tmp_datasets',
    )

def get_checkpoint_dir(log_dir):
    return os.path.join(
        log_dir, 
        'checkpoints'
    )

def get_log_dir(workspace_path, experiment_name, experiment_group=''):
    return os.path.join(
        workspace_path, 
        'experiments',
        experiment_group,
        experiment_name
    )

def get_train_fig_dir(log_dir):
    return os.path.join(
        log_dir, 
        'train_figures'
    )

def get_train_config_exp_file_path(log_dir, experiment_name):
    return os.path.join(
        log_dir,
        '{}.cfg'.format(experiment_name)
    )

def get_checkpoint_basefilename(log_dir):
    return os.path.join(
        log_dir,
        'checkpoints',
        'chkpt'
    )

def get_test_summary_dir(ckpt_dir):
    summary_path = os.path.join(ckpt_dir,'test')
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)
    return summary_path

def get_config_file_path(workspace_path, experiment_name, experiment_group=''):
    return os.path.join(
        workspace_path, 
        'cfg',
        experiment_group,
        '{}.cfg'.format(experiment_name)
    )



def get_eval_config_file_path(workspace_path, eval_cfg='eval.cfg'):
    return os.path.join(
        workspace_path, 
        'cfg_eval',
        eval_cfg
    )

def get_eval_dir(log_dir, evaluation_name, data):
    return os.path.join(
        log_dir,
        'eval',
        evaluation_name,
        data
    )


def tiles(batch, rows, cols, spacing_x=0, spacing_y=0, scale=1.0):
    if batch.ndim == 4:
        N, H, W, C = batch.shape
    elif batch.ndim == 3:
        N, H, W = batch.shape
        C = 1
    else:
        raise ValueError('Invalid batch shape: {}'.format(batch.shape))

    H = int(H*scale)
    W = int(W*scale)
    img = np.ones((rows*H+(rows-1)*spacing_y, cols*W+(cols-1)*spacing_x, C))*255
    i = 0
    for row in range(rows):
        for col in range(cols):
            start_y = row*(H+spacing_y)
            end_y = start_y + H
            start_x = col*(W+spacing_x)
            end_x = start_x + W
            if i < N:
                if C > 1:
                    img[start_y:end_y,start_x:end_x,:] = cv2.resize(batch[i], (W,H))
                else:
                    img[start_y:end_y,start_x:end_x,0] = cv2.resize(batch[i], (W,H))
            i += 1
    return img

def progressbar_init(show_message, maxval):
    widgets = [show_message, progressbar.Percentage(),
         ' ', progressbar.Bar(),
         ' ', progressbar.Counter(), ' / %s' % maxval,
         ' ', progressbar.ETA(), ' ']
    bar = progressbar.ProgressBar(maxval=maxval,widgets=widgets)
    return bar

def file_paths_init(workspace_path, experiment_name, experiment_group, is_training= False):
    cfg_file_path = get_config_file_path(workspace_path, experiment_name, experiment_group)
    log_dir = get_log_dir(workspace_path, experiment_name, experiment_group)
    checkpoint_file = get_checkpoint_basefilename(log_dir)
    ckpt_dir = get_checkpoint_dir(log_dir)
    dataset_path = get_dataset_path(workspace_path)

    if not os.path.exists(cfg_file_path):
        print('Could not find config file:\n')
        print('{}\n'.format(cfg_file_path))
        exit(-1)

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    if is_training:
        train_fig_dir = get_train_fig_dir(log_dir)
        if not os.path.exists(train_fig_dir):
            os.makedirs(train_fig_dir)
        shutil.copy2(cfg_file_path, log_dir)
        test_summary_dir = get_test_summary_dir(ckpt_dir)
        return cfg_file_path, checkpoint_file, ckpt_dir, train_fig_dir, dataset_path, test_summary_dir
    return cfg_file_path, checkpoint_file, ckpt_dir, dataset_path

def load_evaluation_infos(args, dataset):
    eval_keywords = {k:v for k,v in args.items('Evaluation')}
    img_info = inout.load_info(eval_keywords['evaluation_images_info_path'])
    img_gt = inout.load_gt(eval_keywords['evaluation_images_gt_path']) 
    rgb_imgs = dataset.load_images(eval_keywords['evaluation_rgb_images_glob']) 
    depth_imgs = dataset.load_images(eval_keywords['evaluation_depth_images_glob'], depth=True, depth_scale=0.1) 
    obj_id = int(eval_keywords['evaluation_model_index'])
    instance_id =  int(eval_keywords['evaluation_instance_index'])
    return img_info, img_gt, rgb_imgs, depth_imgs, obj_id, instance_id

def load_evaluation_infos_by_scene(args, dataset, scene_id, ycb_v = False):
    SCENE_NAME_LEN = 6 if ycb_v else 2
    eval_keywords = {k:v for k,v in args.items('Evaluation')}
    img_info = inout.load_info(eval_keywords['evaluation_images_info_path_new'].format(str(scene_id).zfill(SCENE_NAME_LEN)))
    img_gt = inout.load_gt(eval_keywords['evaluation_images_gt_path_new'].format(str(scene_id).zfill(SCENE_NAME_LEN))) 
    rgb_imgs = dataset.load_images(eval_keywords['evaluation_rgb_images_glob_new'].format(str(scene_id).zfill(SCENE_NAME_LEN))) 
    depth_imgs = dataset.load_images(eval_keywords['evaluation_depth_images_glob_new'].format(str(scene_id).zfill(SCENE_NAME_LEN)), depth=True, depth_scale=0.1) 
    obj_id = int(eval_keywords['evaluation_model_index'])
    return img_info, img_gt, rgb_imgs, depth_imgs, obj_id

def get_gt_vis_amount(args, obj_index, scene_id = None):
    eval_keywords = {k:v for k,v in args.items('Evaluation')}
    if scene_id is not None:
        gt_info_path = os.path.join(os.path.dirname(eval_keywords['evaluation_images_info_path_new'].format(str(scene_id).zfill(2)))\
        , 'scene_gt_info.json')
    else:
        gt_info_path  = os.path.join(os.path.dirname(eval_keywords['evaluation_images_info_path']), 'scene_gt_info.json')
    import json
    with open(gt_info_path) as json_file:
        data = json.load(json_file)
    image_num = len(data)
    vis_amounts = []
    for i in range(image_num):
        vis_amounts.append(float(data[str(i)][obj_index]['visib_fract']))
    return np.array(vis_amounts)

def get_ave_R(scores, Rs):
    possibility = scores/sum(scores)
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
        ave_dR = get_ave_R(weights, dRs[start:end])
        pred_R = np.matmul(ave_dR, ori_R)
        return pred_R 

def pred_next_translation(dts, ori_t, current_index, average_num = 0):
    current_index -= 1
    if average_num == 0:
        return ori_t
    else:
        start = max(0, current_index - average_num)
        end = current_index
        ave_dt = np.mean(dts[start:end], axis=0)
        pred_t = ave_dt + ori_t 
        return pred_t 

def get_render_possbility(sess, dataset, codebook, R):
    img_patch = dataset.generate_rotation_image_patch(R, R)
    img_patch = img_patch/255.
    # img_patch[:,:, 0:3] = 0 
    _, similarity_score = codebook.nearest_rotation(sess, img_patch, return_similarity_score=True)
    return similarity_score

def get_rotation_similarity_score(rot_list, new_rot):
    temp_matrix = np.eye(4)
    temp_matrix[0:3, 0:3] = new_rot 
    rot_quaternion = transform.quaternion_from_matrix(temp_matrix).reshape((1,-1))
    out = cosine_similarity(rot_list, rot_quaternion)
    return out

def get_quaternion_list(dataset):
    rot_num = len(dataset.viewsphere_for_embedding)
    out = np.empty((rot_num, 4), dtype=np.float64)
    for i in range(rot_num):
        temp_matrix = np.eye(4)
        temp_matrix[0:3, 0:3] = dataset.viewsphere_for_embedding[i]
        out[i] = transform.quaternion_from_matrix(temp_matrix)
    return out

