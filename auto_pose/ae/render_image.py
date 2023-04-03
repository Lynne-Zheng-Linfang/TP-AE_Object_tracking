# -*- coding: utf-8 -*-

import numpy as np
import time
import hashlib
import glob
import os
# import progressbar
import cv2
import random
import math
import json

import tensorflow as tf
from pysixd_stuff import transform
from pysixd_stuff import view_sampler
from utils import lazy_property
from pytless import inout
import im_process
import utils 

def load_json(path, keys_to_int=False):
  """Loads content of a JSON file.

  :param path: Path to the JSON file.
  :return: Content of the loaded JSON file.
  """
  # Keys to integers.
  def convert_keys_to_int(x):
    return {int(k) if k.lstrip('-').isdigit() else k: v for k, v in x.items()}

  with open(path, 'r') as f:
    if keys_to_int:
      content = json.load(f, object_hook=lambda x: convert_keys_to_int(x))
    else:
      content = json.load(f)

  return content


def save_json(path, content):
  """Saves the provided content to a JSON file.

  :param path: Path to the output JSON file.
  :param content: Dictionary/list to save.
  """
  with open(path, 'w') as f:

    if isinstance(content, dict):
      f.write('{\n')
      content_sorted = sorted(content.items(), key=lambda x: x[0])
      for elem_id, (k, v) in enumerate(content_sorted):
        f.write('  \"{}\": {}'.format(k, json.dumps(v, sort_keys=True)))
        if elem_id != len(content) - 1:
          f.write(',')
        f.write('\n')
      f.write('}')

    elif isinstance(content, list):
      f.write('[\n')
      for elem_id, elem in enumerate(content):
        f.write('  {}'.format(json.dumps(elem, sort_keys=True)))
        if elem_id != len(content) - 1:
          f.write(',')
        f.write('\n')
      f.write(']')

    else:
      json.dump(content, f, sort_keys=True)


def load_cam_params(path):
  """Loads camera parameters from a JSON file.

  :param path: Path to the JSON file.
  :return: Dictionary with the following items:
   - 'im_size': (width, height).
   - 'K': 3x3 intrinsic camera matrix.
   - 'depth_scale': Scale factor to convert the depth images to mm (optional).
  """
  c = load_json(path)

  cam = {
    'im_size': (c['width'], c['height']),
    'K': np.array([[c['fx'], 0.0, c['cx']],
                   [0.0, c['fy'], c['cy']],
                   [0.0, 0.0, 1.0]])
  }

  if 'depth_scale' in c.keys():
    cam['depth_scale'] = float(c['depth_scale'])

  return cam


def load_scene_camera(path):
  """Loads content of a JSON file with information about the scene camera.

  See docs/bop_datasets_format.md for details.

  :param path: Path to the JSON file.
  :return: Dictionary with the loaded content.
  """
  scene_camera = load_json(path, keys_to_int=True)

  for im_id in scene_camera.keys():
    if 'cam_K' in scene_camera[im_id].keys():
      scene_camera[im_id]['cam_K'] = \
        np.array(scene_camera[im_id]['cam_K'], np.float).reshape((3, 3))
    if 'cam_R_w2c' in scene_camera[im_id].keys():
      scene_camera[im_id]['cam_R_w2c'] = \
        np.array(scene_camera[im_id]['cam_R_w2c'], np.float).reshape((3, 3))
    if 'cam_t_w2c' in scene_camera[im_id].keys():
      scene_camera[im_id]['cam_t_w2c'] = \
        np.array(scene_camera[im_id]['cam_t_w2c'], np.float).reshape((3, 1))
  return scene_camera


def save_scene_camera(path, scene_camera):
  """Saves information about the scene camera to a JSON file.

  See docs/bop_datasets_format.md for details.

  :param path: Path to the output JSON file.
  :param scene_camera: Dictionary to save to the JSON file.
  """
  for im_id in sorted(scene_camera.keys()):
    im_camera = scene_camera[im_id]
    if 'cam_K' in im_camera.keys():
      im_camera['cam_K'] = im_camera['cam_K'].flatten().tolist()
    if 'cam_R_w2c' in im_camera.keys():
      im_camera['cam_R_w2c'] = im_camera['cam_R_w2c'].flatten().tolist()
    if 'cam_t_w2c' in im_camera.keys():
      im_camera['cam_t_w2c'] = im_camera['cam_t_w2c'].flatten().tolist()
  save_json(path, scene_camera)


def load_scene_gt(path):
  """Loads content of a JSON file with ground-truth annotations.

  See docs/bop_datasets_format.md for details.

  :param path: Path to the JSON file.
  :return: Dictionary with the loaded content.
  """
  scene_gt = load_json(path, keys_to_int=True)

  for im_id, im_gt in scene_gt.items():
    for gt in im_gt:
      if 'cam_R_m2c' in gt.keys():
        gt['cam_R_m2c'] = np.array(gt['cam_R_m2c'], np.float).reshape((3, 3))
      if 'cam_t_m2c' in gt.keys():
        gt['cam_t_m2c'] = np.array(gt['cam_t_m2c'], np.float).reshape((3, 1))
  return scene_gt


def save_scene_gt(path, scene_gt):
  """Saves ground-truth annotations to a JSON file.

  See docs/bop_datasets_format.md for details.

  :param path: Path to the output JSON file.
  :param scene_gt: Dictionary to save to the JSON file.
  """
  for im_id in sorted(scene_gt.keys()):
    im_gts = scene_gt[im_id]
    for gt in im_gts:
      if 'cam_R_m2c' in gt.keys():
        gt['cam_R_m2c'] = gt['cam_R_m2c'].flatten().tolist()
      if 'cam_t_m2c' in gt.keys():
        gt['cam_t_m2c'] = gt['cam_t_m2c'].flatten().tolist()
      if 'obj_bb' in gt.keys():
        gt['obj_bb'] = [int(x) for x in gt['obj_bb']]
  save_json(path, scene_gt)


def load_bop_results(path, version='bop19'):
  """Loads 6D object pose estimates from a file.

  :param path: Path to a file with pose estimates.
  :param version: Version of the results.
  :return: List of loaded poses.
  """
  results = []

  # See docs/bop_challenge_2019.md for details.
  if version == 'bop19':
    header = 'scene_id,im_id,obj_id,score,R,t,time'
    with open(path, 'r') as f:
      line_id = 0
      for line in f:
        line_id += 1
        if line_id == 1 and header in line:
          continue
        else:
          elems = line.split(',')
          if len(elems) != 7:
            raise ValueError(
              'A line does not have 7 comma-sep. elements: {}'.format(line))

          result = {
            'scene_id': int(elems[0]),
            'im_id': int(elems[1]),
            'obj_id': int(elems[2]),
            'score': float(elems[3]),
            'R': np.array(
              list(map(float, elems[4].split())), np.float).reshape((3, 3)),
            't': np.array(
              list(map(float, elems[5].split())), np.float).reshape((3, 1)),
            'time': float(elems[6])
          }

          results.append(result)
  else:
    raise ValueError('Unknown version of BOP results.')

  return results



class RenderImage(object):

    def __init__(self, model_path, store_path, model_id):
        self.model_path = model_path
        self.dataset_path = store_path
        self.target_obj_id = model_id

    @lazy_property
    def t_shape(self):
        return (3,1)
    
    @lazy_property
    def R_shape(self):
        return (3,3)
    
    @lazy_property
    def renderer(self):
        from meshrenderer import meshrenderer_phong

        model_paths = glob.glob(self.model_path)
        if model_paths == []:
            print('No model file found in model path! Please check with your model path.')
            exit()
        model_paths.sort()
        renderer = meshrenderer_phong.Renderer(
            model_paths,
            1,
            self.dataset_path,
            1
        )
        return renderer
    

    @lazy_property
    def render_dim(self):
        return np.array([720,1280])

    @lazy_property
    def render_K(self):
        return np.array([1075.65091572, 0, 1280/2, 0, 1073.90347929, 720/2, 0, 0, 1])

    @lazy_property
    def clip_near(self):
        return 10
    @lazy_property
    def clip_far(self):
        return 10000 

    def render_gt_image(self, R, t, random_light = False):
        K = self.render_K.reshape((3,3)) 
        R = np.array(R).reshape((3,3)) 
        t = np.array(t).flatten()
        H, W = self.render_dim
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

    def render_image_k(self,H,W, R, t, K, random_light = False):
        K = K.reshape((3,3)) 
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

model_path = '/home/linfang/Documents/Dataset/T_LESS/t-less_kinect/models_reconst/*.ply'
store_path = '/home/linfang/Documents/Code/Others'
model_id = 2

ori_json_file = '/home/linfang/Documents/Dataset/T_LESS/tless_primesense/test_primesense/tless_and_ycbv_test_paths.json'
def check_lstm_result():
    import glob
    img_paths = glob.glob('/home/linfang/Documents/Dataset/T_LESS/tless_primesense/test_primesense/01/rgb/*.png')
    img_paths.sort()
    info = inout.load_info('/home/linfang/Documents/Dataset/T_LESS/tless_primesense/test_primesense/01/info.yml')
    data = np.load('/home/linfang/Documents/Code/AAE_tracker/ws1/experiments/lstm/poses.npz')
    gt = inout.load_gt('/home/linfang/Documents/Dataset/T_LESS/tless_primesense/test_primesense/01/gt.yml')
    Rs = data['Rs']
    ts = data['ts']
    output_file_path = '/home/linfang/Documents/Code/AAE_tracker/ws1/experiments/lstm/lstm_gt_pred_R_only.avi'
    fps = 15
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    ori_rgb = cv2.imread(img_paths[0])
    H, W = ori_rgb.shape[:2]
    videoWriter = cv2.VideoWriter(output_file_path, fourcc,fps, (W, H))
    # files.sort()
    render_image = RenderImage(model_path, store_path, model_id)
    for i in range(len(Rs)):
        ori_rgb = cv2.imread(img_paths[i])
        H, W = ori_rgb.shape[:2]
        rgb, _, _ = render_image.render_image_k(H,W,Rs[i], ts[i], info[i]['cam_K'])
        # rgb, _, _ = render_image.render_image_k(H,W,Rs[i],  gt[i][0]['cam_t_m2c'], info[i]['cam_K'])
        # rgb, _, _ = render_image.render_image_k(H,W, gt[i][0]['cam_R_m2c'], ts[i], info[i]['cam_K'])
        overlapping = cv2.addWeighted(rgb, 0.5, ori_rgb, 0.8, 0)
        cv2.imshow('rgb', overlapping)
        cv2.waitKey(80)
    #     videoWriter.write(overlapping)
    # videoWriter.release()

def check_constant_velocity_result():
    import glob
    img_paths = glob.glob('/home/linfang/Documents/Dataset/T_LESS/tless_primesense/test_primesense/01/rgb/*.png')
    img_paths.sort()
    info = inout.load_info('/home/linfang/Documents/Dataset/T_LESS/tless_primesense/test_primesense/01/info.yml')
    gt = inout.load_gt('/home/linfang/Documents/Dataset/T_LESS/tless_primesense/test_primesense/01/gt.yml')
    output_file_path = '/home/linfang/Documents/Code/AAE_tracker/ws1/experiments/lstm/const_velo_gt_pred_R_only.avi'

    eval_num = len(gt)
    dRs = np.empty((eval_num,)+(3,3), dtype=np.float)
    gt_Rs = np.empty((eval_num,)+(3,3), dtype=np.float)
    gt_ts = np.empty((eval_num,)+(3,1), dtype=np.float)
    dts = np.empty((eval_num,)+(3,1), dtype=np.float)

    fps = 15
    average_num = 5 
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    ori_rgb = cv2.imread(img_paths[0])
    H, W = ori_rgb.shape[:2]

    videoWriter = cv2.VideoWriter(output_file_path, fourcc,fps, (W, H))
    # files.sort()
    render_image = RenderImage(model_path, store_path, model_id)
    for i in range(len(gt)):
        ori_rgb = cv2.imread(img_paths[i])
        H, W = ori_rgb.shape[:2]
        gt_Rs[i] = gt[i][0]['cam_R_m2c']
        gt_ts[i] = gt[i][0]['cam_t_m2c']
        if i >= 2:
            dRs[i-2] = np.matmul(gt_Rs[i-2].transpose(), gt_Rs[i-1])
            dts[i-2] = gt_ts[i-1] - gt_ts[i-2]
            # pred_R = np.matmul(dRs[i-2].transpose(), gt_Rs[i-1])
            # pred_R = np.matmul(dRs[i-2].transpose(), gt_Rs[i-1])
            pred_R = utils.pred_next_rotation(dRs, gt_Rs[i-1], i, average_num = average_num)
            pred_t = utils.pred_next_translation(dts, gt_ts[i-1], i, average_num = average_num)
            pred_t = gt_ts[i]
            # pred_R = gt_Rs[i]
        else:
            pred_R = gt_Rs[i]
            pred_t = gt_ts[i]
        rgb, _, _ = render_image.render_image_k(H,W, pred_R, pred_t, info[i]['cam_K'])
        overlapping = cv2.addWeighted(rgb, 0.5, ori_rgb, 0.8, 0)
        cv2.imshow('rgb', overlapping)
        cv2.waitKey(80)
    #     videoWriter.write(overlapping)
    # videoWriter.release()
# check_constant_velocity_result()

def render_BoP_result(render_gt = False):
    import glob
    bop_result = '/home/linfang/Documents/Code/AAE_tracker/ws1/BOP_results/final_version/poseRBPF-rgbd-res-PoseRBPF_tless-test-primesense.csv'
    # bop_result = '/home/linfang/Documents/Code/AAE_tracker/ws1/BOP_results/final_version/NewVisReinit-020-GRU-10-cosy-rpass-single-clean-pure-real-syn-final-vis-r-res-PriAAE_tless-test-primesense.csv'
    # bop_result = '/home/linfang/Documents/Code/cosypose/local_data/bop_predictions_csv/cosypose955843-eccv2020_tless-test-primesense.csv'

    model_path = '/home/linfang/Documents/Dataset/T_LESS/t-less_kinect/models_reconst/*.ply'
    store_path = '/home/linfang/Documents/Code/Others'

    data_base = '/home/linfang/Documents/Dataset/T_LESS/tless_primesense/test_primesense'
    # gt = inout.load_gt(os.path.join(data_base,str(obj_id).zfill(2),'gt.yml'))
    for obj_id in [17]:
        output_base = '/home/linfang/Documents/Code/AAE_tracker/ws1/qualitive_results/gt/obj_'+str(obj_id).zfill(2)
        render_image = RenderImage(model_path, store_path, model_id = obj_id - 1)
        if not os.path.exists(output_base):
            os.makedirs(output_base)
        results = load_bop_results(bop_result)
        # for scene_id in [19]:
        for scene_id in range(1,21):
            if render_gt:
                gt = inout.load_gt(os.path.join(data_base, str(scene_id).zfill(2), 'gt.yml'))
            out_scene_base = os.path.join(output_base, str(scene_id).zfill(2))
            if not os.path.exists(out_scene_base):
                os.makedirs(out_scene_base)
            info = inout.load_info(os.path.join(data_base,str(scene_id).zfill(2),'info.yml'))
            for res in results:
                if res['obj_id'] != obj_id:
                    continue
                if res['scene_id'] != scene_id:
                    continue
                t = res['t']
                if (t.flatten() == np.array([0,0,0])).all(): 
                    continue
                R = res['R']
                img_id = res['im_id']
                if img_id != 447:
                    continue
                rgb_path = os.path.join(data_base,str(scene_id).zfill(2),'rgb', str(img_id).zfill(4)+'.png')
                ori_rgb = cv2.imread(rgb_path) 
                H, W = ori_rgb.shape[:-1]
                if render_gt:
                    count = 0
                    for pose_gt in gt[img_id]:
                        if pose_gt['obj_id'] == obj_id:
                            gt_R = pose_gt['cam_R_m2c']
                            gt_t = pose_gt['cam_t_m2c']
                            rgb, _, _ = render_image.render_image_k(H,W, gt_R, gt_t, info[img_id]['cam_K'])
                            overlapping = cv2.addWeighted(rgb, 0.5, ori_rgb, 0.8, 0)
                            write_path = os.path.join(out_scene_base, 'gt_'+str(img_id).zfill(4)+'_'+str(count)+'.png') 
                            cv2.imwrite(write_path, overlapping)
                    continue

                rgb, _, _ = render_image.render_image_k(H,W, R, t, info[img_id]['cam_K'])
                overlapping = cv2.addWeighted(rgb, 0.5, ori_rgb, 0.8, 0)
                write_path = os.path.join(out_scene_base, str(img_id).zfill(4)+'.png') 
                cv2.imwrite(write_path, overlapping)
                # cv2.imshow('rgb', overlapping)
                # cv2.waitKey(15)
 
# check_lstm_result()
render_BoP_result(render_gt=True)