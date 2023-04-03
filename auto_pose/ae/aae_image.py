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
import heapq
import matplotlib.pyplot as plt
import ae_factory as factory
import utils
from pytless import inout
T_STORE_PATH = '/home/linfang/Documents/Code/AugmentedAutoencoder/translation_pred_drop.npz'
parser = argparse.ArgumentParser()
parser.add_argument("experiment_name")
parser.add_argument("-t", action="store_true", default=False, help='Output translation prediction result')
parser.add_argument("-cl", action="store_true", default=False, help='Classified translation prediction result')
parser.add_argument('-order', action='store_true', default=False)
parser.add_argument('-otp', action='store_true', default=False)
parser.add_argument('-syn', action='store_true', default=False)
parser.add_argument('-tdis', action='store_true', default=False)
arguments = parser.parse_args()

ordered_eval = arguments.order
class_t = arguments.cl
syn_eval = arguments.syn

full_name = arguments.experiment_name.split('/')
experiment_name = full_name.pop()
experiment_group = full_name.pop() if len(full_name) > 0 else ''
t_output = arguments.t
order_t_pred = arguments.otp

print('#'*20+experiment_name+'#'*20)
workspace_path = os.environ.get('AE_WORKSPACE_PATH')
if workspace_path == None:
    print('Please define a workspace path:\n')
    print('export AE_WORKSPACE_PATH=/path/to/workspace\n')
    exit(-1)
log_dir = utils.get_log_dir(workspace_path,experiment_name,experiment_group)
ckpt_dir = utils.get_checkpoint_dir(log_dir)
dataset_path = utils.get_dataset_path(workspace_path)
train_fig_dir = utils.get_train_fig_dir(log_dir)


cfg_file_path = utils.get_config_file_path(workspace_path, experiment_name, experiment_group)
if not os.path.exists(cfg_file_path):
    print('Could not find config file:\n')
    print('{}\n'.format(cfg_file_path))
    exit(-1)
args = configparser.ConfigParser()
args.read(cfg_file_path)

print('t_class:', class_t)
VISABLE_AMOUNT = eval(args.get('Network', 'VISABLE_AMOUNT'))
AUXILIARY_MASK  = eval(args.get('Network', 'AUXILIARY_MASK'))

codebook, predictor, R_encoder, R_decoder, t_encoder, t_decoder, dataset, input_x, no_rot_x = factory.build_inference_architecture(experiment_name, dataset_path, args)

with_pred_image = eval(args.get('Training', 'WITH_PRED_IMAGE'))
if ordered_eval:
    dataset.generate_ordered_eval_image_with_no_prediction(dataset_path, args)
    test_data = dataset.eval_x
    test_gt_t = dataset.eval_gt_t
    test_noisy_t = dataset.eval_noisy_t
    test_noisy_R = dataset.eval_noisy_R
    test_delta_t = test_noisy_t - test_gt_t
    test_no_rot_x = dataset.eval_no_rot_x
else:
    current_config_hash = hashlib.md5((str(args.items('Dataset')+args.items('Paths'))).encode('utf-8')).hexdigest()
    if syn_eval:
        test_data, _ = dataset.load_syn_rotation_only_dataset(dataset_path, args)
    else:
        test_data, test_delta_t, test_delta_R, test_gt_t, test_gt_R, test_noisy_R, test_no_rot_x= dataset.load_eval_images(dataset_path, args)
    
if ordered_eval:
    coarse_t = np.empty_like(test_delta_t.reshape(-1,3))
    fine_t_1 = np.empty_like(test_delta_t.reshape(-1,3))

per_process_gpu_memory_fraction = 0.9
config = factory.config_GPU(per_process_gpu_memory_fraction)
with tf.compat.v1.Session(config=config) as sess:
    factory.restore_checkpoint(sess, tf.compat.v1.train.Saver(), ckpt_dir)

    total_img_num = len(test_data)
    batch_size =64
    t_iter_num = int(total_img_num/batch_size)
    res_img_num = int(total_img_num - t_iter_num*batch_size)
    recursive_num = 1
    if t_output or ordered_eval:
        iter_coarse_t_result = np.empty((total_img_num,)) 
        iter_fine_t_result = np.empty((total_img_num,)) 
        for i in range(t_iter_num):
            start = int(i*batch_size)
            if i == t_iter_num -1:
                end = int((i+1)*batch_size + res_img_num)
            else:
                end = int((i+1)*batch_size)
            if with_pred_image:
                img_patch = np.empty((end-start,)+dataset.shape, dtype= np.uint8)
            this_x = test_data[start:end]
            this_R = test_noisy_R[start:end]
            this_no_rot_x = test_no_rot_x[start:end]
            if with_pred_image:
                for j in range(batch_size):
                    img_patch[j] = dataset.generate_rotation_image_patch(this_R[j],this_R[j])
                this_x = np.concatenate((this_x, img_patch), axis=-1)
            # this_x = test_data[start:end]/255.0
            if not class_t:
                this_t = -test_delta_t[start:end].reshape((-1,3))
            else:
                this_t = -test_delta_t[start:end]

            new_x = this_x.copy()
            new_no_rot_x = this_x.copy()
            # new_no_rot_x = this_no_rot_x.copy()
            
            if new_x.ndim == 3:
                new_x = np.expand_dims(new_x, 0)
            for iteration in range(recursive_num):
                if not class_t:

                    coarse_delta_t = sess.run(predictor.coarse_delta_t_out,feed_dict={input_x: new_x, no_rot_x: new_no_rot_x})
                    coarse_delta_t = coarse_delta_t*dataset.max_delta_t_shift

                    iter_coarse_t_result[start:end] = np.linalg.norm((coarse_delta_t - this_t), axis=1).flatten()
                    if ordered_eval:
                        coarse_t[start:end] = test_noisy_t[start:end].reshape(-1,3) + coarse_delta_t

                    fine_delta_t = sess.run(predictor.fine_delta_t_with_code_out,feed_dict={input_x: new_x , no_rot_x: new_no_rot_x})
                    fine_delta_t = fine_delta_t*dataset.max_delta_t_shift
                    iter_fine_t_result[start:end] = np.linalg.norm((fine_delta_t-this_t), axis=1).flatten()
                    if ordered_eval:
                        fine_t_1[start:end] = test_noisy_t[start:end].reshape(-1,3) + fine_delta_t 
                    reconstr_train = sess.run(t_decoder.x,feed_dict={input_x:new_x})
                    mask = (reconstr_train <= 0.075).all(axis=3)
                    if not with_pred_image:
                        new_x[mask] = 0
                else:
                    delta_t = np.array(sess.run(predictor.t_out, feed_dict={input_x: new_x, no_rot_x: new_no_rot_x}))
                    delta_t = delta_t*dataset.delta_t_resolution - dataset.max_delta_t_shift
                    iter_fine_t_result[start:end] = np.linalg.norm((delta_t-this_t.reshape((-1,3))), axis=1).flatten()
                    if ordered_eval:
                        fine_t_1[start:end] = test_noisy_t[start:end].reshape(-1,3) + delta_t 
                    reconstr_train = sess.run(t_decoder.x,feed_dict={input_x:new_x})
                    mask = (reconstr_train <= 0.075).all(axis=3)
                    if not with_pred_image:
                        new_x[mask] = 0
 
        if not class_t:
            iter_coarse_t_loss = np.mean(iter_coarse_t_result)
            iter_fine_t_loss_2 = np.mean(iter_fine_t_result)
            # print('non_iter_coarse_t_loss', non_iter_coarse_t_loss)

            print('iter_coarse_t_loss', iter_coarse_t_loss)
            print('coarse in 10%', sum(iter_coarse_t_result/dataset.target_obj_diameter <= 0.1)/len(iter_coarse_t_result))
            print('coarse in 7%', sum(iter_coarse_t_result/dataset.target_obj_diameter <= 0.07)/len(iter_coarse_t_result))
            print('coarse in 5%', sum(iter_coarse_t_result/dataset.target_obj_diameter <= 0.05)/len(iter_coarse_t_result))
            # print('non_iter_fine_t_loss', non_iter_fine_t_loss)
            print('iter_fine_t_loss_with_code', iter_fine_t_loss_2)
            print('fine_2 in 10%', sum(iter_fine_t_result/dataset.target_obj_diameter <= 0.1)/len(iter_fine_t_result))
            print('fine_2 in 7%', sum(iter_fine_t_result/dataset.target_obj_diameter <= 0.07)/len(iter_fine_t_result))
            print('fine_2 in 5%', sum(iter_fine_t_result/dataset.target_obj_diameter <= 0.05)/len(iter_fine_t_result))
        
        if ordered_eval:
            np.savez('/home/linfang/Documents/Code/AugmentedAutoencoder/translation_pred_drop.npz', coarse_t=coarse_t, fine_t_1=fine_t_1)
        xaxes = 'translation error'
        yaxes = 'image num'
        titles = ['fine translation prediction error (code)','coarse translation error'] 

        # if not class_t:
        #     f,a = plt.subplots(1,2)
        #     a = a.ravel()
        #     for idx,ax in enumerate(a):
        #         if idx == 0:
        #             data = iter_fine_t_result
        #         else:
        #             data = iter_coarse_t_result
        #         ax.hist(data, bins=300)
        #         ax.set_title(titles[idx])
        #         ax.set_xlabel(xaxes)
        #         ax.set_ylabel(yaxes)
        #     plt.tight_layout()
        # else:
        #     plt.hist(iter_fine_t_result/dataset.target_obj_diameter, bins=300)
        # plt.show() 
        
    elif order_t_pred:
        img_info = inout.load_info(dataset._kw['evaluation_images_info_path'])
        img_gt = inout.load_gt(dataset._kw['evaluation_images_gt_path']) 
        rgb_imgs = dataset.load_images(dataset._kw['evaluation_rgb_images_glob']) 
        depth_imgs = dataset.load_images(dataset._kw['evaluation_depth_images_glob'], depth=True, depth_scale=0.1) 
        height, width, layers = rgb_imgs[0].shape
        out_t = np.empty((3,len(rgb_imgs), 3))
        
        for t_type in range(2):
            if class_t:
                pred_func = predictor.t_out
            else:
                if t_type == 0:
                    pred_func = predictor.coarse_delta_t_out
                else:
                    pred_func = predictor.fine_delta_t_with_code_out

            for i in range(len(rgb_imgs)):
                K = dataset.get_intrinsic_parameters(img_info, i)

                if i == 0:
                    pred_t = test_gt_t[0]
                    pred_R = test_gt_R[0]
                # elif i==1:
                #     pred_t = out_t[t_type, i-1]
                else:
                    dt = out_t[t_type, i-1] - out_t[t_type, i-2]
                    # pred_t = out_t[t_type, i-1] + dt
                    pred_t = out_t[t_type, i-1]
                    pred_R = test_gt_R[i-1]
                image_patch, _= dataset.train_images_preprocess(rgb_imgs[i], depth_imgs[i], pred_R, pred_t, K)
                # if image_patch.dtype == 'uint8':
                image_patch = image_patch
                if image_patch.ndim == 3:
                    image_patch = np.expand_dims(image_patch, 0)
  
                delta_t_pred = ((sess.run(pred_func, feed_dict={input_x: image_patch}))-0.5)*(dataset.target_obj_diameter * dataset.bbox_enlarge_level)
                out_t[t_type, i] = delta_t_pred.flatten()+ pred_t.flatten()
        
        print('Saving t results..')
        np.savez(T_STORE_PATH, coarse_t=out_t[0], fine_t_1= out_t[1])

    else:
        recursive_num = 1
        test_num = 20
        img_patch = np.empty((batch_size,)+dataset.shape, dtype= np.uint8)
        for i in range(test_num):

            np.random.seed(i)
            rand_idcs = np.random.choice(total_img_num, batch_size, replace=False)
            this_x = test_data[rand_idcs]
            this_no_rot_x = test_no_rot_x[rand_idcs]
            this_R = test_noisy_R[rand_idcs]
            if with_pred_image:
                for j in range(batch_size):
                    img_patch[j] = dataset.generate_rotation_image_patch(this_R[j],this_R[j])
                this_x = np.concatenate((this_x, img_patch), axis=-1)
            this_x = this_x/255
            this_no_rot_x = this_no_rot_x/255
            # this_t = test_delta_t[rand_idcs]

            reconstr_train = sess.run(t_decoder.x,feed_dict={t_encoder.x: this_x})
            new_x = this_no_rot_x[:,:,:,0:6].copy()
            new_x[reconstr_train < 0.075] = 0
            non_iter_result = reconstr_train.copy()

            eval_rgb_imgs = np.hstack(( utils.tiles(this_x[:,:,:,0:3], 4, 4),utils.tiles(new_x[:,:,:,0:3], 4, 4),utils.tiles(non_iter_result[:,:,:,0:3], 4, 4), utils.tiles(reconstr_train[:,:,:,0:3], 4, 4)))*255
            cv2.imwrite(os.path.join(train_fig_dir, 'eval_rgb_batch_%s.png' % str(i).zfill(3)), eval_rgb_imgs.astype(np.uint8))
            eval_depth_imgs = np.hstack(( utils.tiles(this_x[:,:,:,3:6], 4, 4), utils.tiles(new_x[:,:,:,3:6], 4, 4),utils.tiles(non_iter_result[:,:,:,3:6], 4, 4),utils.tiles(reconstr_train[:,:,:,3:6], 4,4)))*255
            cv2.imwrite(os.path.join(train_fig_dir,'eval_depth_batch_%s.png' % str(i).zfill(3)), eval_depth_imgs.astype(np.uint8))
    
    print('Finished')
