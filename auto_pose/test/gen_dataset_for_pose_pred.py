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
import progressbar

from auto_pose.ae import factory, utils

parser = argparse.ArgumentParser()
parser.add_argument("experiment_name")
# parser.add_argument("-f", "--folder_str", required=True, help='folder or filename to image(s)')
parser.add_argument("-eval", action='store_true', help='generate evaluation code and image set for pose prediction')

arguments = parser.parse_args()
full_name = arguments.experiment_name.split('/')
experiment_name = full_name.pop()
experiment_group = full_name.pop() if len(full_name) > 0 else ''
eval_dataset_flag = arguments.eval
# file_str = arguments.file_str
# if os.path.isdir(file_str):
#     rgb_file_paths = sorted(glob.glob(os.path.join(str(file_str),'rgb','*.png'))+glob.glob(os.path.join(str(file_str), 'rgb', '*.jpg')))
#     depth_file_paths = sorted(glob.glob(os.path.join(str(file_str), 'depth', '*.png'))+glob.glob(os.path.join(str(file_str), 'depth', '*.jpg')))
# else:
#     print('Error: --folder_str should be a folder which contains a rgb folder and a depth folder')

workspace_path = os.environ.get('AE_WORKSPACE_PATH')
if workspace_path == None:
    print('Please define a workspace path:\n')
    print('export AE_WORKSPACE_PATH=/path/to/workspace\n')
    exit(-1)
log_dir = utils.get_log_dir(workspace_path,experiment_name,experiment_group)
ckpt_dir = utils.get_checkpoint_dir(log_dir)
dataset_path = utils.get_dataset_path(workspace_path)
train_fig_dir = utils.get_train_fig_dir(log_dir)

encoder, dataset, decoder = factory.build_inference_architecture(experiment_name, experiment_group, return_dataset=True, return_decoder=True)

cfg_file_path = utils.get_config_file_path(workspace_path, experiment_name, experiment_group)
if not os.path.exists(cfg_file_path):
    print('Could not find config file:\n')
    print('{}\n'.format(cfg_file_path))
    exit(-1)
args = configparser.ConfigParser()
args.read(cfg_file_path)

print('loading training dataset..')
if eval_dataset_flag:
    # train_x, _, _, _, _ = dataset.load_eval_images(dataset_path, args)
    DATASET_PATH = '/home/linfang/Documents/Code/AugmentedAutoencoder/Eval_Images_For_Obj_2_No_Pred.npz'
    eval_data = np.load(DATASET_PATH)
    train_x = eval_data['eval_x']
else:
    train_x, _, _, _, _, _, _, _, _= dataset.load_all_training_data(dataset_path, args)

gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction = 0.9)
config = tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True

total_img_num = len(train_x)
batch_size = args.getint('Training', 'BATCH_SIZE') 
code_size = args.getint('Network', 'LATENT_SPACE_SIZE')
# recursive_range = [2,5]
# filter_range = [0.02,0.075]
recursive_range = [3,3]
filter_range = [0.075,0.075]

total_generate_num = int(total_img_num/batch_size)
res_num = total_img_num - total_generate_num*batch_size

gen_x = np.empty((total_img_num,)+train_x.shape[1:], dtype=float) #TODO: if want to store pictures, need to be changed to np.uint8
gen_codes = np.empty((total_img_num, code_size), dtype=float)
gen_indices = np.empty((total_img_num,), dtype=int)

widgets = ['Generating: ', progressbar.Percentage(),
     ' ', progressbar.Bar(),
     ' ', progressbar.Counter(), ' / %s' % total_img_num,
     ' ', progressbar.ETA(), ' ']
bar = progressbar.ProgressBar(maxval=total_img_num, widgets=widgets)

bar.start()
with tf.Session(config=config) as sess:

    factory.restore_checkpoint(sess, tf.train.Saver(), ckpt_dir)

    for i in range(total_generate_num):
        bar.update(i*batch_size) 
        start = int(i*batch_size)
        if i == total_generate_num -1:
            end = int((i+1)*batch_size + res_num)
        else:
            end = int((i+1)*batch_size)
        # np.random.seed(i)
        # rand_idcs = np.random.choice(total_img_num, batch_size, replace=False)
        this_x = train_x[start:end]/255

        new_x = this_x.copy()
        recursive_num = random.randint(recursive_range[0], recursive_range[1])
        filter_level = random.uniform(filter_range[0], filter_range[1])

        for iteration in range(recursive_num):
            reconstr_train = sess.run(decoder.x,feed_dict={encoder.x:new_x})
            if iteration < recursive_num-1:
                new_x[reconstr_train <= filter_level] = 0
        gen_x[start: end, :,:,:] = new_x
        # gen_x[start: end, :,:,:] = (new_x*255).astype(np.uint8)
        gen_codes[start: end, :] = sess.run(encoder.z, feed_dict={encoder.x: new_x})
        # print(filter_level)
        # print(recursive_num)
        gen_indices[start: end] = range(start, end) 
        # eval_rgb_imgs = np.hstack(( utils.tiles(this_x[:,:,:,0:3], 4, 4),utils.tiles(new_x[:,:,:,0:3], 4, 4), utils.tiles(reconstr_train[:,:,:,0:3], 4, 4)))*255
        # cv2.imshow('rgb', eval_rgb_imgs.astype(np.uint8))
        # eval_depth_imgs = np.hstack(( utils.tiles(this_x[:,:,:,3:], 4, 4), utils.tiles(new_x[:,:,:,3:], 4, 4),utils.tiles(reconstr_train[:,:,:,3:], 4,4)))*255
        # cv2.imshow('depth', eval_depth_imgs.astype(np.uint8))
        # cv2.waitKey(20000)
        # exit()

bar.finish()
print('Saving results...')
current_config_hash = hashlib.md5((str(args.items('Dataset')+args.items('Paths'))).encode('utf-8')).hexdigest()
if eval_dataset_flag:
    current_file_name = os.path.join(dataset_path, 'eval_codes'+current_config_hash + '.npz')
else:
    current_file_name = os.path.join(dataset_path, 'codes'+current_config_hash + '.npz')
np.savez(current_file_name, gen_x = gen_x, gen_codes= gen_codes, gen_indices = gen_indices)
print('Finished')
