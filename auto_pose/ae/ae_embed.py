# -*- coding: utf-8 -*-
import os
import configparser
import argparse
import numpy as np
import signal
import progressbar
import tensorflow as tf

import ae_factory as factory
import utils as u

def main():
    workspace_path = os.environ.get('AE_WORKSPACE_PATH')

    if workspace_path == None:
        print('Please define a workspace path:\n')
        print('export AE_WORKSPACE_PATH=/path/to/workspace\n')
        exit(-1)

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name")
    parser.add_argument('--at_step', default=None, required=False)
    parser.add_argument('-with_pred', action="store_true", default=False, help='train network with predicted image') 
    arguments = parser.parse_args()
    train_with_pred = arguments.with_pred
    full_name = arguments.experiment_name.split('/')
    
    experiment_name = full_name.pop()
    experiment_group = full_name.pop() if len(full_name) > 0 else ''
    at_step = arguments.at_step

    cfg_file_path, checkpoint_file, ckpt_dir, dataset_path= u.file_paths_init(workspace_path, experiment_name, experiment_group)
    args = configparser.ConfigParser()
    args.read(cfg_file_path)

    VISABLE_AMOUNT = eval(args.get('Network', 'VISABLE_AMOUNT'))
    with tf.compat.v1.variable_scope(experiment_name):
        training = tf.compat.v1.placeholder(tf.bool, name='training')
        dataset = factory.build_dataset(dataset_path, args)
        dataset.dynamic_mode = True 
        dataset.with_pred = train_with_pred
        queue = factory.build_queue(dataset, args)
        R_encoder = factory.build_encoder(queue.x, args, visable_amount= queue.visable_amount , is_training=training)
        t_encoder = factory.build_encoder(queue.x, args, visable_amount= queue.visable_amount , is_training=training)
        AUXILIARY_MASK = eval(args.get('Network', 'AUXILIARY_MASK'))
        encoder_dir_out = t_encoder.encoder_out
        t_decoder = factory.build_decoder(queue.y, t_encoder, args, is_training=training, mask_target=queue.mask, visable_amount_target = queue.visable_amount)
        R_decoder = factory.build_decoder(queue.rot_y, R_encoder, args, is_training=training, visable_amount_target = queue.visable_amount )

        if AUXILIARY_MASK:
            recon_mask = t_decoder.xmask
        else:
            recon_mask = t_decoder.x
        predictor = factory.build_pose_predictor(
            queue.rot_x, recon_mask, t_encoder.z, R_encoder.z,
            queue.delta_t, dataset, args,
            is_training = training, use_mask = AUXILIARY_MASK, 
            visable_amount = queue.visable_amount , encoder_dir_out = encoder_dir_out
            )
        ae = factory.build_ae(R_encoder, t_encoder, R_decoder, t_decoder, predictor, args, vis_amnt_pred=True)
        codebook = factory.build_codebook(R_encoder, dataset, args)
        saver = tf.compat.v1.train.Saver(save_relative_paths=True)

    batch_size = args.getint('Training', 'BATCH_SIZE')
    model = args.get('Dataset', 'MODEL')

    per_process_gpu_memory_fraction = 0.9
    config = factory.config_GPU(per_process_gpu_memory_fraction)
    with tf.compat.v1.Session(config=config) as sess:

        print(ckpt_dir)
        print('#'*20)

        factory.restore_checkpoint(sess, saver, ckpt_dir, at_step=at_step)

        codebook.update_embedding(sess, batch_size)

        print('Saving new checkoint ..')

        saver.save(sess, checkpoint_file, global_step=ae.global_step)

        print('done')

if __name__ == '__main__':
    main()