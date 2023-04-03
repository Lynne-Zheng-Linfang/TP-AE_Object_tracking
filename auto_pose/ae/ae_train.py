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

import ae_factory as factory
import utils as u
import faulthandler
faulthandler.enable()

def main():
    workspace_path = os.environ.get('AE_WORKSPACE_PATH')

    if workspace_path is None:
        print('Please define a workspace path:\n')
        print('export AE_WORKSPACE_PATH=/path/to/workspace\n')
        exit(-1)

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name")
    parser.add_argument("-d", action='store_true', default=False)
    parser.add_argument("-ycb_real", action='store_true', default=False)
    parser.add_argument("-gen", action='store_true', default=False)
    parser.add_argument("-r", action="store_true", default=False)
    parser.add_argument("-fine", action="store_true", default=False)
    parser.add_argument("-p2", action="store_true", default=False)
    parser.add_argument("-tdis", action="store_true", default=False)
    parser.add_argument('-cl', action="store_true", default=False, help='delta t classification mode') 
    parser.add_argument('-dynamic', action="store_true", default=False, help='generate train data online') 
    parser.add_argument('-mask', action="store_true", default=False, help='delta t classification mode') 
    parser.add_argument('-tonly', action="store_true", default=False, help='translation only mode') 
    parser.add_argument('-with_pred', action="store_true", default=False, help='train network with predicted image') 
    arguments = parser.parse_args()

    full_name = arguments.experiment_name.split('/')

    experiment_name = full_name.pop()
    experiment_group = full_name.pop() if len(full_name) > 0 else ''

    debug_mode = arguments.d
    generate_data = arguments.gen
    rotation_only_mode = arguments.r
    fine_tune_mode= arguments.fine
    classication_mode = arguments.cl
    t_distribution_mode = arguments.tdis
    mask_check = arguments.mask
    dynamic_mode = arguments.dynamic
    train_with_pred = arguments.with_pred
    phase_2 = arguments.p2
    ycb_real = arguments.ycb_real

    cfg_file_path, checkpoint_file, ckpt_dir, train_fig_dir, dataset_path, test_summary_dir = u.file_paths_init(workspace_path, experiment_name, experiment_group, is_training= True)

    args = configparser.ConfigParser()
    args.read(cfg_file_path)

    SIREN = eval(args.get('Network', 'SIREN'))

    dataset, queue, training, codebook, predictor, R_encoder, R_decoder, t_encoder, t_decoder, ae, train_op, saver = factory.build_train_architecture(experiment_name, dataset_path, args, is_training=True, phase_2=phase_2, ycb_real = ycb_real, gen= generate_data)

    num_iter = args.getint('Training', 'NUM_ITER') 
    if debug_mode:
        num_iter = 100000
    elif not phase_2:
        num_iter = int(num_iter/10)

    save_interval = args.getint('Training', 'SAVE_INTERVAL')
    model_type = args.get('Dataset', 'MODEL')

    print('class_t_mode:', classication_mode)
    dataset.load_augmentation_images()
   
    if generate_data:
        print('finished generating augmentation training data for ' + experiment_name)
        print('exiting...')
        exit()
    # if gentle_stop[0]:
    #     exit(-1)

    if not phase_2:
        welcoming_message = 'Phase_1 Training: '
    else:
        welcoming_message = 'Phase_2 Training: '

    bar = u.progressbar_init(welcoming_message, num_iter)

    per_process_gpu_memory_fraction = 0.9
    config = factory.config_GPU(per_process_gpu_memory_fraction)
    with tf.compat.v1.Session(config=config) as sess:

        chkpt = tf.train.get_checkpoint_state(ckpt_dir)
        if chkpt and chkpt.model_checkpoint_path:
            saver.restore(sess, chkpt.model_checkpoint_path)
        else:
            sess.run(tf.compat.v1.global_variables_initializer())

        merged_loss_summary = tf.compat.v1.summary.merge_all()
        summary_writer = tf.compat.v1.summary.FileWriter(ckpt_dir, sess.graph)

        gentle_stop = np.array((1,), dtype=np.bool)
        gentle_stop[0] = False
        def on_ctrl_c(signal, frame):
            gentle_stop[0] = True
        signal.signal(signal.SIGINT, on_ctrl_c)
        queue.start(sess)

        if not debug_mode:
            print('Training with %s model' % args.get('Dataset','MODEL'), os.path.basename(args.get('Paths','MODEL_PATH')))
            bar.start()

        for i in range(ae.global_step.eval(), num_iter):
            if not debug_mode:
                sess.run(train_op, feed_dict={training: True})
                if i % 50 == 0:
                    loss = sess.run(merged_loss_summary)
                    summary_writer.add_summary(loss, i)
                bar.update(i)

                if ((i+1) % save_interval == 0):
                    saver.save(sess, checkpoint_file, global_step=ae.global_step)
 
                    test_x, test_y, test_rot_x, test_rot_y, test_mask, test_delta_t, test_vis_amount = sess.run([queue.x,queue.y, queue.rot_x, queue.rot_y, queue.mask, queue.delta_t, queue.visable_amount])
                    feed_dict = {
                        queue.x:test_x, 
                        queue.y: test_y, 
                        queue.rot_x: test_rot_x, 
                        queue.rot_y: test_rot_y,
                        queue.mask: test_mask,
                        queue.delta_t: test_delta_t,
                        queue.visable_amount: test_vis_amount,
                        training: False
                        }
                    t_reconstr_train = sess.run(t_decoder.x,feed_dict=feed_dict)*255
                    R_reconstr_train = sess.run(R_decoder.x,feed_dict=feed_dict)*255
                    predictor_x = sess.run(predictor.img_net_input, feed_dict=feed_dict)*255
                    
                    show_x = test_x
                    show_y = test_y
                    this_rot_y = test_rot_y
                    this_rot_x = test_rot_x
                    train_rgb_imgs = np.hstack(( u.tiles(show_x[:,:,:,0:3], 4, 4), u.tiles(predictor_x[:,:,:,0:3], 4,4), u.tiles(t_reconstr_train[:,:,:,0:3], 4,4),u.tiles(show_y[:,:,:,0:3], 4, 4)))
                    train_depth_imgs = np.hstack(( u.tiles(show_x[:,:,:,3:6], 4, 4), u.tiles(predictor_x[:,:,:,3:6], 4,4), u.tiles(t_reconstr_train[:,:,:,3:6], 4,4),u.tiles(show_y[:,:,:,3:6], 4, 4)))
                    cv2.imwrite(os.path.join(train_fig_dir,'translation_training_rgb_images_%s.png' % i), train_rgb_imgs.astype(np.uint8))
                    cv2.imwrite(os.path.join(train_fig_dir,'translation_training_depth_images_%s.png' % i), train_depth_imgs.astype(np.uint8))

                    train_rgb_imgs = np.hstack(( u.tiles(this_rot_x[:,:,:,0:3], 4, 4), u.tiles(predictor_x[:,:,:,0:3], 4,4), u.tiles(R_reconstr_train[:,:,:,0:3], 4,4),u.tiles(this_rot_y[:,:,:,0:3], 4, 4)))
                    train_depth_imgs = np.hstack(( u.tiles(this_rot_x[:,:,:,3:6], 4, 4), u.tiles(predictor_x[:,:,:,3:6], 4,4), u.tiles(R_reconstr_train[:,:,:,3:6], 4,4),u.tiles(this_rot_y[:,:,:,3:6], 4, 4)))
                    cv2.imwrite(os.path.join(train_fig_dir,'rotation_training_rgb_images_%s.png' % i), train_rgb_imgs.astype(np.uint8))
                    cv2.imwrite(os.path.join(train_fig_dir,'rotation_training_depth_images_%s.png' % i), train_depth_imgs.astype(np.uint8))

            else:
                test_x, test_y, test_rot_x, test_rot_y, test_mask, test_delta_t, test_vis_amount = sess.run([queue.x,queue.y, queue.rot_x, queue.rot_y, queue.mask, queue.delta_t, queue.visable_amount])
                feed_dict = {
                    queue.x:test_x, 
                    queue.y: test_y, 
                    queue.rot_x: test_rot_x, 
                    queue.rot_y: test_rot_y,
                    queue.mask: test_mask,
                    queue.delta_t: test_delta_t,
                    queue.visable_amount: test_vis_amount,
                    training: False
                    }
                    
                t_reconstr_train = sess.run(t_decoder.x,feed_dict=feed_dict)
                R_reconstr_train = sess.run(R_decoder.x,feed_dict=feed_dict)
                this_x = test_x / 255
                this_y = test_y /255
                this_rot_y = test_rot_y/255
                this_rot_x = test_rot_x/255

                cv2.imshow('sample rgb_batch',
                    np.hstack((
                        u.tiles(this_x[:,:,:,0:3], 3, 3), 
                        u.tiles(this_rot_x[:,:,:,0:3], 3, 3), 
                        u.tiles(t_reconstr_train[:,:,:,0:3], 3, 3),
                        u.tiles(this_y[:,:,:,0:3], 3, 3), 
                        u.tiles(R_reconstr_train[:,:,:,0:3], 3, 3),
                        u.tiles(this_rot_y[:,:,:,0:3], 3, 3)
                        )))
                cv2.imshow('sample depth_batch', np.hstack(( 
                    u.tiles(this_x[:,:,:,3:6], 3, 3), 
                    u.tiles(this_rot_x[:,:,:,3:6], 3, 3), 
                    u.tiles(t_reconstr_train[:,:,:,3:6], 3,3),
                    u.tiles(this_y[:,:,:,3:6], 3, 3),
                    u.tiles(R_reconstr_train[:,:,:,3:6], 3,3),
                    u.tiles(this_rot_y[:,:,:,3:6], 3, 3)
                    )))
                
                cv2.imshow('sample mask batch', u.tiles(test_mask[:,:,:].astype(np.float32), 3, 3))
                k = cv2.waitKey(10000)

            if gentle_stop[0]:
                break

        if not debug_mode:
            saver.save(sess, checkpoint_file, global_step=ae.global_step)
        queue.stop(sess)
        # test_queue.stop(sess)
        summary_writer.close()
        # test_summary_writer.close()
        if not debug_mode:
            bar.finish()
        if not gentle_stop[0] and not debug_mode:
            if not phase_2:
                print('Phase_1 training finished.')
                print('To start phase_2 training run:')
                print('ae_train '+ experiment_group+'/'+experiment_name + ' -dynamic -p2')
            else:
                print('Phase_2 training finished.')
                print('To create the embedding run:\n')
                print('ae_embed '+experiment_group+'/'+experiment_name)
if __name__ == '__main__':
    main()
