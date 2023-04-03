# -*- coding: utf-8 -*-

from dataset import Dataset
from my_queue import Queue, CombinerQueue, LSTMQueue
from ae import AE
from encoder import Encoder
from decoder import Decoder
from codebook import Codebook
from predictor import Predictor
from class_predictor import ClassPredictor
# from visable_amount_predictor import VisableAmountEstimator
from siren_encoder import SirenEncoder
from siren_decoder import SirenDecoder
from rotation_autoencoder import RotationAutoEncoder
import hashlib
import shutil
import os
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution() #TODO: Need to be modified later
import tf_slim as slim
# import tensorflow_datasets as tfds
from combiner import Combiner
from lstm_dataset import PriorPoseDataset
from pose_lstm import PriorPoseModule

def build_dataset(dataset_path, args, ycb_real=False, is_training=False, gen=False):
    dataset_args = { k:v for k,v in
        args.items('Dataset') +
        args.items('Paths') +
        args.items('Augmentation')+
        args.items('Queue') +
        args.items('Embedding') +
        args.items('Evaluation') +
        args.items('Background') +
        args.items('Rotation_Dataset')}
    test_ratio = args.getfloat('Training', 'TEST_RATIO')
    dataset = Dataset(dataset_path,test_ratio, args, **dataset_args)
    dataset.siren = eval(args.get('Network', 'SIREN'))
    dataset.batch_size = eval(args.get('Training', 'BATCH_SIZE')) 
    dataset.with_pred = eval(args.get('Training', 'WITH_PRED_IMAGE'))
    dataset.ycb_real = ycb_real
    dataset.is_training = is_training
    print('is_training:', is_training)
    if is_training and not gen:
        dataset.check_real_img_status()
    return dataset

def build_queue(dataset, args):
    NUM_THREADS = args.getint('Queue', 'NUM_THREADS')
    QUEUE_SIZE = args.getint('Queue', 'QUEUE_SIZE')
    BATCH_SIZE = args.getint('Training', 'BATCH_SIZE')
    # queue is a FIFO queue
    queue = Queue(
        dataset,
        NUM_THREADS,
        QUEUE_SIZE,
        BATCH_SIZE,
    )
    return queue

def build_lstm_queue(dataset, args, eval_data = False):
    NUM_THREADS = args.getint('Queue', 'LSTM_NUM_THREADS')
    QUEUE_SIZE = args.getint('Queue', 'LSTM_QUEUE_SIZE')
    BATCH_SIZE = args.getint('Training', 'LSTM_BATCH_SIZE')
    # queue is a FIFO queue
    queue = LSTMQueue(
        dataset,
        NUM_THREADS,
        QUEUE_SIZE,
        BATCH_SIZE,
        test_set_flag = eval_data
    )
    return queue

def build_combiner_queue(dataset, args):
    NUM_THREADS = args.getint('Queue', 'NUM_THREADS')
    QUEUE_SIZE = args.getint('Queue', 'QUEUE_SIZE')
    BATCH_SIZE = args.getint('Training', 'BATCH_SIZE')
    # queue is a FIFO queue
    queue = CombinerQueue(
        dataset,
        NUM_THREADS,
        QUEUE_SIZE,
        BATCH_SIZE,
    )
    return queue

def build_test_queue(dataset, args, number_threads = 3):
    NUM_THREADS = min(args.getint('Queue', 'NUM_THREADS'), number_threads)
    QUEUE_SIZE = args.getint('Queue', 'QUEUE_SIZE')
    BATCH_SIZE = args.getint('Training', 'BATCH_SIZE')
    # queue is a FIFO queue
    queue = Queue(
        dataset,
        NUM_THREADS,
        QUEUE_SIZE,
        BATCH_SIZE,
        test_set_flag=True
    )
    return queue


def build_encoder(x, args, visable_amount = None, is_training=False, test_x= None):
    LATENT_SPACE_SIZE = args.getint('Network', 'LATENT_SPACE_SIZE')
    NUM_FILTER = eval(args.get('Network', 'NUM_FILTER'))
    KERNEL_SIZE_ENCODER = args.getint('Network', 'KERNEL_SIZE_ENCODER')
    STRIDES = eval(args.get('Network', 'STRIDES'))
    BATCH_NORM = args.getboolean('Network', 'BATCH_NORMALIZATION')
    SIREN = eval(args.get('Network', 'SIREN'))
    DROP_OUT = args.getboolean('Network', 'DROP_OUT')
    if visable_amount is not None:
        VIS_AMOUNT_NET =  eval(args.get('Network', 'VIS_AMNT_NET'))
    else: 
        VIS_AMOUNT_NET = None
    if SIREN:
        encoder = SirenEncoder(
            x,
            LATENT_SPACE_SIZE,
            NUM_FILTER,
            KERNEL_SIZE_ENCODER,
            STRIDES,
            BATCH_NORM,
            is_training=is_training,
            visable_amount = visable_amount,
            visable_amount_net = VIS_AMOUNT_NET,
            drop_out = DROP_OUT,
            test_x = test_x
        )
    else:
        encoder = Encoder(
            x,
            LATENT_SPACE_SIZE,
            NUM_FILTER,
            KERNEL_SIZE_ENCODER,
            STRIDES,
            BATCH_NORM,
            is_training=is_training,
            visable_amount = visable_amount,
            visable_amount_net = VIS_AMOUNT_NET,
            drop_out = DROP_OUT,
            test_x = test_x
        )
    return encoder

def build_decoder(reconstruction_target, encoder, args, is_training=False, mask_target = None,  visable_amount_target = None, t_decode = False, phase_2=False):
    NUM_FILTER = eval(args.get('Network', 'NUM_FILTER'))
    KERNEL_SIZE_DECODER = args.getint('Network', 'KERNEL_SIZE_DECODER')
    STRIDES = eval(args.get('Network', 'STRIDES'))
    LOSS = args.get('Network', 'LOSS')
    BOOTSTRAP_RATIO = args.getint('Network', 'BOOTSTRAP_RATIO')
    AUXILIARY_MASK = args.getboolean('Network', 'AUXILIARY_MASK')
    DROP_OUT = args.getboolean('Network', 'DROP_OUT')
    BATCH_NORM = args.getboolean('Network', 'BATCH_NORMALIZATION')
    try:
        GRAY_BG = eval(args.get('Background', 'GRAY_BG'))
    except:
        GRAY_BG = False
    decoder = Decoder(
        reconstruction_target,
        encoder.translation_z if t_decode else encoder.z,
        list( reversed(NUM_FILTER) ),
        KERNEL_SIZE_DECODER,
        list( reversed(STRIDES) ),
        LOSS,
        BOOTSTRAP_RATIO,
        AUXILIARY_MASK if t_decode else False,
        BATCH_NORM,
        is_training=is_training,
        mask_target = mask_target,
        drop_out = DROP_OUT,
        visable_amount_target =  visable_amount_target,
        phase_2 = phase_2,
        gray_bg = GRAY_BG
    )
    return decoder
def build_prior_pose_dataset(args):
    dataset_args = { k:v for k,v in
        args.items('Dataset') +
        args.items('Paths') +
        args.items('Training') +
        args.items('Queue') +
        args.items('Evaluation')}
    dataset = PriorPoseDataset(**dataset_args)
    return dataset

def build_prior_pose_estimator_architecture(args, experiment_name, is_training = False):
    with tf.compat.v1.variable_scope('prior_pose'):
        dataset = build_prior_pose_dataset(args)
        queue = build_lstm_queue(dataset, args) if is_training else None
        test_queue = build_lstm_queue(dataset, args, eval_data = True) if is_training else None
        training = tf.compat.v1.placeholder(tf.bool, name='training')
        x_shape = [None,] + list(dataset.x_shape)
        input_x = queue.x if is_training else tf.compat.v1.placeholder(tf.float32, x_shape,name='input_x')
        target_y = queue.y if is_training else tf.compat.v1.placeholder(tf.float32, [None,] + list(dataset.y_shape))
        prior_pose_module = build_prior_pose_module(args, target_y, input_x, is_training= is_training) 
        train_op = build_train_op(prior_pose_module, args) if is_training else None
        saver = tf.compat.v1.train.Saver(save_relative_paths=True)
    return (dataset, prior_pose_module, training, saver, input_x, train_op, queue, test_queue)

def build_prior_pose_module(args, pose_target, input_seqs, is_training = False):
    LOWER_BOUND = args.getfloat('Dataset', 'LSTM_T_LOWER_BOUND')
    UPPER_BOUND = args.getfloat('Dataset', 'LSTM_T_UPPER_BOUND')
    TARGET_LENGTH = args.getint('Dataset', 'LSTM_TARGET_LENGTH')
    LSTM_NET = eval(args.get('Network', 'LSTM_NET'))
    BATCH_NORM = args.getboolean('Network', 'BATCH_NORMALIZATION')
    DROP_OUT = args.getboolean('Network', 'LSTM_DROP_OUT')
    TRANS_LOSS_SCALE = args.getfloat('Training', 'TRANS_LOSS_SCALE')
    
    prior_pose_module = PriorPoseModule(
        pose_target, 
        input_seqs, 
        LSTM_NET,
        is_training=is_training, 
        lower_bound = LOWER_BOUND, 
        upper_bound = UPPER_BOUND,
        drop_out = DROP_OUT,
        batch_norm = BATCH_NORM,
        trans_loss_scale = TRANS_LOSS_SCALE
        )
    return prior_pose_module 

def build_ae(R_encoder, t_encoder, R_decoder, t_decoder, predictor, args, vis_amnt_pred=False):
    NORM_REGULARIZE = args.getfloat('Network', 'NORM_REGULARIZE')
    VARIATIONAL = args.getfloat('Network', 'VARIATIONAL')
    ae = AE(R_encoder, t_encoder, R_decoder, t_decoder, predictor, NORM_REGULARIZE, VARIATIONAL, vis_amnt_pred = vis_amnt_pred)
    return ae

def build_train_op(ae, args, phase_2 = False):
    LEARNING_RATE = args.getfloat('Training', 'LEARNING_RATE')
    OPTIMIZER_NAME = args.get('Training', 'OPTIMIZER')
    import tensorflow as tf
    optimizer = eval('tf.compat.v1.train.{}Optimizer'.format(OPTIMIZER_NAME))
    optim = optimizer(LEARNING_RATE)

    train_op = slim.learning.create_train_op(ae.loss, optim, global_step=ae.global_step)

    return train_op

def build_codebook(encoder, dataset, args):
    embed_bb = args.getboolean('Embedding', 'EMBED_BB')
    codebook = Codebook(encoder, dataset, embed_bb)
    return codebook

def build_visable_amount_estimator(args, image_feature, encoder_feature, target, is_training = False, phase_2 = False):
    VIS_AMNT_NET = eval(args.get('Network', 'VIS_AMNT_NET'))
    BATCH_NORM = args.getboolean('Network', 'BATCH_NORMALIZATION')
    DROP_OUT = args.getboolean('Network', 'DROP_OUT')
    vis_amnt_predictor = VisableAmountEstimator(
        image_feature, 
        encoder_feature, 
        target, 
        VIS_AMNT_NET, 
        BATCH_NORM, 
        DROP_OUT,
        is_training=is_training, 
        phase_2 = phase_2
        ) 
    return vis_amnt_predictor

def build_codebook_from_name(experiment_name, experiment_group='', return_dataset=False, return_decoder = False, return_vis_amnt_predictor = False):
    import os
    import configparser
    workspace_path = os.environ.get('AE_WORKSPACE_PATH')

    if workspace_path == None:
        print('Please define a workspace path:\n')
        print('export AE_WORKSPACE_PATH=/path/to/workspace\n')
        exit(-1)

    from . import utils as u
    import tensorflow as tf

    log_dir = u.get_log_dir(workspace_path, experiment_name, experiment_group)
    checkpoint_file = u.get_checkpoint_basefilename(log_dir)
    cfg_file_path = u.get_train_config_exp_file_path(log_dir, experiment_name)
    dataset_path = u.get_dataset_path(workspace_path)

    if os.path.exists(cfg_file_path):
        args = configparser.ConfigParser()
        args.read(cfg_file_path)
    else:
        print('ERROR: Config File not found: ', cfg_file_path)
        exit()

    with tf.compat.v1.variable_scope(experiment_name):
        dataset = build_dataset(dataset_path, args)
        x = tf.compat.v1.placeholder(tf.float32, [None,] + list(dataset.shape))
        encoder = build_encoder(x, args)
        codebook = build_codebook(encoder, dataset, args)
        if return_decoder:
            reconst_target = tf.compat.v1.placeholder(tf.float32, [None,] + list(dataset.shape))
            decoder = build_decoder(reconst_target, encoder, args)

    if return_dataset:
        if return_decoder:
            return codebook, dataset, decoder
        else:
            return codebook, dataset
    else:
        return codebook

def build_inference_architecture(experiment_name, dataset_path, args, use_combiner_queue= False):
    import os
    import configparser

    with tf.compat.v1.variable_scope(experiment_name):
        training = tf.compat.v1.placeholder(tf.bool, name='training')
        dataset = build_dataset(dataset_path, args)
        #TODO: need to be modified here
        dataset.dynamic_mode = True 
        if use_combiner_queue:
            queue = build_combiner_queue(dataset, args)
            x = queue.x
        else:
            queue = None
            x_shape = [None,] + list(dataset.shape)
            x = tf.compat.v1.placeholder(tf.uint8, x_shape,name='input_x')
            no_rot_x = tf.compat.v1.placeholder(tf.uint8, x_shape,name='no_rot_x')
        mask_target = tf.compat.v1.placeholder(tf.bool, [None,] + list(dataset.mask_shape))
        vis_amount_target = tf.compat.v1.placeholder(tf.float32, [None,] + list(dataset.visable_amount_shape)) 
        R_encoder = build_encoder(x, args, visable_amount=vis_amount_target , is_training=training)
        t_encoder = build_encoder(x, args, visable_amount= vis_amount_target , is_training=training)
        t_reconst_target = tf.compat.v1.placeholder(tf.uint8, [None,] + list(dataset.shape))
        encoder_dir_out = t_encoder.encoder_out
        t_decoder = build_decoder(t_reconst_target, t_encoder, args, is_training=training, mask_target=mask_target, visable_amount_target = vis_amount_target)
        R_reconst_target = tf.compat.v1.placeholder(tf.uint8, [None,] + list(dataset.shape))
        R_decoder = build_decoder(R_reconst_target, R_encoder, args, is_training=training, visable_amount_target = vis_amount_target)

        AUXILIARY_MASK = eval(args.get('Network', 'AUXILIARY_MASK'))
        if AUXILIARY_MASK:
            recon_mask = t_decoder.xmask
        else:
            recon_mask = t_decoder.x
        delta_t = tf.compat.v1.placeholder(tf.float32,  [None,]+list(dataset.t_shape))
        predictor = build_pose_predictor(
            no_rot_x, recon_mask, t_encoder.z, R_encoder.z, 
            delta_t, dataset, args,
            is_training = training,
            use_mask = AUXILIARY_MASK, 
            visable_amount = vis_amount_target, encoder_dir_out = encoder_dir_out
            )
        codebook = build_codebook(R_encoder, dataset, args)

    if use_combiner_queue:
        return (codebook, predictor, R_encoder, R_decoder, t_encoder, t_decoder, dataset, x, queue)
    else:
        return (codebook, predictor, R_encoder, R_decoder, t_encoder, t_decoder, dataset, x, no_rot_x)

def build_train_architecture(experiment_name, dataset_path, args, is_training= False, phase_2 = False, ycb_real=False, gen= False):
    with tf.compat.v1.variable_scope(experiment_name):
        dataset = build_dataset(dataset_path, args, ycb_real=ycb_real, is_training=is_training, gen=gen)
        dataset.dynamic_mode = True 
        queue = build_queue(dataset, args)
        training = tf.compat.v1.placeholder(tf.bool, name='training')
        R_encoder = build_encoder(queue.x, args, visable_amount= queue.visable_amount , is_training=training)
        t_encoder = build_encoder(queue.x, args, visable_amount= queue.visable_amount , is_training=training)
        AUXILIARY_MASK = eval(args.get('Network', 'AUXILIARY_MASK'))
        encoder_dir_out = t_encoder.encoder_out
        t_decoder = build_decoder(queue.y, t_encoder, args, is_training=training, mask_target=queue.mask, visable_amount_target = queue.visable_amount, phase_2 = phase_2)
        R_decoder = build_decoder(queue.rot_y, R_encoder, args, is_training=training, visable_amount_target = queue.visable_amount, phase_2 = phase_2)
        predictor = build_pose_predictor(
            queue.rot_x, t_decoder.x, t_encoder.z, R_encoder.z, 
            queue.delta_t, dataset, args,
            is_training = training, use_mask = AUXILIARY_MASK, 
            visable_amount = queue.visable_amount , encoder_dir_out = encoder_dir_out, phase_2= phase_2
            )
        ae = build_ae(R_encoder, t_encoder, R_decoder, t_decoder, predictor, args, vis_amnt_pred=True)
        codebook = build_codebook(R_encoder, dataset, args)
        train_op = build_train_op(ae, args, phase_2=phase_2)
        saver = tf.compat.v1.train.Saver(save_relative_paths=True)
    return (dataset, queue, training, codebook, predictor, R_encoder, R_decoder, t_encoder, t_decoder, ae, train_op, saver)


def build_combiner(input_x, target, args, is_training = False):
    NET_STRUCTURE = eval(args.get('Network', 'COMBINER_NET'))
    DROP_OUT = args.getboolean('Network', 'DROP_OUT')
    BATCH_NORM = args.getboolean('Network', 'BATCH_NORMALIZATION')
    combiner = Combiner(input_x, target, NET_STRUCTURE, BATCH_NORM, is_training=is_training, drop_out = DROP_OUT)
    return combiner
    
def build_combiner_architecture(experiment_name, dataset_path, args, is_training= False):
    OUTPUT_SIZE = args.getint('Network', 'LATENT_SPACE_SIZE')
    input_shape = [None,]+[3*(OUTPUT_SIZE+1),]
    output_shape = [None,]+[OUTPUT_SIZE]
    with tf.compat.v1.variable_scope(experiment_name):
        training = tf.compat.v1.placeholder(tf.bool, name='training')
        input_x = tf.compat.v1.placeholder(tf.float32, shape=input_shape, name='combiner_input')
        target = tf.compat.v1.placeholder(tf.float32, shape=output_shape, name='combiner_output')
        combiner = build_combiner(input_x, target, args, is_training=is_training)
        train_op = build_train_op(combiner, args)
        saver = tf.compat.v1.train.Saver(save_relative_paths=True)
    return (training, combiner, train_op, saver, input_x, target)

def build_rotation_autoencoder(args, input_img, target, visible_amount, is_training=False, phase_2 = False):
    rot_autoencoder = RotationAutoEncoder(args, input_img, target, visible_amount, is_training=is_training, phase_2 = phase_2)
    return rot_autoencoder 

def build_rotation_estimator_architecture(experiment_name, dataset_path, args, is_training= False, phase_2 = False):
    OUTPUT_SIZE = args.getint('Network', 'LATENT_SPACE_SIZE')
    input_shape = [None,]+[3*(OUTPUT_SIZE+1),]
    output_shape = [None,]+[OUTPUT_SIZE]
    with tf.compat.v1.variable_scope(experiment_name):
        dataset = build_dataset(dataset_path, args)
        dataset.dynamic_mode = True 
        training = tf.compat.v1.placeholder(tf.bool, name='training')
        queue = build_queue(dataset, args) if is_training else None
        x_shape = [None,] + list(dataset.shape)
        input_x = queue.x if is_training else tf.compat.v1.placeholder(tf.uint8, x_shape,name='input_x')
        target = queue.y if is_training else tf.compat.v1.placeholder(tf.uint8, [None,] + list(dataset.shape))
        visible_amount = queue.visable_amount if is_training else tf.compat.v1.placeholder(tf.float32, [None,] + list(dataset.visable_amount_shape)) 
        rot_autoencoder = RotationAutoEncoder(args, input_x, target, visible_amount, is_training=training, phase_2 = phase_2)
        codebook = build_codebook(rot_autoencoder.encoder, dataset, args)
        train_op = build_train_op(rot_autoencoder, args, phase_2=phase_2) if is_training else None
        saver = tf.compat.v1.train.Saver(save_relative_paths=True)
    return (dataset, rot_autoencoder, training, saver, input_x, target, train_op, queue)


def build_predictor_inference_architecture(experiment_name, experiment_group='', return_dataset=False, class_t = False, t_distr = False, use_mask = False, output_visable_amount = False):
    print('t_distr:', t_distr)
    print('output_visable_amount:', output_visable_amount)
    import os
    import configparser
    workspace_path = os.environ.get('AE_WORKSPACE_PATH')

    if workspace_path == None:
        print('Please define a workspace path:\n')
        print('export AE_WORKSPACE_PATH=/path/to/workspace\n')
        exit(-1)

    from . import utils as u
    import tensorflow as tf

    log_dir = u.get_log_dir(workspace_path, experiment_name, experiment_group)
    checkpoint_file = u.get_checkpoint_basefilename(log_dir)
    cfg_file_path = u.get_train_config_exp_file_path(log_dir, experiment_name)
    dataset_path = u.get_dataset_path(workspace_path)

    if os.path.exists(cfg_file_path):
        args = configparser.ConfigParser()
        args.read(cfg_file_path)
    else:
        print('ERROR: Config File not found: ', cfg_file_path)
        exit()

    with tf.compat.v1.variable_scope(experiment_name):
        dataset = build_dataset(dataset_path, args)
        x = tf.compat.v1.placeholder(tf.float32, [None,] + list(dataset.shape))
        encoder = build_encoder(x, args)
        reconst_target = tf.compat.v1.placeholder(tf.float32, [None,] + list(dataset.shape))
        mask_target = tf.compat.v1.placeholder(tf.float32, [None,] + list(dataset.mask_shape))
        vis_amount_target = tf.compat.v1.placeholder(tf.float32, [None,] + list(dataset.visable_amount_shape)) 
        if output_visable_amount:
            encoder_dir_out = encoder.encoder_out
            visable_amount = vis_amount_target
        else:
            encoder_dir_out = None
            visable_amount = None
        decoder = build_decoder(reconst_target, encoder, args, mask_target = mask_target,  visable_amount_target = visable_amount)
        t_target = tf.compat.v1.placeholder(tf.float32, [None,] + list(dataset.t_shape))
        if use_mask:
            seg_mask = decoder.xmask
        else:
            seg_mask = decoder.x
        if not class_t:
            predictor = build_pose_predictor(
                encoder.x, 
                seg_mask, 
                encoder.z, 
                t_target, 
                dataset, 
                args, 
                t_distr = t_distr,         
                use_mask = use_mask,
                visable_amount = visable_amount,
                encoder_dir_out = encoder_dir_out
            )
        else:
            predictor = build_t_class_predictor(encoder.x, decoder.x, encoder.z, t_target, args)

    if return_dataset:
        return predictor, encoder, decoder, dataset
    else:
        return predictor, encoder, decoder

def restore_checkpoint(session, saver, ckpt_dir, at_step=None):
    import tensorflow as tf
    import os

    chkpt = tf.train.get_checkpoint_state(ckpt_dir)

    if chkpt and chkpt.model_checkpoint_path:
        if at_step is None:
            saver.restore(session, chkpt.model_checkpoint_path)
        else:
            for ckpt_path in chkpt.all_model_checkpoint_paths:

                if str(at_step) in str(ckpt_path):
                    saver.restore(session, ckpt_path)
                    print('restoring' , os.path.basename(ckpt_path))
    else:
        print('No checkpoint found. Expected one in:\n')
        print('{}\n'.format(ckpt_dir))
        exit(-1)

def build_pose_predictor(ori_img, reconstr_img, latent_code, R_latent_code, t_target, dataset, args, is_training = False, fine_tune = False, t_distr = False, use_mask= False, visable_amount = None, encoder_dir_out = None, phase_2 = False):
    LATENT_SPACE_SIZE = args.getint('Network', 'LATENT_SPACE_SIZE')
    NUM_FILTER = eval(args.get('Network', 'NUM_FILTER'))
    KERNEL_SIZE_ENCODER = args.getint('Network', 'KERNEL_SIZE_ENCODER')
    STRIDES = eval(args.get('Network', 'STRIDES'))
    BATCH_NORM = args.getboolean('Network', 'BATCH_NORMALIZATION')
    DROP_OUT = args.getboolean('Network', 'DROP_OUT')
    DELTA_T_NET = eval(args.get('Network', 'DELTA_T_NET'))
    PRED_IMAGE_NET = eval(args.get('Network', 'PREDICTION_IMAGE_NET_FILTER'))
    max_translation_shit= args.getfloat('Dataset', 'MAX_TRANSLATION_SHIFT')
    try:
        GRAY_BG = eval(args.get('Background', 'GRAY_BG'))
    except:
        GRAY_BG = False

    scale_factor = 2*dataset.max_delta_t_shift/(dataset.bbox_enlarge_level*dataset.target_obj_diameter)
    predictor = Predictor(
        ori_img,
        reconstr_img,
        latent_code, 
        R_latent_code,
        t_target, 
        NUM_FILTER, 
        KERNEL_SIZE_ENCODER, 
        STRIDES, 
        BATCH_NORM, 
        DROP_OUT,
        PRED_IMAGE_NET,
        DELTA_T_NET,
        scale_factor,
        max_translation_shit,
        is_training = is_training,
        fine_tune = fine_tune,
        t_distr = t_distr,
        use_mask = use_mask,
        visable_amount = visable_amount,
        encoder_dir_out = encoder_dir_out,
        phase_2 = phase_2,
        gray_bg = GRAY_BG
    )

    return predictor


def build_t_class_predictor(ori_img, reconstr_img, latent_code, t_target, args, is_training = False, fine_tune = False):
    LATENT_SPACE_SIZE = args.getint('Network', 'LATENT_SPACE_SIZE')
    NUM_FILTER = eval(args.get('Network', 'NUM_FILTER'))
    KERNEL_SIZE_ENCODER = args.getint('Network', 'KERNEL_SIZE_ENCODER')
    STRIDES = eval(args.get('Network', 'STRIDES'))
    BATCH_NORM = args.getboolean('Network', 'BATCH_NORMALIZATION')
    DROP_OUT = args.getboolean('Network', 'DROP_OUT')
    DELTA_T_NET = eval(args.get('Network', 'DELTA_T_NET'))
    PRED_IMAGE_NET = eval(args.get('Network', 'PREDICTION_IMAGE_NET_FILTER'))
    CLASSIFICATION_T_NET = eval(args.get('Network', 'CLASSIFICATION_T_NET'))
    T_CLASS_NUM = args.getint('Embedding', 'T_CLASS_NUM')
    predictor = ClassPredictor(
        ori_img,
        reconstr_img,
        latent_code, 
        t_target, 
        NUM_FILTER, 
        KERNEL_SIZE_ENCODER, 
        STRIDES, 
        BATCH_NORM, 
        DROP_OUT,
        PRED_IMAGE_NET,
        CLASSIFICATION_T_NET,
        T_CLASS_NUM,
        is_training = is_training,
        fine_tune = fine_tune
    )

    return predictor

def config_GPU(per_process_gpu_memory_fraction):
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction = per_process_gpu_memory_fraction)
    config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allow_growth = True
    return config

