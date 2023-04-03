'''
Copyright (c) 2019 [Jia-Yau Shiau]
Code work by Jia-Yau (jiayau.shiau@gmail.com).
--------------------------------------------------
The implementation of 6D rotatiton representation,
based on 

    https://arxiv.org/abs/1812.07035

"On the continuity of rotation representations in neural networks"
Yi Zhou, Connelly Barnes, Jingwan Lu, Jimei Yang, Hao Li.
Conference on Neural Information Processing Systems (NeurIPS) 2019.
'''
import tensorflow as tf
import numpy as np

def tf_rotation6d_to_matrix(r6d):
    """ Compute rotation matrix from 6D rotation representation.
        Implementation base on 
            https://arxiv.org/abs/1812.07035
        [Inputs]
            6D rotation representation (last dimension is 6)
        [Returns]
            flattened rotation matrix (last dimension is 9)
    """
    tensor_shape = r6d.get_shape().as_list()

    if not tensor_shape[-1] == 6:
        raise AttributeError("The last demension of the inputs in tf_rotation6d_to_matrix should be 6, \
            but found tensor with shape {}".format(tensor_shape[-1]))

    with tf.compat.v1.variable_scope('rot6d_to_matrix'):
        r6d   = tf.reshape(r6d, [-1,6])
        x_raw = r6d[:,0:3]
        y_raw = r6d[:,3:6]
    
        x = tf.nn.l2_normalize(x_raw, axis=-1)
        z = tf.linalg.cross(x, y_raw)
        z = tf.nn.l2_normalize(z, axis=-1)
        y = tf.linalg.cross(z, x)

        x = tf.reshape(x, [-1,3,1])
        y = tf.reshape(y, [-1,3,1])
        z = tf.reshape(z, [-1,3,1])
        matrix = tf.concat([x,y,z], axis=-1)

        if len(tensor_shape) == 1:
            matrix = tf.reshape(matrix, [9])
        else:
            matrix = tf.reshape(matrix, [-1, 9])

    return matrix

def tf_matrix_to_rotation6d(mat):
    """ Get 6D rotation representation for rotation matrix.
        Implementation base on 
            https://arxiv.org/abs/1812.07035
        [Inputs]
            flattened rotation matrix (last dimension is 9)
        [Returns]
            6D rotation representation (last dimension is 6)
    """
    tensor_shape = mat.get_shape().as_list()

    if not ((tensor_shape[-1] == 3 and tensor_shape[-2] == 3) or (tensor_shape[-1] == 9)):
        raise AttributeError("The inputs in tf_matrix_to_rotation6d should be [...,9] or [...,3,3], \
            but found tensor with shape {}".format(tensor_shape[-1]))

    with tf.compat.v1.variable_scope('matrix_to_ration_6d'):
        mat = tf.reshape(mat, [-1, 3, 3])
        r6d = tf.concat([mat[...,0], mat[...,1]], axis=-1)

        if len(tensor_shape) == 1:
            r6d = tf.reshape(r6d, [6])

    return r6d

def tf_rotation6d_to_matrix_seq(r6d):
    """ Compute rotation matrix from 6D rotation representation.
        Implementation base on 
            https://arxiv.org/abs/1812.07035
        [Inputs]
            6D rotation representation (last dimension is 6)
        [Returns]
            flattened rotation matrix (last dimension is 9)
    """
    tensor_shape = r6d.get_shape().as_list()

    if not tensor_shape[-1] == 6:
        raise AttributeError("The last demension of the inputs in tf_rotation6d_to_matrix should be 6, \
            but found tensor with shape {}".format(tensor_shape[-1]))
    if len(tensor_shape) == 2:
        r6d = tf.expand_dims(r6d, axis=0)

    tensor_shape = r6d.get_shape().as_list()

    with tf.compat.v1.variable_scope('rot6d_to_matrix'):
        r6d   = tf.reshape(r6d, [-1,tensor_shape[1],6])
        x_raw = r6d[:,:,0:3]
        y_raw = r6d[:,:,3:6]
    
        x = tf.nn.l2_normalize(x_raw, axis=-1)
        z = tf.linalg.cross(x, y_raw)
        z = tf.nn.l2_normalize(z, axis=-1)
        y = tf.linalg.cross(z, x)

        x = tf.reshape(x, [-1, tensor_shape[1],3,1])
        y = tf.reshape(y, [-1, tensor_shape[1],3,1])
        z = tf.reshape(z, [-1, tensor_shape[1],3,1])
        matrix = tf.concat([x,y,z], axis=-1)

        if len(tensor_shape) == 1:
            matrix = tf.reshape(matrix, [9])
        else:
            output_shape = tensor_shape[:-1] + [9]
            matrix = tf.reshape(matrix, output_shape)

    return matrix

def tf_matrix_to_rotation6d_seq(mat):
    """ Get 6D rotation representation for rotation matrix.
        Implementation base on 
            https://arxiv.org/abs/1812.07035
        [Inputs]
            flattened rotation matrix (last dimension is 9)
        [Returns]
            6D rotation representation (last dimension is 6)
    """
    
    tensor_shape = mat.get_shape().as_list()
    if len(tensor_shape) == 3:
        tf.expand_dims(mat, axis=0)

    if not ((tensor_shape[-1] == 3 and tensor_shape[-2] == 3) or (tensor_shape[-1] == 9)):
        raise AttributeError("The inputs in tf_matrix_to_rotation6d should be [...,9] or [...,3,3], \
            but found tensor with shape {}".format(tensor_shape[-1]))

    tensor_shape = mat.get_shape().as_list()
    with tf.compat.v1.variable_scope('matrix_to_ration_6d'):
        mat = tf.reshape(mat, [-1, tensor_shape[1], 3, 3])
        r6d = tf.concat([mat[...,0], mat[...,1]], axis=-1)

        if len(tensor_shape) == 1:
            r6d = tf.reshape(r6d, [6])

    return r6d

def np_matrix_to_rotation6d_seq(mat):
    """ Get 6D rotation representation for rotation matrix.
        Implementation base on 
            https://arxiv.org/abs/1812.07035
        [Inputs]
            flattened rotation matrix (last dimension is 9)
        [Returns]
            6D rotation representation (last dimension is 6)
    """
    
    shape = mat.shape
    if len(shape) == 3:
        np.expand_dims(mat, axis=0)

    if not ((shape[-1] == 3 and shape[-2] == 3) or (shape[-1] == 9)):
        raise AttributeError("The inputs in tf_matrix_to_rotation6d should be [...,9] or [...,3,3], \
            but found tensor with shape {}".format(shape[-1]))

    shape = mat.shape
    mat = np.reshape(mat, [-1, shape[1], 3, 3])
    r6d = np.concatenate([mat[...,0], mat[...,1]], axis=-1)

    if len(shape) == 1:
        r6d = np.reshape(r6d, [6])
    return r6d

def np_rotation6d_to_matrix_seq(r6d):
    """ Compute rotation matrix from 6D rotation representation.
        Implementation base on 
            https://arxiv.org/abs/1812.07035
        [Inputs]
            6D rotation representation (last dimension is 6)
        [Returns]
            flattened rotation matrix (last dimension is 9)
    """
    shape = r6d.shape

    if not shape[-1] == 6:
        raise AttributeError("The last demension of the inputs in np_rotation6d_to_matrix should be 6, \
            but found tensor with shape {}".format(shape[-1]))
    if len(shape) == 2:
        r6d = np.expand_dims(r6d, axis=0)

    shape = r6d.shape

    r6d   = np.reshape(r6d, [-1,shape[1],6])
    x_raw = r6d[:,:,0:3]
    y_raw = r6d[:,:,3:6]
    
    x = x_raw/np.linalg.norm(x_raw, axis=-1, keepdims=True)
    # x = np.nn.l2_normalize(x_raw, axis=-1)
    z = np.cross(x, y_raw)
    z = z/np.linalg.norm(z, axis=-1, keepdims=True)
    # z = np.nn.l2_normalize(z, axis=-1)
    y = np.cross(z, x)

    x = np.reshape(x, [-1, shape[1],3,1])
    y = np.reshape(y, [-1, shape[1],3,1])
    z = np.reshape(z, [-1, shape[1],3,1])
    matrix = np.concatenate([x,y,z], axis=-1)

    if len(shape) == 1:
        matrix = np.reshape(matrix, [9])
    else:
        output_shape = shape[:-1] + (9,)
        matrix = np.reshape(matrix, output_shape)

    return matrix

# batch geodesic loss for rotation matrices
def loss(Rgts,Rps, reduction = 'mean'):
    eps = 1e-6 
    Rds = tf.matmul(tf.transpose(Rgts, perm=[0, 2, 1]),Rps)
    # Rt = tf.reduce_sum(Rds[:,tf.eye(3, dtype=tf.int32)],axis=1) #batch trace
    Rt = tf.reduce_sum(tf.linalg.diag_part(Rds), axis= 1)
    # necessary or it might lead to nans and the likes
    theta = tf.clip_by_value(0.5*(Rt-1), -1+eps, 1-eps)
    loss = tf.math.acos(theta)
    if reduction == 'mean':
        return tf.reduce_mean(loss) 
    else:
        return loss 