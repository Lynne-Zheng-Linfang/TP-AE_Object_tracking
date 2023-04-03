import json
import numpy as np
import os
import rot_6d
import tensorflow as tf
import transform

ori_json_file = '/home/linfang/Documents/Dataset/pose_paths_6.json'
# with open(ori_json_file) as json_file:
#     data = json.load(json_file)
#     print(len(data))
#     for j in range(len(data)):
#         Ps = np.array(data[str(len(data) - j -1 )])
#         Rs = Ps[:,:3,:3]
#         Rs = tf.convert_to_tensor(Rs, dtype=tf.float64)
#         rot_6d_repre = rot_6d.tf_matrix_to_rotation6d(Rs)
#         back_rot = rot_6d.tf_rotation6d_to_matrix(rot_6d_repre)
#         back_rot = tf.reshape(back_rot, (-1,3,3))
#         print(rot_6d.loss(Rs[:6], back_rot[:6]))
#         break
path_num = 20
Rs = np.empty((path_num,5,3,3))
for i in range(path_num):
    for j in range(5):
        Rs[i,j] = transform.random_rotation_matrix()[:3,:3]

Rs_tf = tf.convert_to_tensor(Rs, dtype=tf.float64)
rot_6d_repre_seq = rot_6d.tf_matrix_to_rotation6d_seq(Rs_tf)
# np_rot_6d_repre_seq = rot_6d.np_matrix_to_rotation6d_seq(Rs)
proto_tensor = tf.make_tensor_proto(rot_6d_repre_seq)
np_rot_6d_repre_seq = tf.make_ndarray(proto_tensor)
# for i in range(path_num):
#     if ((rot_6d_repre_seq[i] - np_rot_6d_repre_seq[i])<0.00000001).all():
#         print(True)

back_rot_seq = rot_6d.tf_rotation6d_to_matrix_seq(rot_6d_repre_seq)
np_back_rot_seq = rot_6d.np_rotation6d_to_matrix_seq(np_rot_6d_repre_seq)
proto_tensor = tf.make_tensor_proto(back_rot_seq)
back_rot_seq = tf.make_ndarray(proto_tensor)
for i in range(path_num):
    # back_rot = rot_6d.tf_rotation6d_to_matrix(rot_6d_repre_seq[i])
    if tf.reduce_all((back_rot_seq[i] -np_back_rot_seq[i]) < 0.00000001 ):
        print(True)
    else:
        print(False)
        # print(back_rot, back_rot_seq[i])
# print(rot_6d.loss(Rs[:6], back_rot[:6]))