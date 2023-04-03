import os
import glob
from pytless import inout
from collections import Counter
import argparse

parser = argparse.ArgumentParser()
# parser.add_argument("experiment_group")
parser.add_argument("-v", type=float, default = 0.5, help='re-initial visible amount')
parser.add_argument("-len", type=int, default = 15, help='length for prediction')
parser.add_argument("-GRU", action='store_true', default = False)
parser.add_argument("-cosy", action='store_true', default = False)
# parser.add_argument("-inst", type=int,  help='instance_id')
# parser.add_argument("-scene", type=int, help='scene_id')
parser.add_argument("-r_conf", action='store_true', default=False)

arguments = parser.parse_args()
# full_name = arguments.experiment_group

reinit_v = arguments.v
pred_len = arguments.len
GRU = arguments.GRU
R_pass =  arguments.r_conf
USE_COSY_INIT = arguments.cosy

gt_files = glob.glob('/home/linfang/Documents/Dataset/T_LESS/tless_primesense/test_primesense/*/gt.yml')
gt_files.sort()

for scene_id, gt_file in enumerate(gt_files):
    # if (scene_id + 1) <= 3:
    #     continue
    if (scene_id +1) not in [8]:
        continue
    gt_info = inout.load_gt(gt_file)
    obj_num = len(gt_info[0])
    obj_list = []
    for i in range(obj_num):
        obj_list.append(int(gt_info[0][i]['obj_id']))
    obj_dict = dict(Counter(obj_list))
    for obj_id in obj_dict.keys():
        if obj_id not in [24]:
            continue
        if obj_id in [6,25,30]:
            experiment_name = 'obj_{}_no_rot_in_predictor_w_h_sep_rot_crop_in_render_position'.format(obj_id)
        else:
            experiment_name = 'obj_' + str(obj_id).zfill(2)

        total_instance_num = obj_dict[obj_id]
        for inst_id in range(total_instance_num):
            delete_file_cmd = 'rm /home/linfang/Documents/Code/AAE_tracker/pose_prediction.npz'
            acc_cmd = 'python3 rot_shift_exp.py complete_exps/'+experiment_name+' -o'
            ae_tracker_cmd = 'python3 ae_tracker.py complete_exps/'\
                +experiment_name + ' -o -scene ' + str(scene_id+1)+ ' -inst '+str(inst_id) + ' -v '+ str(reinit_v) +' -len ' + str(pred_len) 
            if R_pass:
                ae_tracker_cmd += ' -r_conf'
            if GRU:
                ae_tracker_cmd += ' -GRU'
            if USE_COSY_INIT:
                ae_tracker_cmd += ' -cosy'

            # print(ae_tracker_cmd)
            # os.system(ae_tracker_cmd)
            # print(acc_cmd)
            # os.system(acc_cmd)
            # os.system(delete_file_cmd)

            # reinit_ae_tracker_cmd = ae_tracker_cmd + ' -reinit'
            # print(reinit_ae_tracker_cmd )
            # os.system(reinit_ae_tracker_cmd )
            # print(acc_cmd)
            # os.system(acc_cmd)
            # os.system(delete_file_cmd)

            # lstm_ae_tracker_cmd = ae_tracker_cmd + ' -lstm'
            # print(lstm_ae_tracker_cmd )
            # os.system(lstm_ae_tracker_cmd )
            # print(acc_cmd)
            # os.system(acc_cmd)
            # os.system(delete_file_cmd)

            reinit_lstm_ae_tracker_cmd = ae_tracker_cmd + ' -reinit -lstm'
            print(reinit_lstm_ae_tracker_cmd )
            os.system(reinit_lstm_ae_tracker_cmd )
            print(acc_cmd)
            os.system(acc_cmd)
            os.system(delete_file_cmd)
