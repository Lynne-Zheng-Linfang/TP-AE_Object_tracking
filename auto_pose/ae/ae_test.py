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
parser.add_argument("-lstm", action='store_true', default = False)
parser.add_argument("-rtrack", action='store_true', default = False)
parser.add_argument("-cosy", action='store_true', default = False)
parser.add_argument("-noise", action='store_true', default = False)
parser.add_argument("-dist", action='store_true', default = False)
parser.add_argument("-pure", action='store_true', default = False)
parser.add_argument("-refine", action='store_true', default = False)
parser.add_argument("-ronly", action='store_true', default = False)
parser.add_argument("-tonly", action='store_true', default = False)
# parser.add_argument("-inst", type=int,  help='instance_id')
# parser.add_argument("-scene", type=int, help='scene_id')
parser.add_argument("-r_conf", action='store_true', default=False)
parser.add_argument("-YCB", action='store_true', default=False)
parser.add_argument("-icp", action='store_true', default=False)
parser.add_argument("-posecnn", action='store_true', default=False)
parser.add_argument("-ori", action='store_true', default=False)

arguments = parser.parse_args()
# full_name = arguments.experiment_group

reinit_v = arguments.v
pred_len = arguments.len
GRU = arguments.GRU
lstm = arguments.lstm
R_pass =  arguments.r_conf
USE_COSY_INIT = arguments.cosy
NOISE_VERSION = arguments.noise
DISTRIBUTION = arguments.dist
PURE = arguments.pure
ROTAIION_TRACK_ONLY = arguments.rtrack
REFINE_MODE = arguments.refine
YCB = arguments.YCB
R_ONLY = arguments.ronly
T_ONLY = arguments.tonly
ICP = arguments.icp
POSECNN = arguments.posecnn
ORI = arguments.ori

if not YCB:
    gt_files = glob.glob('/home/linfang/Documents/Dataset/T_LESS/tless_primesense/test_primesense/*/gt.yml')
    group_name = 'complete_exps'
else:
    gt_files = glob.glob('/home/linfang/Documents/Dataset/YCB-video/test/*/scene_gt.yaml')
    group_name = 'ycb_v'
group_name = 'tless_real_syn'

gt_files.sort()

for gt_file in gt_files:
    scene_id = int(os.path.basename(os.path.dirname(gt_file)).split('.')[0])
    # if (scene_id) == 1:
    #     continue
    # if (scene_id) not in [56,57,58,59]:
    #     continue
    gt_info = inout.load_gt(gt_file)
    start_id = list(gt_info.keys())[0]
    obj_num = len(gt_info[start_id])
    obj_list = []
    for i in range(obj_num):
        obj_list.append(int(gt_info[start_id][i]['obj_id']))
    obj_dict = dict(Counter(obj_list))
    for obj_id in obj_dict.keys():
        if (obj_id not in [11]):
            continue
        # if obj_id not in [14,19,20]:
        #     continue
        if (not YCB) and (obj_id in [6,25,30]):
            experiment_name = 'obj_{}_no_rot_in_predictor_w_h_sep_rot_crop_in_render_position'.format(obj_id)
        else:
            experiment_name = 'obj_' + str(obj_id).zfill(2)

        if DISTRIBUTION:
            total_instance_num = obj_dict[obj_id]
        else:
            total_instance_num = 1
        for inst_id in range(total_instance_num):
            # if inst_id != 2:
            #     continue
            delete_file_cmd = 'rm /home/linfang/Documents/Code/AAE_tracker/pose_prediction.npz'
            acc_cmd = 'python3 show_result.py ' +group_name+'/'+experiment_name + ' --out_filename ' + 'icp_ycb_scene_'+str(scene_id)+'_obj_'+str(obj_id)+ ' --scene '+str(scene_id)
            # acc_cmd = 'python3 rot_shift_exp.py ' +group_name+'/'+experiment_name+' -o'
            if YCB:
                acc_cmd += ' -YCB'
            ae_tracker_cmd = 'python3 get_ae_test_result.py ' + group_name +'/'\
                +experiment_name + ' -o -scene ' + str(scene_id)+ ' -inst '+str(inst_id) + ' -v '+ str(reinit_v) +' -len ' + str(pred_len) 
            if R_pass:
                ae_tracker_cmd += ' -r_conf'
            if GRU:
                ae_tracker_cmd += ' -GRU -lstm'
            if lstm:
                ae_tracker_cmd += ' -lstm'
            if USE_COSY_INIT:
                ae_tracker_cmd += ' -cosy -pure'
            if NOISE_VERSION:
                ae_tracker_cmd += ' -noise'
            if DISTRIBUTION:
                ae_tracker_cmd += ' -dist'
            if PURE:
                ae_tracker_cmd += ' -pure'
            if ROTAIION_TRACK_ONLY:
                ae_tracker_cmd += ' -rtrack'
            if REFINE_MODE:
                ae_tracker_cmd += ' -refine'
            if YCB:
                ae_tracker_cmd += ' -YCB'
            if R_ONLY:
                ae_tracker_cmd += ' -ronly'
            if T_ONLY:
                ae_tracker_cmd += ' -tonly'
            if ICP:
                ae_tracker_cmd += ' -icp'
            if POSECNN:
                ae_tracker_cmd += ' -posecnn'
            if ORI:
                ae_tracker_cmd += ' -ori'

            # print(ae_tracker_cmd)
            # os.system(ae_tracker_cmd)
            # print(acc_cmd)
            # os.system(acc_cmd)
            # os.system(delete_file_cmd)

            reinit_ae_tracker_cmd = ae_tracker_cmd + ' -reinit'
            print(reinit_ae_tracker_cmd )
            os.system(reinit_ae_tracker_cmd )
            # print(acc_cmd)
            # os.system(acc_cmd)
            # exit()

            # lstm_ae_tracker_cmd = ae_tracker_cmd + ' -lstm'
            # print(lstm_ae_tracker_cmd )
            # os.system(lstm_ae_tracker_cmd )
            # print(acc_cmd)
            # os.system(acc_cmd)
            # os.system(delete_file_cmd)

            # reinit_lstm_ae_tracker_cmd = ae_tracker_cmd + ' -reinit -lstm'
            # print(reinit_lstm_ae_tracker_cmd )
            # os.system(reinit_lstm_ae_tracker_cmd )
