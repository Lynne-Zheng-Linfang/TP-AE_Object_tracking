import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse
import glob

def plot_results(ori_data, crt_data, zs, distances, error_type, type_name = 'Error', fontsize = '23'):
    line_colors = []
    plt.figure(figsize=(6,8))
    plt.title(error_type+' Metric (' +type_name +')', fontsize=str(int(fontsize)+4))
    zs = zs.copy()/10
    distances = distances.copy()/10
    if type_name == 'Error':
        if error_type == 'Rotation Error':
            unit = 'degree'
        else:
            unit = 'mm'
    else:
        unit = ''
    for i, z in enumerate(zs):
        if i % 2 == 0:
            continue
        plt.plot(distances,ori_data[i], marker = 'o', markersize= 10, linewidth = 4, linestyle=':', label='z={}cm, w/o crt'.format(int(z)))
        plt.plot(distances,crt_data[i], marker = '^',  markersize= 10, linewidth = 4,label='z={}cm'.format(int(z)))
    plt.legend(fontsize=str(int(fontsize)-1))
    plt.xlabel('distance to camera z-axis (cm)',fontsize=fontsize)
    if type_name == 'Error':
        plt.ylabel(error_type+' Mean '+ type_name+ ' ' + unit, fontsize=fontsize)
    else:
        plt.ylabel(error_type+' '+ type_name+ ' ' + unit,fontsize=fontsize)
    if type_name == 'AR':
        axes = plt.gca()
        axes.set_ylim([-0.1,1.1])
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    
    plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.1)
    plt.savefig('/home/linfang/Desktop/rotation_shift_exp/'+error_type+'_'+type_name+'.png', bbox_inchs = 'tight')
    # plt.show()
base = '/home/linfang/Documents/Code/AAE_tracker/tests/rot_shift_exp'
files = glob.glob(os.path.join(base, 'rot_shift_exp_obj_25_*.npz'))
for filename in files:
    error_info = (filename.split('.')[0]).split('_')[-2:]
    if error_info[0] == 'vsd':
        error_type = 'VSD-'+error_info[1]
    elif error_info[1] == 'adi':
        error_type = 'ADD-S'
    elif error_info[1] == 're':
        error_type = 'Rotation Error'
    else:
        error_type = error_info[1].upper()

    results = np.load(filename)
    acc = results['acc'] 
    correctified_acc = results['correctified_acc'] 
    mean_errors = np.round(results['mean_errors'],2)
    
    mean_correctified_errors = np.round(results['mean_correctified_errors'],2 )
    zs = results['zs'] 
    distances = results['distances'] 
    plot_results(acc, correctified_acc, zs, distances, error_type, type_name = 'AR')
    # exit()
    plot_results(mean_errors, mean_correctified_errors, zs, distances, error_type, type_name = 'Error')

exit()
parser = argparse.ArgumentParser()
parser.add_argument("--obj_id", type=int, help='object id')
parser.add_argument("--error_type", type=str, help='should be one of [re, vsd_step, vsd_tlinear, adi, add]')
arguments = parser.parse_args()
obj_id = arguments.obj_id
error_type = arguments.error_type
error_types = ['re','vsd_step', 'vsd_tlinear', 'adi', 'add']
if error_type not in error_types:
    print('Error: error_type should be one of the following:')
    print(error_types)
    exit()
workspace_path = os.environ.get('AE_WORKSPACE_PATH')
if workspace_path == None:
    print('Please define a workspace path:\n')
    print('export AE_WORKSPACE_PATH=/path/to/workspace\n')
    exit(-1)

main_work_folder = os.path.dirname(workspace_path)
results = np.load(main_work_folder+'/rot_shift_exp_obj_'+str(obj_id)+'_'+error_type+'.npz')

acc = results['acc'] 
correctified_acc = results['correctified_acc'] 
mean_errors = results['mean_errors'] 
mean_correctified_errors = results['mean_correctified_errors'] 
zs = results['zs'] 
distances = results['distances'] 
error_type = str(results['error_type']).upper()
if error_type == 'RE':
    unit = '(degrees)'
    error_type = 'Rotation'
else:
    unit = '(mm)'
# angles = results['angles']
plt.title(error_type+'_error')
for i, z in enumerate(zs):
    plt.plot(distances, mean_errors[i], label='z={}mm'.format(z))
plt.legend()
plt.xlabel('distance (mm)')
plt.ylabel(error_type+'_error ' + unit)
plt.show()

plt.title(error_type+'_accuracy')
for i, z in enumerate(zs):
    plt.plot(distances, acc[i]*100, label='z={}mm'.format(z))
plt.legend()
plt.xlabel('distance (mm)')
plt.ylabel(error_type+'_accuracy '+unit)
plt.show()

plt.title(error_type+'_correctified_accuracy')
for i, z in enumerate(zs):
    plt.plot(distances, correctified_acc[i]*100, label='z={}mm'.format(z))
plt.legend()
plt.xlabel('distance (mm)')
plt.ylabel(error_type+'_correctified_accuracy '+ unit)
plt.show()

plt.title(error_type+'_correctified_error')
for i, z in enumerate(zs):
    plt.plot(distances, mean_correctified_errors[i], label='z={}mm'.format(z))
plt.legend()
plt.xlabel('distance (mm)')
plt.ylabel(error_type+'_correctified_error '+ unit)
plt.show()

# if error_type == 'vsd_step':
#     vsd_acc_step = results['vsd_acc_step'] 
#     vsd_acc_tlinear = results['vsd_acc_tlinear']
#     rot_acc = results['rot_acc']
#     adi_acc = results['adi_acc'] 
#     add_acc = results['add_acc'] 
#     mean_vsd_errors_step = results['mean_vsd_errors_step'] 
#     mean_vsd_errors_tlinear = results['mean_vsd_errors_tlinear'] 
#     mean_add_errors =  results['mean_add_errors'] 
#     mean_adi_errors = results['mean_adi_errors'] 
#     mean_rot_errors = results['mean_rot_errors'] 
#     zs = results['zs'] 
#     distances = results['distances'] 
#     # angles = results['angles']
#     # 
#     plt.title('VSD_Error_tlinear')
#     for i, z in enumerate(zs):
#         plt.plot(distances, mean_vsd_errors_tlinear[i], label='z={}'.format(z))
#     plt.legend()
#     plt.xlabel('distance')
#     plt.ylabel('VSD_Tlinear_Error')
#     plt.show()
#     # 
#     plt.title('VSD_Accuracy_step')
#     for i, z in enumerate(zs):
#         plt.plot(distances, vsd_acc_step[i]*100, label='z={}'.format(z))
#     plt.legend()
#     plt.xlabel('distance')
#     plt.ylabel('VSD_Step_Accuray')
#     plt.show()
#     # 
#     plt.title('VSD_Error_step')
#     for i, z in enumerate(zs):
#         plt.plot(distances, mean_vsd_errors_step[i], label='z={}'.format(z))
#     plt.legend()
#     plt.xlabel('distance')
#     plt.ylabel('VSD_Step_Error')
#     plt.show()
#     # 
# if error_type == 'vsd_tlinear':
#     plt.title('VSD_Tlinear_Accuracy')
#     for i, z in enumerate(zs):
#         plt.plot(distances, vsd_acc_tlinear[i]*100, label='z={}'.format(z))
#     plt.legend()
#     plt.xlabel('distance')
#     plt.ylabel('VSD_TLiner_Accuray')
#     plt.show()
#     # 
#     plt.title('Rotation_Accuracy')
#     for i, z in enumerate(zs):
#         plt.plot(distances, rot_acc[i]*100, label='z={}'.format(z))
#     plt.legend()
#     plt.xlabel('distance')
#     plt.ylabel('Rotation_Accuray')
#     plt.show()
#     # 
#     plt.title('Rotation_Error')
#     for i, z in enumerate(zs):
#         plt.plot(distances, mean_rot_errors[i]*100, label='z={}'.format(z))
#     plt.legend()
#     plt.xlabel('distance')
#     plt.ylabel('Rotation_Error')
#     plt.show()

# if error_type == 'add':
#     plt.title('ADD_Accuracy')
#     for i, z in enumerate(zs):
#         plt.plot(distances, add_acc[i]*100, label='z={}'.format(z))
#     plt.legend()
#     plt.xlabel('distance')
#     plt.ylabel('ADD_Accuray')
#     plt.show()
#     # 
#     plt.title('ADD_Error')
#     for i, z in enumerate(zs):
#         plt.plot(distances, mean_add_errors[i]*100, label='z={}'.format(z))
#     plt.legend()
#     plt.xlabel('distance')
#     plt.ylabel('ADD_Error')

# if error_type == 'adi':
#     plt.title('ADI_Accuracy')
#     for i, z in enumerate(zs):
#         plt.plot(distances, adi_acc[i]*100, label='z={}'.format(z))
#     plt.legend()
#     plt.xlabel('distance')
#     plt.ylabel('ADI_Accuray')
#     plt.show()
#     # 
#     plt.title('ADI_Error')
#     for i, z in enumerate(zs):
#         plt.plot(distances, mean_adi_errors[i]*100, label='z={}'.format(z))
#     plt.legend()
#     plt.xlabel('distance')
#     plt.ylabel('ADI_Error')

# acc = results['acc'] 
# correctified_acc = results['correctified_acc'] 
# mean_errors = results['mean_errors'] 
# mean_correctified_errors = results['mean_correctified_errors'] 
# zs = results['zs'] 
# distances = results['distances'] 
# error_type = str(results['error_type']).upper()
# error_type = 'Rotation'
# # angles = results['angles']
# plt.title(error_type+'_error')
# for i, z in enumerate(zs):
#     plt.plot(distances, mean_errors[i], label='z={}mm'.format(z))
# plt.legend()
# plt.xlabel('distance (mm)')
# plt.ylabel(error_type+'_error ' + unit)
# plt.show()

# plt.title(error_type+'_accuracy')
# for i, z in enumerate(zs):
#     plt.plot(distances, acc[i]*100, label='z={}mm'.format(z))
# plt.legend()
# plt.xlabel('distance (mm)')
# plt.ylabel(error_type+'_accuracy '+unit)
# plt.show()

# plt.title(error_type+'_correctified_accuracy')
# for i, z in enumerate(zs):
#     plt.plot(distances, correctified_acc[i]*100, label='z={}mm'.format(z))
# plt.legend()
# plt.xlabel('distance (mm)')
# plt.ylabel(error_type+'_correctified_accuracy '+ unit)
# plt.show()

# plt.title(error_type+'_correctified_error')
# for i, z in enumerate(zs):
#     plt.plot(distances, mean_correctified_errors[i], label='z={}mm'.format(z))
# plt.legend()
# plt.xlabel('distance (mm)')
# plt.ylabel(error_type+'_correctified_error '+ unit)
# plt.show()
