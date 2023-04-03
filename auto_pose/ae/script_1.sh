#!/bin/bash 

export TPAE_WORKSPACE_PATH=~/Documents/Code/TPAE_Object_tracking/ws1


export CUDA_VISIBLE_DEVICES=0

python3 ae_train.py tless_real_syn/obj_05 -dynamic 
python3 ae_train.py tless_real_syn/obj_05 -dynamic -p2
