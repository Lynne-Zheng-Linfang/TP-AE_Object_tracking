#!/bin/bash

export TPAE_WORKSPACE_PATH=~/Documents/Code/TPAE_Object_tracking/ws1
export CUDA_VISIBLE_DEVICES=1

python3 ae_test.py -v 0.2 -len 10 -cosy -GRU -r_conf -pure -ori 

#python3 prior_pose_estimator_train.py GRU/GRU_02 -epoch 400 -steps 300
