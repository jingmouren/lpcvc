#!/usr/bin/env bash

eval "$(conda shell.bash hook)"
conda activate obj
CONFIG_PATH=samples/configs/test-dev_mmlab.config
EXP_DIR=exps/test-dev_mmlab

mkdir -p $EXP_DIR
cp $CONFIG_PATH $EXP_DIR/pipeline.config

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python model_main.py \
    --pipeline_config_path=$CONFIG_PATH \
    --num_train_steps=500000 \
    --model_dir=$EXP_DIR \
    --num_gpus=8 2>&1 | tee -a $EXP_DIR/logs.txt

