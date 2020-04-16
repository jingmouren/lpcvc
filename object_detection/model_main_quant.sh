#!/usr/bin/env bash

eval "$(conda shell.bash hook)"
conda activate obj
CONFIG_PATH=samples/configs/quant-test-dev_mmlab_$2_lr$3.config
EXP_DIR=exps/after_quant_$2/lr_$3

mkdir -p $EXP_DIR
cp $CONFIG_PATH $EXP_DIR/pipeline.config

CUDA_VISIBLE_DEVICES=$1 python model_main.py \
    --pipeline_config_path=$CONFIG_PATH \
    --num_train_steps=500000 \
    --model_dir=$EXP_DIR \
    --num_gpus=1 2>&1 | tee -a $EXP_DIR/logs.txt

