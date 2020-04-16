#!/usr/bin/env bash

eval "$(conda shell.bash hook)"
conda activate obj
NAME=exps/after_quant_$2/lr_$3

export CONFIG_FILE=$NAME/pipeline.config
export CHECKPOINT_PATH=$NAME/model.ckpt-$4
export OUTPUT_DIR=$NAME/$4


CUDA_VISIBLE_DEVICES=$1 python export_tflite_ssd_graph.py \
    --pipeline_config_path=$CONFIG_FILE \
    --trained_checkpoint_prefix=$CHECKPOINT_PATH \
    --output_directory=$OUTPUT_DIR \
    --max_detections=100 \
    --add_postprocessing_op=true \
    --use_regular_nms=true


export OUTPUT_DIR=/NFS/tianzhe/tensorflow/models/research/object_detection/exps/after_quant_$2/lr_$3/$4

CUDA_VISIBLE_DEVICES=$1 tflite_convert \
    --graph_def_file=$OUTPUT_DIR/tflite_graph.pb \
    --output_file=$OUTPUT_DIR/detect.tflite \
    --input_arrays=normalized_input_image_tensor \
    --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
    --input_shapes=1,300,300,3 \
    --inference_type=QUANTIZED_UINT8 \
    --change_concat_input_ranges=false \
    --allow_custom_ops \
    --mean_values=128    \
    --std_dev_values=128