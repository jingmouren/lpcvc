#!/usr/bin/env bash
eval "$(conda shell.bash hook)"
conda activate tf12
EXP_DIR=exps/lpcvc_proxy_$3ms/post_$2

CUDA_VISIBLE_DEVICES=$1 python export_inference_graph.py \
  --alsologtostderr \
  --model_name=proxyless_github_$2 \
  --output_file=${EXP_DIR}/inf_graph.pb \
  --image_size=$4 \
  --latency=$3 \
  --quantize=False

CUDA_VISIBLE_DEVICES=$1 python freeze_graph.py \
    --input_graph=${EXP_DIR}/inf_graph.pb \
    --input_checkpoint=${EXP_DIR}/model.ckpt-0 \
    --input_binary=True \
    --output_graph=${EXP_DIR}/frozen.pb \
    --output_node_names=Softmax

eval "$(conda shell.bash hook)"
conda activate tf15
CUDA_VISIBLE_DEVICES=$1 python post_quant.py \
    --frozen_graph_dir=${EXP_DIR}/frozen.pb \
    --output_tflite=${EXP_DIR}/model_quantized.tflite \
    --input_name=input \
    --image_size=$4 \
    --output_name=Softmax \
    --data_dir=/NFS/tianzhe/imagenet-data/tfrecord
