#!/usr/bin/env bash
eval "$(conda shell.bash hook)"
conda activate tf12
per_decay=2.5
exp_dir=exps/lpcvc_proxy_$3ms/post_$2

mkdir -p $exp_dir

CUDA_VISIBLE_DEVICES=$1 python train_proxy.py \
	--train_dir=$exp_dir \
	--dataset_split_name=$2 \
	--dataset_dir=/NFS/tianzhe/imagenet-data/tfrecord \
	--model_name=proxyless_github_$2 \
	--batch_size=64 \
	--learning_rate=0.1 \
	--save_interval_secs=100 \
	--save_summaries_secs=100 \
	--log_every_n_steps=100 \
	--train_image_size=$4 \
	--latency=$3 \
	--moving_average_decay=0.9999 \
	--label_smoothing=0.1 \
	--quantize_delay=-1 \
	--learning_rate_decay_factor=0.98 \
	--num_epochs_per_decay=2.5
