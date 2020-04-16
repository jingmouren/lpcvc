#!/usr/bin/env bash
eval "$(conda shell.bash hook)"
conda activate tf12

python get_tfinit.py \
	--data_split=$1 \
	--latency=$2 \
