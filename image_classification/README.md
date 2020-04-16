# lpcvc-cls
Solution for LPCVC classification track.

## Prerequisites

* TF12 environment
    - TensorFlow version = 1.12
    - Pytorch version >= 1.0
    - Python version >= 3.6
    - Nvidia GPUs
    
* TF15 environment
    - Tensorflow version = 1.15
    - Python version >= 3.6
    - Nvidia GPUs
    
 * Download [checkpoint](https://drive.google.com/open?id=1X9jmAM4mvBZ8BZZ6LKEI_S93qxkiTIHS) and put it into **root directory** (./finetune)
  

## Convert pth to tfinit with 1001 classes
Run the following command: 
``````bash
SPLIT=all # dataset split [train, all]
LATENCY=33 # latency of the model
bash get_tfinit.sh $SPLIT $LATENCY
``````
You will get the tfinit file under the resource constraint 33ms trained on train+val dataset.
## Get the ckpt for the model
Run the following command: 
``````bash
GPU=0 # the GPU id
SPLIT=all # dataset split [train, all]
LATENCY=33 # latency of the model
IMAGE_SIZE=192 # image size of the model input
bash train_proxy.sh $GPU $SPLIT $LATENCY $IMAGE_SIZE
``````
You will get the ckpt file under the resource constraint 33ms trained on train+val dataset.
## Freeze the ckpt and get tflite model
Run the following command: 
``````bash
GPU=0
SPLIT=all
LATENCY=33
IMAGE_SIZE=192
bash post_quant.sh $GPU $SPLIT $LATENCY $IMAGE_SIZE
``````
You will get the tflite model under the resource constraint 33ms trained on train+val dataset.