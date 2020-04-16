# lpcvc-det

Solution for LPCVC detection track.

## Prerequisites
  * [Installation](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)
  * Download [data](https://drive.google.com/open?id=1sDOwc6aR9XZ0LvPnq5AoKFC1ijTrgWIB) and put it into **root directory** (./coco_testdev)
  * Download [checkpoint](https://drive.google.com/open?id=13GbiTEwjZ0v3t8WO8zCRKMJtOt4OTdiC) for further float-point training and put it into **exps** folder (./exps/fp_train_lr4e-3)

## Pipeline
### Fine-tune the on pseudo-labeled data 
Run the following command: 
`````````bash
bash model_main.sh
`````````
You will get the ckpt file of float-point model trained on pseudo-labeled dataset.

You need to train for at least **45000** steps to get the ckpt for next-step input.
### Quantization-aware training the on pseudo-labeled data 
Run the following command: 
`````````bash
bash model_main_quant.sh 0 45000 4e-5
`````````
You will get the ckpt file of model with fake-quantization graph trained on pseudo-labeled dataset.
  
You need to train for at least **15000** steps to get the ckpt for next-step input.
### Export the tflite model
Run the following command: 
`````````bash
bash export_ssd.sh 0 15000 4e-5
`````````
You will get the tflite for final submission.

## Training on TPU
If you want to use TPU for training, refer this [guideline](README_tpu.md).
