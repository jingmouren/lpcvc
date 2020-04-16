
# Winner Solution for 4th LPCVC

## Authors
The **1st place winner** of the **4th On-device Visual Intelligence Competition** ([OVIC](https://docs.google.com/document/d/1Rxm_N7dGRyPXjyPIdRwdhZNRye52L56FozDnfYuCi0k/edit#)) of Low-Power Computer Vision Challenge ([LPCVC](https://lpcv.ai/)), both classification track and detection track. The challenge competes for the best accuracy given latency constraint deploying neural networks on mobile phones.  

* Tianzhe Wang
* Han Cai
* Shuai Zheng
* Jia Li
* [Song Han](https://songhan.mit.edu)

## Description
<!-- Provide description of the model -->
The model submitted for the OVIC and implementation code for training and exportation.

* OVIC track: Image Classification, Object Detection

### Software
<!-- Provide details of the software used -->
We use **Google's Pixel2** to measure the real latency for our exported tflite model.

### Model for Classification
|Model|Download|MD5 checksum|
|-|-|-|
|33ms_top1@0.80585| [Download Link](https://drive.google.com/open?id=1s7TUp_4_sZ7zUT48LNGzeEJ1wBi4b21J) |0091c33f6756b0494d967599695a1c3f|
|35ms_top1@0.7329| [Download Link](https://drive.google.com/open?id=12-qCq2z193FVOGH3xpfbysC38ekLFzFf) |3107acf731434762d87621d824165333|
|36ms_top1@0.73405| [Download Link](https://drive.google.com/open?id=1zy-_M40j00DPtVbejVCJImU7P-VSAjN2) |833e3b56f034427b2a929cc44933a447|

### Model for Detection
|Model|Download|MD5 checksum|
|-|-|-|
|mmlab-distill_23.6| [Download Link](https://drive.google.com/open?id=1AWmKf6h5Wi1ZGrNiot_PcfwAUDENSa0i) |d7945dc1dc52c9372db769facbda1f99|

We provide tflite models for evaluation here. User can use the scripts in the corresponding folder to get checkpoint, frozen graph and tflite.

## Algorithm: once-for-all networks
<!-- Provide details of the algorithms used -->

We address the challenging problem of efficient inference across many devices and resource constraints, especially on edge devices. We propose an [Once-for-All Network (OFA, ICLR'2020)](https://github.com/mit-han-lab/once-for-all) that supports diverse architectural settings by **decoupling model training and architecture search**. We can quickly get a specialized sub-network by selecting from the OFA network without additional training. We also propose a novel progressive shrinking algorithm, a generalized pruning method that reduces the model size across many more dimensions than pruning (depth, width, kernel size, and resolution), which can obtain a surprisingly large number of sub-networks (> 10<sup>19</sup>) that can fit different latency constraints. On edge devices, OFA consistently outperforms SOTA NAS methods (up to **4.0% ImageNet top1 accuracy improvement over MobileNetV3**, or same accuracy but **1.5x faster than MobileNetV3, 2.6x faster than EfficientNet** w.r.t measured latency) while reducing many orders of magnitude GPU hours and CO<sub>2</sub> emission. In particular, OFA achieves a new SOTA **80.0% ImageNet top1 accuracy under 600M MACs**. OFA is the winning solution for 4th Low Power Computer Vision Challenge, both classification track and detection track. Code and 50 pre-trained models on CPU/GPU/DSP/mobile CPU/mobile GPU (for different device & different latency constraints) are released at https://github.com/mit-han-lab/once-for-all.


### 80% top1 ImageNet accuracy with <600M MACs
![](https://hanlab.mit.edu/files/OnceForAll/figures/cnn_imagenet_new.png)

![](https://hanlab.mit.edu/files/OnceForAll/figures/ImageNet_mobile_80acc_mac.png)

OFA achieves 80.0% top1 accuracy with 595M MACs and 80.1% top1 accuracy with
143ms Pixel1 latency, setting a new SOTA ImageNet Top1 accuracy on the mobile device.


### Consistently outperforms MobileNet-V3
![](https://hanlab.mit.edu/files/OnceForAll/figures/diverse_hardwares_new.png)

OFA consistently outperforms MobileNetV3 on mobile platforms.

### Support diverse hardware platforms
![](https://hanlab.mit.edu/files/OnceForAll/figures/many_hardwares_new2.png)

Specialized OFA models consistently achieve significantly higher ImageNet accuracy
with similar latency than non-specialized neural networks on CPU, GPU, mGPU, and FPGA. More
remarkably, specializing for a new hardware platform does not add training cost using OFA.



## References
<!-- Link to references -->
* Once for All: Train One Network and Specialize it for Efficient Deployment [[Website]](https://ofa.mit.edu/) [[arXiv]](https://arxiv.org/abs/1908.09791) [[Slides]](https://hanlab.mit.edu/files/OnceForAll/OFA%20Slides.pdf) [[Video]](https://youtu.be/a_OeT8MXzWI)
```BibTex
@inproceedings{
  cai2020once,
  title={Once for All: Train One Network and Specialize it for Efficient Deployment},
  author={Han Cai and Chuang Gan and Tianzhe Wang and Zhekai Zhang and Song Han},
  booktitle={International Conference on Learning Representations},
  year={2020},
  url={https://arxiv.org/pdf/1908.09791.pdf}
}
```
* APQ: Joint Search for Network Architecture, Pruning and Quantization Policy (To be appear in CVPR 2020)
```BibTex
@inproceedings{
  wang2020apq,
  title={APQ: Joint Search for Network Architecture, Pruning and Quantization Policy},  
  author={Wang, Tianzhe and Wang, Kuan and Cai, Han and Lin, Ji and Liu, Zhijian and Wang, Hanrui and Lin, Yujun and Han, Song},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  year={2020}
}
```

## Training and Evaluation
See the corresponding folder for details.

## License
Apache License 2.0
