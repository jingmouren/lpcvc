# Object detection training with TPU


## Prerequisites
1. Open a Google Cloud acount, with 300 USD free credits, and also request the Google Cloud TPU v2 or v3 resources from the quota limits request.
2. Install [Google Cloud SDK](https://cloud.google.com/sdk/install) and [CTPU](https://cloud.google.com/tpu/docs/quickstart). Optional, this can be skipped if you are using google cloud Activate cloud shell (The first icon on the google cloud dashboard on the top right corner).
3. Use the following command to set up the google cloud account on the local environment.
```
gcloud auth login
```
4. Use the following command to allow to access to the resources and APIs from google cloud.
```
gcloud auth application-default login
```
5. Create new TPU v3 resource using the following command. REPLACE_WITH_YOUR_COMPUTE_ZONE can be europe-west4-a, us-east1-a.
```
ctpu up --tpu-only --tpu-size=v3-8 --zone=REPLACE_WITH_YOUR_COMPUTE_ZONE --name tpu-detection-1 --tf-version 1.15
```
6. Create new Google Cloud VM.
```
ctpu up  --vm-only --disk-size-gb=300 --machine-type=n1-standard-2 --zone=REPLACE_WITH_YOUR_COMPUTE_ZONE --name detection-1 --tf-version 1.15
```
7. Login the google Cloud VM. REPLACE_WITH_YOUR_PROJECT_NAME can be found on google cloud dashboard when you log in the console.
```
gcloud beta compute ssh --zone "REPLACE_WITH_YOUR_COMPUTE_ZONE" "detection-1" --project "REPLACE_WITH_YOUR_PROJECT_NAME"
```
8. Once login, apply the following comments to install the essential dependencies and set up the VM properly.
```
sudo apt-get install -y python-tk && \
      pip install --user Cython matplotlib opencv-python-headless pyyaml Pillow && \
      pip install --user 'git+https://github.com/cocodataset/cocoapi#egg=pycocotools&subdirectory=PythonAPI' && \
      pip install --user -U gast==0.2.2
sudo /sbin/sysctl \
       -w net.ipv4.tcp_keepalive_time=120 \
       net.ipv4.tcp_keepalive_intvl=120 \
       net.ipv4.tcp_keepalive_probes=5
```
9. Create the bucket on the google cloud storage
```
gsutil mb gs://lpcvc2019
```

## Fine-tune the on pseudo-labeled data
Upload the tfrecord files and config file to the google cloud bucket.
```
gsutil -m cp -r samples/config/ssd_mobilenetv2_1_pipeline_ssdlite_mobilenet_v2_coco.config gs://lpcvc2019/object_detection/ssdlite_mobilenetv2_continue_from_det_1/ssd_mobilenetv2_1_pipeline_ssdlite_mobilenet_v2_coco.config
```

Run the following command:
```
bash run_train_tpu.sh
```

## Quantization-aware training the on pseudo-labeled data
Uncomment the following line in the config file and upload to the gs, and retrain the model, you would be able to get the quantization-aware training.
```
# uncomments the following to enable the quantization-aware training
#graph_rewriter {
#  quantization {
#    delay: 0
#    activation_bits: 8
#    weight_bits: 8
#  }
#}
```

## Visualization on the progress and monitoring the training
There are two ways to launch the Tensorboard to monitor the training progress.

One is to launch the tensorboard directly in the google code shell.
```
tensorboard --logdir=gs://lpcvc2019/object_detection/ssdlite_mobilenetv2_continue_from_det_1
```

On your browser, you can open the link directly.


Alternatively, you can forward the port of VM to the localhost:
```
gcloud beta compute ssh --zone "REPLACE_WITH_YOUR_COMPUTE_ZONE" "detection-1" --project "REPLACE_WITH_YOUR_PROJECT_NAME" -- -NfL 6006:localhost:6006
```

On the VM, you can type
```
tensorboard --logdir=gs://lpcvc2019/object_detection/ssdlite_mobilenetv2_continue_from_det_1
```