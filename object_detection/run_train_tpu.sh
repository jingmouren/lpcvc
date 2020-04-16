export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
export TPU_NAME='tpu-detection-1'

python object_detection/model_tpu_main.py \
	--tpu_zone='REPLACE_WITH_YOUR_COMPUTE_ZONE' \
	--tpu_name=${TPU_NAME} \
	--mode='train' \
	--model_dir='gs://lpcvc2019/object_detection/ssdlite_mobilenetv2_continue_from_det_1/' \
	--pipeline_config_path='gs://lpcvc2019/object_detection/ssdlite_mobilenetv2_continue_from_det_1/ssd_mobilenetv2_1_pipeline_ssdlite_mobilenet_v2_coco.config'
