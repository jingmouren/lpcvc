# Once-for-All: Train One Network and Specialize it for Efficient Deployment on Diverse Hardware Platforms
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

# APQ: Joint Search for Network Architecture, Pruning and Quantization Policy
# Tianzhe Wang, Kuan Wang, Han Cai, Ji Lin, Zhijian Liu, Hanrui Wang, Yujun Lin, Song Han
# Conference on Computer Vision and Pattern Recognition (CVPR), 2020.

# Winner Solution for 4th Low-Power Computer Vision Challenge (LPCVC)

"""Post-training full quantization script from TF to TFLite."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import tensorflow as tf

import imagenet_input

flags.DEFINE_string("frozen_graph_dir", None, "Path to input frozen_graph bundle.")
flags.DEFINE_enum(
    "input_name", "float_image_input", ["float_image_input", "input"],
    "Name of the input node. `float_image_input` is for image "
    "array input and `truediv` is for normalized input. Please "
    "use `truediv` if require_int8=True and be aware that "
    "users need to handle normalization in the client side.")
flags.DEFINE_string("output_name", "logits", "Name of the output node.")
flags.DEFINE_integer(
    "num_steps", 100,
    "Number of post-training quantization calibration steps to run.")
flags.DEFINE_integer("image_size", 224, "Size of the input image.")
flags.DEFINE_integer("batch_size", 1, "Batch size of input tensor.")
flags.DEFINE_string("output_tflite", None, "Path to output tflite file.")
flags.DEFINE_string("data_dir", None, "Image dataset directory.")
flags.DEFINE_bool(
    "require_int8", False, "Whether all ops should be built-in"
                           " int8, which is necessary for EdgeTPU.")

FLAGS = flags.FLAGS


def representative_dataset_gen():
    """Gets a python generator of image numpy arrays for ImageNet."""
    params = dict(batch_size=FLAGS.batch_size)
    imagenet_eval = imagenet_input.ImageNetInput(
        is_training=False,
        data_dir=FLAGS.data_dir,
        transpose_input=False,
        cache=False,
        image_size=FLAGS.image_size,
        num_parallel_calls=1,
        use_bfloat16=False)

    data = imagenet_eval.input_fn(params)

    def preprocess_map_fn(images, labels):
        del labels
        print(images)
        if FLAGS.input_name == "input":
            images -= tf.constant(
                imagenet_input.MEAN_RGB, shape=[1, 1, 3], dtype=images.dtype)
            images /= tf.constant(
                imagenet_input.STDDEV_RGB, shape=[1, 1, 3], dtype=images.dtype)
        return images

    data = data.map(preprocess_map_fn)
    iterator = data.make_one_shot_iterator()
    for _ in range(FLAGS.num_steps):
        # In eager context, we can get a python generator from a dataset iterator.
        images = iterator.get_next()
        yield [images]


def main(_):
    # Enables eager context for TF 1.x. TF 2.x will use eager by default.
    # This is used to conveniently get a representative dataset generator using
    # TensorFlow training input helper.
    tf.enable_eager_execution()

    converter = tf.lite.TFLiteConverter.from_frozen_graph(
        FLAGS.frozen_graph_dir,
        input_arrays=[FLAGS.input_name],
        output_arrays=[FLAGS.output_name],
        input_shapes={"input": [1, FLAGS.image_size, FLAGS.image_size, 3]})
    # Chooses a tf.lite.Optimize mode:
    # https://www.tensorflow.org/api_docs/python/tf/lite/Optimize
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.inference_output_type = tf.uint8
    converter.inference_input_type = tf.uint8
    converter.representative_dataset = tf.lite.RepresentativeDataset(
        representative_dataset_gen)
    if FLAGS.require_int8:
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    tflite_buffer = converter.convert()
    tf.gfile.GFile(FLAGS.output_tflite, "wb").write(tflite_buffer)
    print("tflite model written to %s" % FLAGS.output_tflite)


if __name__ == "__main__":
    flags.mark_flag_as_required("frozen_graph_dir")
    flags.mark_flag_as_required("output_tflite")
    flags.mark_flag_as_required("data_dir")
    app.run(main)
