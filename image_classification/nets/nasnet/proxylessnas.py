import sys
from nets.ProxylessNAS.proxyless_nas_tensorflow.tf_model_zoo import proxyless_mobile
import tensorflow as tf


def build_proxyless_mobile_train(images, num_classes,
                                 latency, img_sz,
                                 is_training=True,
                                 final_endpoint=None,
                                 config=None,
                                 current_step=None):
    net = proxyless_mobile(pretrained=True, graph=tf.get_default_graph(), \
                           sess=tf.get_default_session(), is_training=is_training, images=images, img_size=img_sz,
                           only_train=True, latency=latency)
    logits = net.logits
    # https://github.com/tensorflow/models/blob/3ee027fb143b8dc3e39ac37d7cdd37488b261516/research/slim/train_image_classifier.py#L493
    end_points = {}
    return logits, end_points


def build_proxyless_mobile_all(images, num_classes,
                               latency, img_sz,
                               is_training=True,
                               final_endpoint=None,
                               config=None,
                               current_step=None):
    net = proxyless_mobile(pretrained=True, graph=tf.get_default_graph(), \
                           sess=tf.get_default_session(), is_training=is_training, images=images, img_size=img_sz,
                           only_train=False, latency=latency)
    logits = net.logits
    # https://github.com/tensorflow/models/blob/3ee027fb143b8dc3e39ac37d7cdd37488b261516/research/slim/train_image_classifier.py#L493
    end_points = {}
    return logits, end_points


# https://github.com/mit-han-lab/ProxylessNAS/blob/1710d0eeb028b40c142af85283f665c73cb377c0/eval_tf.py#L34
build_proxyless_mobile_all.default_image_size = 192
build_proxyless_mobile_train.default_image_size = 192

arg_scope = tf.contrib.framework.arg_scope


def proxyless_mobile_arg_scope(weight_decay=0):
    with arg_scope([]) as sc:
        return sc
