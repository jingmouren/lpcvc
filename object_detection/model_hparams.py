# Once-for-All: Train One Network and Specialize it for Efficient Deployment on Diverse Hardware Platforms
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

# APQ: Joint Search for Network Architecture, Pruning and Quantization Policy
# Tianzhe Wang, Kuan Wang, Han Cai, Ji Lin, Zhijian Liu, Hanrui Wang, Yujun Lin, Song Han
# Conference on Computer Vision and Pattern Recognition (CVPR), 2020.

# Winner Solution for 4th Low-Power Computer Vision Challenge (LPCVC)

"""Hyperparameters for the object detection model in TF.learn.

This file consolidates and documents the hyperparameters used by the model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def create_hparams(hparams_overrides=None):
  """Returns hyperparameters, including any flag value overrides.

  Args:
    hparams_overrides: Optional hparams overrides, represented as a
      string containing comma-separated hparam_name=value pairs.

  Returns:
    The hyperparameters as a tf.HParams object.
  """
  hparams = tf.contrib.training.HParams(
      # Whether a fine tuning checkpoint (provided in the pipeline config)
      # should be loaded for training.
      load_pretrained=True)
  # Override any of the preceding hyperparameter values.
  if hparams_overrides:
    hparams = hparams.parse(hparams_overrides)
  return hparams
