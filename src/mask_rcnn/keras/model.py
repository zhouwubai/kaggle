"""
Mask R-CNN

Most code borrow https://github.com/matterport/Mask_RCNN
As a learning process
"""

import os
import random
import datetime
import re
import math
import logging
from collections import OrderedDict
import multiprocessing
import numpy as np
import skimage.transform
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM

import utils


# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')


############################################################
#  Utility Functions
############################################################


def log(text, array=None):
    """
    Prints a text message. And, optionally, if a Numpy array is
    provided it prints its shape, min and max value

    Args:
        text: str
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20} min: {:10.5f} max: {:10.5f} {}".format(
            str(array.shape),
            array.min() if array.size else "",
            array.max() if array.size else "",
            array.dtype))
    print(text)


class BatchNorm(KL.BatchNormalization):
    """
    Batch normalization has a negative effect on training if batches
    are small so this layer often frozen and functions as linear layer
    """
    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. Default mode
            False: Freeze BN layers.
            True: (do not use). Set layer in training mode even when inference
        """
        return super(self.__class__, self).call(inputs, training=training)


def compute_backbone_shapes(config, image_shape):
    """Computes the width and height of each stage of the
    backbone network.

    Returns:
        [N, (height, width)]. Where N is the number of stages
    """
    assert config.backbone in ["resnet50", "resnet101"]
    return np.array(
        [[int(math.ceil(image_shape[0] / stride)),
         int(math.ceil(image_shape[1] / stride))]
         for stride in config.backbone_strides])

############################################################
#  Resnet Graph
############################################################


def identity_block(input_tensor, kernel_size, filters, stage, block,
                   use_bias=True, train_bn=True):
    """
    The identy_block is the block that has no conv layer at shortcut

    Args:

    """






























