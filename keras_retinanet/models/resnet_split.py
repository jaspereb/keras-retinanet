# -*- coding: utf-8 -*-keras_resnet

"""
This is a modified version of the keras_resnet resnet _2d.py library file.

It adds a layer that splits RGBD at the input into RGB and D.

This module implements popular two-dimensional residual models.
"""

import keras.backend
import keras.layers
import keras.models
import keras.regularizers
import tensorflow as tf

import keras_resnet.blocks
import keras_resnet.layers

from . import blocks_split

import os
import numpy as np

def ResNetSplit(inputs, blocks, include_top=True, classes=1000, freeze_bn=True, numerical_names=None, *args, **kwargs):
    """
    Constructs a `keras.models.Model` object using the given block count.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    :param blocks: the network’s residual architecture

    :param block: a residual block (e.g. an instance of `keras_resnet.blocks.basic_2d`)

    :param include_top: if true, includes classification layers

    :param classes: number of classes to classify (include_top must be true)

    :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)

    :param numerical_names: list of bool, same size as blocks, used to indicate whether names of layers should include numbers or letters

    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)

    Usage:

        >>> import keras_resnet.blocks
        >>> import keras_resnet.models

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> blocks = [2, 2, 2, 2]

        >>> block = keras_resnet.blocks.basic_2d

        >>> model = keras_resnet.models.ResNet(x, classes, blocks, block, classes=classes)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """

    printValues = False
    # saveValues = True

    if keras.backend.image_data_format() == "channels_last":
        axis = 3
    else:
        print("Only channels_last format is supported")
        return

    if numerical_names is None:
        numerical_names = [True] * len(blocks)

    def slice_rgb(x):
        return x[:, :, :, 0:3]

    def slice_d(x):
        return x[:, :, :, 3:]

    def print_stuff(x):
        x = keras.backend.print_tensor(x)
        return x

    # def save_stuff(x):
    #     if(not os.path.exists('/home/jasper/git/CEIG/keras-retinanet/examples/sampleData/save_stuff/data.npy'))
    #     return x

    x_rgb = keras.layers.Lambda(slice_rgb)(inputs)
    x_d = keras.layers.Lambda(slice_d)(inputs)

    if(printValues):
        x_rgb = keras.layers.Lambda(print_stuff)(x_rgb)
        # x_rgb = keras.backend.print_tensor(x_rgb, "RGB is : ")
        # x_rgb = tf.print(x_rgb)


    x_rgb = keras.layers.ZeroPadding2D(padding=3, name="padding_conv1_rgb")(x_rgb)
    x_rgb = keras.layers.Conv2D(64, (7, 7), strides=(2, 2), use_bias=False, name="conv1_rgb")(x_rgb)
    x_rgb = keras_resnet.layers.BatchNormalization(axis=axis, epsilon=1e-5, freeze=freeze_bn, name="bn_conv1_rgb")(x_rgb)
    x_rgb = keras.layers.Activation("relu", name="conv1_relu_rgb")(x_rgb)
    x_rgb = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="pool1_rgb")(x_rgb)

    x_d = keras.layers.ZeroPadding2D(padding=3, name="padding_conv1_d")(x_d)
    x_d = keras.layers.Conv2D(64, (7, 7), strides=(2, 2), use_bias=False, name="conv1_d")(x_d)
    # x_d = keras_resnet.layers.BatchNormalization(axis=axis, epsilon=1e-5, freeze=freeze_bn, name="bn_conv1_d")(x_d)
    x_d = keras.layers.Activation("relu", name="conv1_relu_d")(x_d)
    x_d = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="pool1_d")(x_d)

    features = 64

    outputs = []

    for stage_id, iterations in enumerate(blocks):
        for block_id in range(iterations):
            x_rgb = blocks_split.bottleneck_2d_rgb(features, stage_id, block_id, numerical_name=(block_id > 0 and numerical_names[stage_id]), freeze_bn=freeze_bn)(x_rgb)
            x_d = blocks_split.bottleneck_2d_d(features, stage_id, block_id, numerical_name=(block_id > 0 and numerical_names[stage_id]), freeze_bn=freeze_bn)(x_d)
            x_rgbd = keras.layers.Concatenate(axis=3)([x_rgb, x_d])

        features *= 2

        outputs.append(x_rgbd)

    if include_top:
        print("ERROR: Include_top is not implemented")
        return

    else:
        # Else output each stages features
        return keras.models.Model(inputs=inputs, outputs=outputs, *args, **kwargs)


def ResNet101Split(inputs, blocks=None, include_top=True, classes=1000, *args, **kwargs):
    """
    Constructs a `keras.models.Model` according to the ResNet101 specifications.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    :param blocks: the network’s residual architecture

    :param include_top: if true, includes classification layers

    :param classes: number of classes to classify (include_top must be true)

    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)

    Usage:

        >>> import keras_resnet.models

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> model = keras_resnet.models.ResNet101(x, classes=classes)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    if blocks is None:
        blocks = [3, 4, 23, 3]
    numerical_names = [False, True, True, False]

    return ResNetSplit(inputs, blocks, numerical_names=numerical_names, include_top=include_top, classes=classes, *args, **kwargs)
