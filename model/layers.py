'''
Description: 
Author: notplus
Date: 2022-01-07 14:30:23
LastEditors: notplus
LastEditTime: 2022-01-07 16:31:58
FilePath: /model/layers.py

Copyright (c) 2022 notplus
'''

from re import T
import tensorflow as tf
import tensorflow.keras.layers as layers

def _inverted_res_block(inputs, expansion, stride, filters, use_res_connect, dw_kernel_size=3, stage=1, block_id=1, expand=True, output2=False):
    in_channels = tf.keras.backend.int_shape(inputs)[-1]
    x = inputs
    name = 'bbn_stage{}_block{}'.format(stage, block_id)

    if expand:
        x = layers.Conv2D(expansion*in_channels, kernel_size=1,
                          padding='same', use_bias=False, name=name + '_expand_conv')(x)
        x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name=name + '_expand_bn')(x)

        x = layers.ReLU(6, name=name + 'expand_relu')(x)

    out2 = x

    # Depthwise
    x = layers.DepthwiseConv2D(kernel_size=dw_kernel_size, strides=stride, use_bias=False, 
                               padding='same', name=name+'_dw_conv')(x)

    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name=name+'_dw_bn')(x)

    x = layers.ReLU(6, name=name + '_dw_relu')(x)

    # Project
    x = layers.Conv2D(filters, kernel_size=1, padding='same', activation=None,
                      use_bias=False, name=name + '_project_conv')(x)
    
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name=name + '_project_bn')(x)

    if use_res_connect:
        return layers.Add(name=name+'_add')([inputs, x])
    
    if output2:
        return x, out2
    
    return x

def _mb_conv_block(inputs, expansion, stride, filters, use_res_connect, stage=1, block_id=1, expand=True, output2=False):
    in_channels = tf.keras.backend.int_shape(inputs)[-1]
    x = inputs
    name = 'bbn_stage{}_block{}'.format(stage, block_id)

    if expand:
        x = layers.Conv2D(expansion*in_channels, kernel_size=1,
                          padding='same', use_bias=False, name=name + '_expand_conv')(x)
        x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name=name + '_expand_bn')(x)

        x = layers.ReLU(6, name=name + 'expand_relu')(x)

    out2 = x

    # Depthwise
    x = layers.DepthwiseConv2D(kernel_size=5, strides=stride, use_bias=False, 
                               padding='same', name=name+'_dw_conv')(x)

    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name=name+'_dw_bn')(x)

    x = layers.ReLU(6, name=name + '_dw_relu')(x)

    # Project
    x = layers.Conv2D(filters, kernel_size=1, padding='same', activation=None,
                      use_bias=False, name=name + '_project_conv')(x)
    
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name=name + '_project_bn')(x)

    if use_res_connect:
        return layers.Concatenate(name=name+'_con')([inputs, x])
    
    if output2:
        return x, out2
    
    return x

def create_efficient_pose_rt_lite(input_size):
    image_input = layers.Input(shape=(input_size, input_size, 3))

    x = layers.Conv2D(32, kernel_size=3, strides=2, padding='same', use_bias=False, name='conv1')(image_input)
    x = layers.BatchNormalization(name='conv1_bn')(x)
    x = layers.ReLU(6, name='conv1_relu6')(x)

    x = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same', use_bias=False, name='dw_conv2')(x)
    x = layers.BatchNormalization(name='dw_conv2_bn')(x)
    x = layers.ReLU(6, name='dw_conv2_relu6')(x)

    x = layers.Conv2D(16, kernel_size=1, strides=1, padding='same', use_bias=False, name='conv3')(x)
    x = layers.BatchNormalization(name='conv3_bn')(x)

    x = _inverted_res_block(x, expansion=6, stride=2, filters=24, use_res_connect=False, stage=1, block_id=0)

    x = _inverted_res_block(x, expansion=6, stride=1, filters=24, use_res_connect=True, stage=1, block_id=1)

    x = _inverted_res_block(x, expansion=6, stride=2, filters=40, use_res_connect=False, dw_kernel_size=5, stage=2, block_id=1)
    
    p1 = p2 = x = _inverted_res_block(x, expansion=6, stride=1, filters=40, use_res_connect=True, dw_kernel_size=5, stage=2, block_id=2)

    x = _mb_conv_block(x, expansion=6, stride=1, filters=40, use_res_connect=False, stage=3, block_id=1)
    x = _mb_conv_block(x, expansion=6, stride=1, filters=40, use_res_connect=True, stage=3, block_id=2)
    x = _mb_conv_block(x, expansion=3, stride=1, filters=40, use_res_connect=True, stage=3, block_id=3)

    x = layers.Concatenate()([p2, x])
    
    x = _mb_conv_block(x, expansion=1.5, stride=1, filters=40, use_res_connect=False, stage=4, block_id=1)

    x = _mb_conv_block(x, expansion=6, stride=1, filters=40, use_res_connect=True, stage=4, block_id=2)
    x = _mb_conv_block(x, expansion=3, stride=1, filters=40, use_res_connect=True, stage=4, block_id=3)

    x = layers.Concatenate()([p1, x])
    
    x = _mb_conv_block(x, expansion=1.5, stride=1, filters=40, use_res_connect=False, stage=5, block_id=1)    
    x = _mb_conv_block(x, expansion=6, stride=1, filters=40, use_res_connect=True, stage=5, block_id=2)
    x = _mb_conv_block(x, expansion=3, stride=1, filters=40, use_res_connect=True, stage=5, block_id=3)

    x = layers.Conv2D(16, kernel_size=1, strides=1, padding='same', use_bias=False, name='conv6')(x)
    x = layers.BatchNormalization(name='conv6_bn')(x)

    x = layers.UpSampling2D(size=(2,2), interpolation='bilinear')(x)
    x = layers.UpSampling2D(size=(2,2), interpolation='bilinear')(x)
    x = layers.UpSampling2D(size=(2,2), interpolation='bilinear')(x)

    # x = layers.Conv2DTranspose(16, (4, 4), strides=2, padding='same')(x)
    # x = layers.Conv2DTranspose(16, (4, 4), strides=2, padding='same')(x)
    # x = layers.Conv2DTranspose(16, (4, 4), strides=2, padding='same')(x)


    model = tf.keras.Model(inputs=image_input, outputs=[x])
    return model


if __name__ == '__main__':
    model = create_efficient_pose_rt_lite(224)

    model.summary()

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)
