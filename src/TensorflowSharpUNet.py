# Copyright 2024 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# 2024/03/20 TensorflowSharpUNet.py

# You can customize your TensorflowUnNet model by using a configration file
# Example: train_eval_infer.config


import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"]="false"

import shutil
import sys
import glob
import traceback
import numpy as np
import cv2
import tensorflow as tf


from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Input

from tensorflow.keras.layers import (Conv2D, Dropout, Conv2D, MaxPool2D, MaxPooling2D, DepthwiseConv2D,
                                     Activation, BatchNormalization, UpSampling2D, Concatenate)

from tensorflow.keras.layers import Conv2DTranspose, AveragePooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.activations import elu, relu
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

import tensorflow.keras.backend as K

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from ConfigParser import ConfigParser

from EpochChangeCallback import EpochChangeCallback
from GrayScaleImageWriter import GrayScaleImageWriter
from losses import dice_coef, basnet_hybrid_loss, jacard_loss, sensitivity, specificity
from losses import iou_coef, iou_loss, bce_iou_loss

from TensorflowUNet import TensorflowUNet

MODEL  = "model"
TRAIN  = "train"
INFER  = "infer"

BEST_MODEL_FILE = "best_model.h5"

# The methods in TensorflowSharpUNet class have been taken from
# the following code.
# https://github.com/hasibzunair/sharp-unets/blob/master/demo.ipynb

class TensorflowSharpUNet (TensorflowUNet):

  def __init__(self, config_file):

    self.kernel_size  = 3
    self.enc_filters = [32, 64, 128, 256, 512]
    self.dec_filters = [256, 128, 64, 32,]

    super().__init__(config_file)
    print("=== TensorflowSharpUNet.__init__")
    
  def get_kernel(self):
    """
    See https://setosa.io/ev/image-kernels/
    """

    k1 = np.array([[0.0625, 0.125, 0.0625],
                   [0.125,  0.25, 0.125],
                   [0.0625, 0.125, 0.0625]])
    
    # Sharpening Spatial Kernel, used in paper
    k2 = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])
    
    k3 = np.array([[0, -1, 0],
                   [-1,  5, -1],
                   [0, -1, 0]])
    
    return k1, k2, k3


  def build_sharp_blocks(self, layer):
    """
    Sharp Blocks
    """
    # Get number of channels in the feature
    in_channels = layer.shape[-1]
    # Get kernel
    _, w, _ = self.get_kernel()    
    # Change dimension
    w = np.expand_dims(w, axis=-1)
    # Repeat filter by in_channels times to get (H, W, in_channels)
    w = np.repeat(w, in_channels, axis=-1)
    # Expand dimension
    w = np.expand_dims(w, axis=-1)
    return w

  
  def create(self, num_classes, image_height, image_width, image_channels,
               base_filters = 16, num_layers = 6):
    print("==== TensorflowSharpUNet.create ")
    image_size = (image_width, image_height, image_channels)
    
    "Unet with sharp Blocks in skip connections"

    # Kernel size for sharp blocks
    kernel_size = self.kernel_size
 
    inputs = Input(image_size)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    
    # Skip connection 1
    # 1. Get sharpening kernel weights(1, H, W, channels) 
    W1 = self.build_sharp_blocks(conv4)
    # 2. Build depthwise convolutional layer with random weights
    sb1 = DepthwiseConv2D(kernel_size, use_bias=False, padding='same')
    # 3. Pass input to layer
    conv4 = sb1(conv4)
    # 4. Set filters as layer weights 
    sb1.set_weights([W1])
    # 5. Dont update weights
    sb1.trainable = False
    
    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
    
    # Skip connection 2
    W2 = self.build_sharp_blocks(conv3) 
    sb2 = DepthwiseConv2D(kernel_size, use_bias=False, padding='same')
    conv3 = sb2(conv3)
    sb2.set_weights([W2])
    sb2.trainable = False
   
    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    # Skip connection 3
    W3 = self.build_sharp_blocks(conv2) 
    sb3 = DepthwiseConv2D(kernel_size, use_bias=False, padding='same')
    conv2 = sb3(conv2)
    sb3.set_weights([W3])
    sb3.trainable = False
    
    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
    
    
    # Skip connection 4
    W4 = self.build_sharp_blocks(conv1)
    sb4 = DepthwiseConv2D(kernel_size, use_bias=False, padding='same')
    conv1 = sb4(conv1)
    sb4.set_weights([W4])
    sb4.trainable = False
    
    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    # for multi-class segmentation, use the 'softmax' activation
    activation = "softmax" 
    if num_classes == 1:
      activation = "sigmoid"
      
    conv10 = Conv2D(num_classes, (1, 1), activation= activation)(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])    
    
    return model
