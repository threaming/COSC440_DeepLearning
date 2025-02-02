from __future__ import absolute_import

import os
import tensorflow as tf
import numpy as np
import random
import math

def conv2d(inputs, filters, strides, padding):
  """
	Performs 2D convolution given 4D inputs and filter Tensors.
	:param inputs: tensor with shape [num_examples, in_height, in_width, in_channels]
	:param filters: tensor with shape [filter_height, filter_width, in_channels, out_channels]
	:param strides: MUST BE [1, 1, 1, 1] - list of strides, with each stride corresponding to each dimension in input
	:param padding: either "SAME" or "VALID", capitalization matters
	:return: outputs, NumPy array or Tensor with shape [num_examples, output_height, output_width, output_channels]
	"""
  num_examples = inputs.shape[0]
  in_height = inputs.shape[1]
  in_width = inputs.shape[2]
  input_in_channels = inputs.shape[3]

  filter_height = filters.shape[0]
  filter_width = filters.shape[1]
  filter_in_channels = filters.shape[2]
  filter_out_channels = filters.shape[3]

  num_examples_stride = strides[0]
  strideX = strides[1]
  strideY = strides[2]
  channels_stride = strides[3]

  tf.debugging.assert_equal(input_in_channels, filter_in_channels, \
                            message="input channels and filter channels must be equal")

	# Padding, only stride of 1 allowed
  tf.debugging.assert_equal(strides, [1,1,1,1], \
                            message="in this version only strides of [1,1,1,1] is allowed")
  
  if padding == "SAME":
    paddingX = np.floor((filter_width-1)/2).astype(int)
    paddingY = np.floor((filter_height-1)/2).astype(int)
    inputs = np.pad(inputs, ((0,0),(paddingY,paddingY),(paddingX,paddingX),(0,0)))
  elif padding == "VALID":
    paddingX = 0
    paddingY = 0
  else:
    print("Invalid padding option, must be SAME or VALID")
    return 0
  
  # Execute Convolution
  new_height = (int)((in_height+2*paddingY-filter_height) / strideY +1)
  new_width = (int)((in_width+2*paddingX-filter_width) / strideX +1)
  outputs = np.zeros([num_examples,new_height,new_width,filter_out_channels])
  for y in range(0, new_height):
        for x in range(0, new_width):
            region = inputs[:,y * strideY : y * strideY + filter_height, \
                            x * strideX : x * strideX + filter_width, :]
            for o in range(filter_out_channels):
                outputs[:, y, x, o] = np.sum(region * filters[..., o], axis=(1, 2, 3))

  return  tf.cast(outputs, tf.float32)


