# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu). All Rights Reserved.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part of
# the code.
from enum import Enum

import torch
from MinkowskiEngine import MinkowskiNetwork


class NetworkType(Enum):
  """
  Classification or segmentation.
  """
  SEGMENTATION = 0, 'SEGMENTATION',
  CLASSIFICATION = 1, 'CLASSIFICATION'

  def __new__(cls, value, name):
    member = object.__new__(cls)
    member._value_ = value
    member.fullname = name
    return member

  def __int__(self):
    return self.value


class Model(MinkowskiNetwork):
  """
  Base network for all sparse convnet

  By default, all networks are segmentation networks.
  """
  OUT_PIXEL_DIST = -1
  NETWORK_TYPE = NetworkType.SEGMENTATION

  def __init__(self, in_channels, out_channels, config, D, **kwargs):
    super(Model, self).__init__(D)
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.config = config

  def permute_label(self, label, max_label):
    if not isinstance(self.OUT_PIXEL_DIST, (list, tuple)):
      assert self.OUT_PIXEL_DIST > 0, "OUT_PIXEL_DIST not set"
    return super(Model, self).permute_label(label, max_label, self.OUT_PIXEL_DIST)
