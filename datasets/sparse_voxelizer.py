# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu). All Rights Reserved.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part of
# the code.
import torch
import numpy as np
import MinkowskiEngine as ME


class SparseVoxelizer:

  def __init__(self, voxel_size=1):
    """
    Args:
      voxel_size: side length of a voxel
    """
    self.voxel_size = voxel_size

  def get_transformation_matrix(self):
    voxelization_matrix = np.eye(4)
    scale = 1 / self.voxel_size
    np.fill_diagonal(voxelization_matrix[:3, :3], scale)
    return voxelization_matrix

  def voxelize(self, coords):
    assert coords.shape[1] == 3
    coords = np.array(coords)
    rigid_transformation = self.get_transformation_matrix()
    homo_coords = np.hstack((coords, np.ones((coords.shape[0], 1), dtype=coords.dtype)))
    coords_aug = np.floor(homo_coords @ rigid_transformation.T[:, :3])

    [coords_sparse, idx, idx_inverse] = ME.utils.sparse_quantize(coords_aug, return_index=True, return_inverse=True)

    return coords_sparse, idx_inverse


def test():
  N = 16575
  coords = np.random.rand(N, 3) * 10
  feats = np.random.rand(N, 4)
  labels = np.floor(np.random.rand(N) * 3)
  coords[:3] = 0
  labels[:3] = 2
  voxelizer = SparseVoxelizer()
  print(voxelizer.voxelize(coords, feats, labels))


if __name__ == '__main__':
  test()
