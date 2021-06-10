import numpy as np
import sys
import mayavi.mlab as mlab
import pickle
import os


SCALE_FACTOR = 0.1
MODE = 'sphere'

root_path = 'result/FT3D'
idx = 2677

pc1 = np.load(os.path.join(root_path, str(idx), 'pc1.npy')).squeeze()
pc2 = np.load(os.path.join(root_path, str(idx), 'pc2.npy')).squeeze()
flow = np.load(os.path.join(root_path, str(idx), 'flow.npy')).squeeze()

fig = mlab.figure(figure=None, bgcolor=(0.9,0.9,0.9), fgcolor=(1,1,1), engine=None, size=(1600, 1000))
mlab.points3d(pc1[:, 0], pc1[:, 1], pc1[:, 2], color=(0,0,1), scale_factor=SCALE_FACTOR, figure=fig, mode=MODE)
mlab.points3d(pc2[:, 0], pc2[:, 1], pc2[:, 2], color=(1,0,0), scale_factor=SCALE_FACTOR, figure=fig, mode=MODE)
mlab.points3d(pc1[:, 0] + flow[:, 0], pc1[:, 1] + flow[:, 1], pc1[:, 2] + flow[:, 2], color=(0,1,0), scale_factor=SCALE_FACTOR, figure=fig, mode=MODE)


mlab.view(90, # azimuth
          150, # elevation
          50, # distance
          [0, -1.4, 18], # focalpoint
          roll=0)
mlab.orientation_axes()
mlab.show()
