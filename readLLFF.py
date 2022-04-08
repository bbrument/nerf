import os
import numpy as np
import scipy.io as sio
import imageio
import math

basedir = 'data/syntheticGauss/'

poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
print(poses_arr.shape)
poses = poses_arr[:, :-2]
print(poses.shape)
poses = poses.reshape([-1, 3, 5]).transpose([1,2,0])
bds = poses_arr[:, -2:].transpose([1,0])