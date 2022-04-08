import os
import numpy as np
import scipy.io as sio
import imageio
import math

basedir = 'data/pepper_gauss/'

mat_contents = sio.loadmat(os.path.join(basedir, 'data.mat'))
poses = mat_contents['poses']
K = mat_contents['intrinsics']
bds = mat_contents['bds']

poses = np.array(poses).astype(np.float32) #[down right backward]
N = poses.shape[-1]
print(N)
print(poses.shape)
K = np.array(K).astype(np.float32) #[f1 0 p1; 0 f2 p2; 0 0 1]
bds = np.array(bds).astype(np.float32) #[far; near] x N

imgdir = os.path.join(basedir, 'images')
imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
img = imageio.imread(imgfiles[0])
img = (np.array(img) / 255.).astype(np.float32)

H, W = img.shape[:2]
focal = (K[0,0] + K[1,1])/2.
hwf = np.array([H, W, focal]).astype(np.float32)
hwf = np.transpose(np.tile(hwf, (N, 1)))
hwf = np.expand_dims(hwf,1)
print(hwf.shape)

posesANDhwf = np.concatenate([poses[:,:,:], hwf[:,:]],1)
print(posesANDhwf.shape)

posesANDhwfReshaped = posesANDhwf.reshape(-1, posesANDhwf.shape[-1])
print(posesANDhwfReshaped.shape)

bds = np.flip(bds,0)
print(bds)

final = np.transpose(np.concatenate((posesANDhwfReshaped,bds),0))
print(final.shape)

np.save(os.path.join(basedir, 'poses_bounds.npy'),final)