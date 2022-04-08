import os
import tensorflow as tf
import numpy as np
import scipy.io as sio
import imageio
import matplotlib.pyplot as plt


def load_matfile_data(basedir, factor=None):

    mat_contents = sio.loadmat(os.path.join(basedir, 'data.mat'))
    poses = mat_contents['poses']
    poses = np.array(poses).astype(np.float32)
    poses = np.concatenate([poses[:, 0:1, :], -poses[:, 1:3, :], poses[:,3:,:]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    intrinsics = mat_contents['intrinsics']

    imgdir = os.path.join(basedir, 'images')
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return

    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[0] != len(imgfiles):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[0]) )
        return
    
    imgs = list()
    for f in imgfiles:
        img = imageio.imread(f)
        imgs.append(img)
    imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
    imgs = np.stack(imgs, -1)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    print('Loaded image data', imgs.shape)
    
    i_list = np.arange(imgs.shape[0])
    np.random.shuffle(i_list)
      
    nb_train = tf.math.floor(.7*imgs.shape[0])
    nb_test = tf.math.floor(.2*imgs.shape[0])
    
    i_split = list()
    i_split.append(i_list[:nb_train])
    i_split.append(i_list[nb_train:nb_train+nb_test])
    i_split.append(i_list[nb_train+nb_test:])

    H, W = imgs[0].shape[:2]
    focal = (intrinsics[0,0] + intrinsics[1,1])/2.

    #render_poses = tf.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]],0)
    render_poses = poses[i_split[1],:,:]


    if factor is not None:
        H = H//factor
        W = W//factor
        focal = focal/factor
        imgs = tf.image.resize_area(imgs, [H, W]).numpy()

    return imgs, poses, render_poses, [H, W, focal], i_split


