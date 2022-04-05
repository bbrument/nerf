import os
import tensorflow as tf
import numpy as np
import scipy.io as sio
import imageio
import math

trans_t = lambda t : tf.convert_to_tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1],
], dtype=tf.float32)

rot_phi = lambda phi : tf.convert_to_tensor([
    [1,0,0,0],
    [0,tf.cos(phi),-tf.sin(phi),0],
    [0,tf.sin(phi), tf.cos(phi),0],
    [0,0,0,1],
], dtype=tf.float32)

rot_theta = lambda th : tf.convert_to_tensor([
    [tf.cos(th),0,-tf.sin(th),0],
    [0,1,0,0],
    [tf.sin(th),0, tf.cos(th),0],
    [0,0,0,1],
], dtype=tf.float32)


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
    return c2w


def load_matfile_data(basedir, factor=None):

    mat_contents = sio.loadmat(os.path.join(basedir, 'data.mat'))
    poses = mat_contents['poses']
    poses = np.array(poses).astype(np.float32)
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
    imgs = np.stack(imgs, -1)
    imgs = np.moveaxis(imgs, -1, 0)
    print('Loaded image data', imgs.shape)
    
    i_list = np.arange(imgs.shape[0])
    np.random.shuffle(i_list)
      
    nb_train = math.floor(.7*imgs.shape[0])
    nb_test = math.floor(.2*imgs.shape[0])
    
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



