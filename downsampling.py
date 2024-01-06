# coding=utf-8
# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018

import argparse
import importlib
import models
import numpy as np
import tensorflow as tf
from data_util import resample_pcd
from open3d import *
from tensorflow.contrib.framework.python.framework import checkpoint_utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='demo_data/xyzrgb_statuette.pcd')
    args = parser.parse_args()

    partial = read_point_cloud(args.input_path)
    partial = np.array(partial.points,dtype=np.float32)
    partial = resample_pcd(partial,2048)
    partial = np.expand_dims(partial,axis=0)
    partial = tf.convert_to_tensor(partial)
    print(partial.shape)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    tf.compat.v1.disable_eager_execution()
    complete = sess.run(partial)
    print(complete[0])
    f = open('demo_data/detail/xyzrgb_statuette.pcd', 'a')
    f.writelines("# .PCD v0.7 - Point Cloud Data file format")
    f.writelines("\n")
    f.writelines("VERSION 0.7")
    f.writelines("\n")
    f.writelines("FIELDS x y z")
    f.writelines("\n")
    f.writelines("SIZE 4 4 4")
    f.writelines("\n")
    f.writelines("TYPE F F F")
    f.writelines("\n")
    f.writelines("COUNT 1 1 1")
    f.writelines("\n")
    f.writelines("WIDTH 2048")
    f.writelines("\n")
    f.writelines("HEIGHT 1")
    f.writelines("\n")
    f.writelines("VIEWPOINT 0 0 0 1 0 0 0")
    f.writelines("\n")
    f.writelines("POINTS 2048")
    f.writelines("\n")
    f.writelines("DATA ascii")
    f.writelines("\n")
    
    for j in range(2048):
        for i in range(3):
            f.writelines(str(complete[0][j][i])+" ")
        f.writelines("\n")
    f.close() 