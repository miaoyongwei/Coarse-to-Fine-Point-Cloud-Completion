# coding=utf-8
# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018

import argparse
import importlib
import models
import numpy as np
import tensorflow as tf
#from matplotlib import pyplot as plt
from data_util import resample_pcd
#from mpl_toolkits.mplot3d import Axes3D
from open3d import *

tf.reset_default_graph()

def plot_pcd(ax, pcd):
    ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir='y', c=pcd[:, 0], s=0.5, cmap='Reds', vmin=-1, vmax=0.5)
    ax.set_axis_off()
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(-0.3, 0.3)
    ax.set_zlim(-0.3, 0.3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='demo_data/fig66/range_hood_0038_break2.pcd')
    parser.add_argument('--model_type', default='pcn1')
    parser.add_argument('--checkpoint', default='log/pcn111/model-300000')
    parser.add_argument('--num_gt_points', type=int, default=1024)
    args = parser.parse_args()

    partial = read_point_cloud(args.input_path)
    print(partial)
    print(type(partial))
    #print(partial.shape)
    partial = np.array(partial.points,dtype=np.float32)
    print(type(partial))
    print(partial.shape)
    #partial = resample_pcd(partial,1024)
    #print(type(partial))
    #print(partial.shape)
    partial = np.expand_dims(partial,axis=0)
    #print(type(partial))
    #print(partial.shape)
    partial = np.tile(partial,[16,1,1])
    
    
   
    model_module = importlib.import_module('.%s' % args.model_type, 'models')
    g_net = model_module.Generator()
    
    g_shape = g_net(partial)
    #print(type(g_shape))   #tensor
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    saver = tf.train.Saver()
    saver.restore(sess, args.checkpoint)

        
    
    complete = sess.run(g_shape)
    print(complete.shape)
    #partial = tf.convert_to_tensor(partial)
    #print(type(partial))
    #complete = sess.run(partial)
    print(complete[0])
 
    f = open('demo_data/fig66/range_hood_0038_break2_net2.pcd', 'a')
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
    f.writelines("WIDTH 1024")
    f.writelines("\n")
    f.writelines("HEIGHT 1")
    f.writelines("\n")
    f.writelines("VIEWPOINT 0 0 0 1 0 0 0")
    f.writelines("\n")
    f.writelines("POINTS 1024")
    f.writelines("\n")
    f.writelines("DATA ascii")
    f.writelines("\n")
    
    for j in range(1024):
        for i in range(3):
            f.writelines(str(complete[0][j][i])+" ")
        f.writelines("\n")
    f.close() 
