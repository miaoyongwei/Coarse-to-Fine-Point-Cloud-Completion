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
import os
#tf.reset_default_graph()
import sys
import math
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR) # model
sys.path.append('./utils')
sys.path.append(os.path.join(ROOT_DIR, 'pc_distance'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
import tf_util



def plot_pcd(ax, pcd):
    ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir='y', c=pcd[:, 0], s=0.5, cmap='Reds', vmin=-1, vmax=0.5)
    ax.set_axis_off()
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(-0.3, 0.3)
    ax.set_zlim(-0.3, 0.3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='demo_data/saum_data/add/saum3/airplane_6_break.pcd')
    parser.add_argument('--model_type', default='crn1')
    parser.add_argument('--checkpoint', default='log/crn/model-300000')
    parser.add_argument('--num_gt_points', type=int, default=2048)
    args = parser.parse_args()

    partial = read_point_cloud(args.input_path)
    #print(partial)
    #print(type(partial))
    #print(partial.shape)
    partial = np.array(partial.points,dtype=np.float32)
    #print(type(partial))
    #print(partial.shape)
    #partial = resample_pcd(partial,540)
    #print(type(partial))
    #print(partial.shape)
    partial = np.expand_dims(partial,axis=0)
    #print(type(partial))
    print(partial.shape)
    #partial = np.tile(partial,[16,1,1])
    
    
   
    model_module = importlib.import_module('.%s' % args.model_type, 'models')
    g_net = model_module.Generator()
    
    #partial = tf.convert_to_tensor(partial) 
    coarse, fine = g_net(partial)
    #print(type(g_shape))   #tensor
    #print(fine2.shape)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    

    print("********************************")
    saver = tf.train.Saver()
    print("---------------------------------")
    #sess.run(tf.global_variables_initializer())
    var_list = checkpoint_utils.list_variables(args.checkpoint)
    #for v in var_list:
     #   print(v)
    saver.restore(sess, args.checkpoint)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        
    
    complete1 = sess.run(fine)
    print("------------------------")
    print(complete1.shape)
    #partial = tf.convert_to_tensor(partial)
    #print(type(partial))
    #complete = sess.run(partial)
    print(complete1[0])
 
    f = open('demo_data/saum_data/add/saum3/airplane_6_break_crn.pcd', 'a')
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
            f.writelines(str(complete1[0][j][i])+" ")
        f.writelines("\n")
    f.close() 

    