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
#tf.reset_default_graph()



def plot_pcd(ax, pcd):
    ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir='y', c=pcd[:, 0], s=0.5, cmap='Reds', vmin=-1, vmax=0.5)
    ax.set_axis_off()
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(-0.3, 0.3)
    ax.set_zlim(-0.3, 0.3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='demo_data/saum_data1/add/miss_saum1/lamp_0084_break.pcd')
    parser.add_argument('--model_type', default='saum1')
    parser.add_argument('--checkpoint', default='log/saum_model/saum1/model-300000')
    parser.add_argument('--num_gt_points', type=int, default=1024)
    args = parser.parse_args()

    partial = read_point_cloud(args.input_path)
    #print(partial)
    #print(type(partial))
    #print(partial.shape)
    partial = np.array(partial.points,dtype=np.float32)
    #print(type(partial))
    #print(partial.shape)
    partial = resample_pcd(partial,540)
    #print(type(partial))
    #print(partial.shape)
    partial = np.expand_dims(partial,axis=0)
    #print(type(partial))
    print(partial.shape)
    #partial = np.tile(partial,[16,1,1])
    
    
   
    model_module = importlib.import_module('.%s' % args.model_type, 'models')
    g_net = model_module.Generator()
    
    #partial = tf.convert_to_tensor(partial) 
    saum, encoder_decoder, concat = g_net(partial)
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

        
    
    complete1 = sess.run(saum)
    print("------------------------")
    print(complete1.shape)
    #partial = tf.convert_to_tensor(partial)
    #print(type(partial))
    #complete = sess.run(partial)
    print(complete1[0])
 
    f = open('demo_data/saum_data1/add/miss_saum1/lamp_0084_break_saum.pcd', 'a')
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
    f.writelines("WIDTH 1620")
    f.writelines("\n")
    f.writelines("HEIGHT 1")
    f.writelines("\n")
    f.writelines("VIEWPOINT 0 0 0 1 0 0 0")
    f.writelines("\n")
    f.writelines("POINTS 1620")
    f.writelines("\n")
    f.writelines("DATA ascii")
    f.writelines("\n")
    
    for j in range(1620):
        for i in range(3):
            f.writelines(str(complete1[0][j][i])+" ")
        f.writelines("\n")
    f.close() 

    complete2 = sess.run(encoder_decoder)
    print("------------------------")
    print(complete2.shape)
    #partial = tf.convert_to_tensor(partial)
    #print(type(partial))
    #complete = sess.run(partial)
    print(complete2[0])
 
    f = open('demo_data/saum_data1/add/miss_saum1/lamp_0084_break_en_de.pcd', 'a')
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
            f.writelines(str(complete2[0][j][i])+" ")
        f.writelines("\n")
    f.close()

    complete3 = sess.run(concat)
    print("------------------------")
    print(complete3.shape)
    #partial = tf.convert_to_tensor(partial)
    #print(type(partial))
    #complete = sess.run(partial)
    print(complete3[0])
 
    f = open('demo_data/saum_data1/add/miss_saum1/lamp_0084_break_concat.pcd', 'a')
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
    f.writelines("WIDTH 2644")
    f.writelines("\n")
    f.writelines("HEIGHT 1")
    f.writelines("\n")
    f.writelines("VIEWPOINT 0 0 0 1 0 0 0")
    f.writelines("\n")
    f.writelines("POINTS 2644")
    f.writelines("\n")
    f.writelines("DATA ascii")
    f.writelines("\n")
    
    for j in range(2644):
        for i in range(3):
            f.writelines(str(complete3[0][j][i])+" ")
        f.writelines("\n")
    f.close() 

