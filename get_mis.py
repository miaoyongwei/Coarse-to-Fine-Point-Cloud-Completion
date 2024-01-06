# coding=utf-8
# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018
import argparse
import importlib
import models
import os
import tensorflow as tf
import tensorflow.contrib as tc
import time
import numpy as np
from data_util import lmdb_dataflow, get_queued_data
from termcolor import colored
from tf_util import chamfer,earth_mover
from data_util import resample_pcd

class TrainProvider:
    def __init__(self, args):
        #读取lmdb数据
        df_train, self.num_train = lmdb_dataflow(args.lmdb_train, args.batch_size,
                                                 args.num_input_points, args.num_pre_points, args.num_gt_points, is_training=True)
        
        #train.lmdb的数量是45416
        batch_train = get_queued_data(df_train.get_data(), [tf.string, tf.float32, tf.float32],
                                      [[args.batch_size],
                                       [args.batch_size, args.num_input_points, 3], [args.batch_size, args.num_pre_points, 3]
                                       [args.batch_size, args.num_gt_points, 3]])
        
        self.batch_data = batch_train


class WassersteinGAN(object):
    
    def __init__(self,g_net,inputs,gt,alpha):
        
        #生成器网络
        self.g_net = g_net
        
        #接受的真实的数据:(16,1024,3)
        self.x = gt
        
        self.z = inputs
        
        #生成模型
        self.x_= self.g_net(self.z)
        
        #CD值loss的东东
        self.loss_fine = chamfer(self.x_, self.x) + earth_mover(self.x_, self.x)
        
        
        

def fileSave(step,complete):

    for k in range(4):
        f = open('data/result14/bookcase'+str(k)+'_break'+str(step)+'.pcd', 'a')
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
        f.writelines("WIDTH 2070")
        f.writelines("\n")
        f.writelines("HEIGHT 1")
        f.writelines("\n")
        f.writelines("VIEWPOINT 0 0 0 1 0 0 0")
        f.writelines("\n")
        f.writelines("POINTS 2070")
        f.writelines("\n")
        f.writelines("DATA ascii")
        f.writelines("\n")
    
        for j in range(2070):
            for i in range(3):
                f.writelines(str(complete[k][j][i])+" ")
            f.writelines("\n")
    f.close()

        
def train(args):  
    
    provider = TrainProvider(args)
    
    ids, inputs, pres, gts = provider.batch_data
    with tf.Session() as sess:
        print(sess.run(ids))
    print(inputs.shape)
    print(type(inputs))
    print(pres.shape)
    print(type(pres))
    print(gts.shape)
    print(type(gts))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb_train', default='data/output/1119.lmdb')
    parser.add_argument('--log_dir', default='log/pcn_cd_earthmove2')
    parser.add_argument('--model_type', default='pcn_cd_earthmove1')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_input_points', type=int, default=540)
    parser.add_argument('--num_pre_points', type=int, default=484)
    parser.add_argument('--num_gt_points', type=int, default=1024)
    parser.add_argument('--base_lr', type=float, default=0.0001)
    parser.add_argument('--lr_decay', action='store_true')
    #if python3 train.py ==> args.lr_decay==FALSE  
    #if python3 train.py --lr_cay  ===> args.lr_decay==TRUE
    parser.add_argument('--lr_decay_steps', type=int, default=50000)
    parser.add_argument('--lr_decay_rate', type=float, default=0.7)
    parser.add_argument('--lr_clip', type=float, default=1e-6)
    parser.add_argument('--max_step', type=int, default=300000)
    parser.add_argument('--steps_per_save', type=int, default=10000)
    args = parser.parse_args()

    train(args)
