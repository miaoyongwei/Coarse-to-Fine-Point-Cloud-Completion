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
                                                 args.num_input_points, args.num_gt_points, is_training=True)
        
        #train.lmdb的数量是45416
        batch_train = get_queued_data(df_train.get_data(), [tf.string, tf.float32, tf.float32],
                                      [[args.batch_size],
                                       [args.batch_size, args.num_input_points, 3],
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
        self.x_, self.x__= self.g_net(self.z)

        
        #CD值loss的东东
        self.loss_fine = 0.5*earth_mover(self.x__, self.x) + 0.5*earth_mover(self.x_, self.x)
        
        
        

def fileSave(step,complete):

    for k in range(4):
        f = open('data/result11/bookcase'+str(k)+'_break'+str(step)+'.pcd', 'a')
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
                f.writelines(str(complete[k][j][i])+" ")
            f.writelines("\n")
    f.close()

        
def train(args):  
    print("aaaaa")
    step = tf.Variable(0, trainable=False)
    print("bbbbb")
    alpha = tf.train.piecewise_constant(step, [70000, 150000, 200000],[0.01, 0.1, 0.5, 0.99])
    print("ccccc")
    provider = TrainProvider(args)
    print("ddddd")
    ids, inputs, gts = provider.batch_data
    print("eeeee")
    model_module = importlib.import_module('.%s' % args.model_type, 'models')
    print("fffff")
    g_net = model_module.Generator()
    print("ggggg")
    wgan = WassersteinGAN(g_net,inputs,gts,alpha)
    print("hhhhh")
    
   
    #定义cd loss训练函数
    trainer_op = tf.train.AdamOptimizer(0.0001).minimize(wgan.loss_fine)

    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    
    sess =tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    

    saver = tf.train.Saver()
        
    

    #创建一个线程协调器
    coord = tf.train.Coordinator()
    #将当前会话和协调器加入线程
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
     
    step = sess.run(step)
    
    try:
        while not coord.should_stop():
            #训练步数+1
            step += 1
 
            #cd loss函数
            loss_fine,_ = sess.run([wgan.loss_fine,trainer_op])
            if(step%1000==0):
                print('step %d , loss_fine is %.15f'%(step,loss_fine))
            
            
            #每10000次保存
            if step % args.steps_per_save == 0:
                #saver.save(sess, os.path.join(args.log_dir, 'model'), step)
                #print(colored('Model saved at %s' % args.log_dir, 'white', 'on_blue'))
                  
                complete = sess.run(wgan.x__)
                
                fileSave(step,complete)
                
                saver.save(sess, os.path.join(args.log_dir, 'model'), step)

            #如果步数大于最大步数
            if step >= args.max_step:
                break
            
    except tf.errors.OutOfRangeError:
        print('end of queue!')
    finally:    
        coord.request_stop()
    coord.join(threads)
    sess.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb_train', default='data/2016.lmdb')
    parser.add_argument('--log_dir', default='log/refine')
    parser.add_argument('--model_type', default='refine_concat')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_input_points', type=int, default=540)
    parser.add_argument('--num_gt_points', type=int, default=1024)
    parser.add_argument('--base_lr', type=float, default=0.0001)
    parser.add_argument('--lr_decay', action='store_true')
    #if python3 train.py ==> args.lr_decay==FALSE  
    #if python3 train.py --lr_cay  ===> args.lr_decay==TRUE
    parser.add_argument('--lr_decay_steps', type=int, default=50000)
    parser.add_argument('--lr_decay_rate', type=float, default=0.7)
    parser.add_argument('--lr_clip', type=float, default=1e-6)
    parser.add_argument('--max_step', type=int, default=1000000)
    parser.add_argument('--steps_per_save', type=int, default=10000)
    args = parser.parse_args()

    train(args)
