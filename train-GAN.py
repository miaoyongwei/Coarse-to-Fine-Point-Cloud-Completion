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
from tf_util import chamfer
from open3d import *
from data_util import resample_pcd

class TrainProvider:
    def __init__(self, args):
        #读取lmdb数据
        #self.ids_batch, self.inputs_batch, self.gts_batch,_ = lmdb_dataflow(args.lmdb_train, args.batch_size,args.num_input_points,args.num_gt_points)
        df_train, self.num_train = lmdb_dataflow(args.lmdb_train, args.batch_size,
                                                 args.num_input_points, args.num_gt_points, is_training=True)
        
        #train.lmdb的数量是45416
        batch_train = get_queued_data(df_train.get_data(), [tf.string, tf.float32, tf.float32],
                                      [[args.batch_size],
                                       [args.batch_size, args.num_input_points, 3],
                                       [args.batch_size, args.num_gt_points, 3]])
        
        self.batch_data = batch_train

class WassersteinGAN(object):
    
    def __init__(self, g_net, d_net,inputs,gt,alpha):
        
        #生成器网络
        self.g_net = g_net
        #鉴别器网络
        self.d_net = d_net
       
        #接受的真实的数据:(16,1024,3)
        self.x = gt
        #接受的(16,512,3)的噪声
        #self.z = tf.random_uniform((16,512,3), minval=-1,maxval=1, dtype=tf.float32)
        self.z = inputs
        
        #生成模型
        self.x_= self.g_net(self.z) 
        #鉴别器 : 真实
        self.d = self.d_net(self.x, reuse=False)
        #(16,1024,1024)
        #鉴别器 : 虚假
        self.d_ = self.d_net(self.x_)
                
        #生成器的损失
        #self.g_loss = (1-alpha) *tf.reduce_mean(self.d_) + alpha * self.loss_fine 
        self.g_loss = tf.reduce_mean(self.d_)
        self.d_loss = tf.reduce_mean(self.d) - tf.reduce_mean(self.d_)
        
        epsilon = tf.random_uniform([], 0.0, 1.0)
        x_hat = epsilon * self.x + (1 - epsilon) * self.x_
        d_hat = self.d_net(x_hat)

        ddx = tf.gradients(d_hat, x_hat)[0]
        print(ddx.get_shape().as_list())
        ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
        ddx = tf.reduce_mean(tf.square(ddx - 1.0) * 10.0)

        self.d_loss = self.d_loss + ddx

        self.d_adam, self.g_adam = None, None
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_adam = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9)\
                .minimize(self.d_loss, var_list=self.d_net.vars)
            self.g_adam = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9)\
                .minimize(self.g_loss, var_list=self.g_net.vars)

        
        
        
def train(args):       
    step = tf.Variable(0, trainable=False)
    alpha = tf.train.piecewise_constant(step, [30000, 50000, 70000],[0.01, 0.1, 0.5, 1.0])
       
    provider = TrainProvider(args)
    
    ids, inputs_batch, gts_batch = provider.batch_data
    #ids_batch,inputs_batch,gts_batch = lmdb_dataflow(args.lmdb_train, args.batch_size,args.num_input_points, args.num_gt_points,is_training=False)

    
    model_module = importlib.import_module('.%s' % args.model_type, 'models')
    
    d_net = model_module.Discriminator()
    g_net = model_module.Generator()


    wgan = WassersteinGAN(g_net,d_net,inputs_batch,gts_batch,alpha)
    
   
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
            diter = 5

            if(step <= 20 or step % 500 == 0):
                diter = 50
            
            for _ in range(0,diter):
                 #先训练鉴别器5次
                sess.run(wgan.d_adam)
        
            sess.run(wgan.g_adam)
         
        
            #获取当前的训练周期数
            epoch = step *args.batch_size // 2048 + 1
            
        
            
         
            #当每到100步的时候,输出训练loss
            if step % args.steps_per_print == 0:
                d_loss = sess.run(wgan.d_loss)
                g_loss = sess.run(wgan.g_loss)
                print('epoch %d  step %d  D_loss %.10f And G_loss %.10f' %(epoch, step, d_loss,g_loss))

            #每100000次保存
            if step % args.steps_per_save == 0:
                saver.save(sess, os.path.join(args.log_dir, 'model'), step)
                print(colored('Model saved at %s' % args.log_dir, 'white', 'on_blue'))
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
    parser.add_argument('--log_dir', default='log/pcn_cd')
    parser.add_argument('--model_type', default='pcn_cd_GAN')
    parser.add_argument('--restore', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_input_points', type=int, default=540)
    parser.add_argument('--num_gt_points', type=int, default=1024)
    parser.add_argument('--base_lr', type=float, default=0.0001)
    parser.add_argument('--lr_decay', action='store_true')
    #if python3 train.py ==> args.lr_decay==FALSE  
    #if python3 train.py --lr_cay  ===> args.lr_decay==TRUE
    parser.add_argument('--lr_decay_steps', type=int, default=50000)
    parser.add_argument('--lr_decay_rate', type=float, default=0.7)
    parser.add_argument('--lr_clip', type=float, default=1e-6)
    parser.add_argument('--max_step', type=int, default=350000)
    parser.add_argument('--steps_per_print', type=int, default=100)
    parser.add_argument('--steps_per_eval', type=int, default=1000)
    parser.add_argument('--steps_per_visu', type=int, default=3000)
    parser.add_argument('--steps_per_save', type=int, default=10000)
    args = parser.parse_args()

    train(args)
