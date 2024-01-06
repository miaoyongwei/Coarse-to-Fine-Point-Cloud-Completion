#coding=utf-8
# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018

import tensorflow as tf
from pc_distance import tf_nndistance, tf_approxmatch


def mlp(features, layer_dims, bn=None, bn_params=None,activation_fn=None,reuse=None):
    for i, num_outputs in enumerate(layer_dims[:-1]):
        features = tf.contrib.layers.fully_connected(
            features, num_outputs,
            normalizer_fn=bn,
            normalizer_params=bn_params,
            reuse=reuse,
            activation_fn=tf.nn.leaky_relu,
            scope='fc_%d' % i)
    outputs = tf.contrib.layers.fully_connected(
        features, layer_dims[-1],
        activation_fn=activation_fn,
        reuse=reuse,
        scope='fc_%d' % (len(layer_dims) - 1))
    return outputs


def mlp_conv(inputs, layer_dims, bn=None, bn_params=None,activation_fn=None,reuse=None):
    #layer_dims == [128,256]
    #layer_dims[:-1] == 128
    #layer_dims[-1] == 256

    #                                      128
    for i, num_out_channel in enumerate(layer_dims[:-1]):
        inputs = tf.contrib.layers.conv1d(
            inputs, num_out_channel,
            kernel_size=1,
            normalizer_fn=bn,
            normalizer_params=bn_params,
            reuse=reuse,
            activation_fn=activation_fn,
            scope='conv_%d' % i)


    #(32, 2048, 128)    
    outputs = tf.contrib.layers.conv1d(
        inputs, layer_dims[-1],
        kernel_size=1,
        activation_fn=activation_fn,
        reuse=reuse,
        scope='conv_%d' % (len(layer_dims) - 1))
    
    return outputs


#CD缺失函数
def chamfer(pcd1, pcd2):
    dist1, _, dist2, _ = tf_nndistance.nn_distance(pcd1, pcd2)
    #tf.reduce_mean()平均值
    dist1 = tf.reduce_mean(tf.sqrt(dist1))
    dist2 = tf.reduce_mean(tf.sqrt(dist2))
    return (dist1 + dist2) / 2

def chamfer1(pcd1, pcd2):
    dist1, _, dist2, _ = tf_nndistance.nn_distance(pcd1, pcd2)
    #tf.reduce_mean()平均值
    mdist1 = tf.reduce_mean(tf.sqrt(dist1))
    mdist2 = tf.reduce_mean(tf.sqrt(dist2))
    return dist1, dist2, tf.sum(mdist1), mdist2

#EMD缺失函数
def earth_mover(pcd1, pcd2):
    #assert pcd1.shape[1] == pcd2.shape[1]
    num_points = tf.cast(pcd1.shape[1], tf.float32)
    match = tf_approxmatch.approx_match(pcd1, pcd2)
    cost = tf_approxmatch.match_cost(pcd1, pcd2, match)
    return tf.reduce_mean(cost / num_points)

def add_train_summary(name, value):
    tf.summary.scalar(name, value, collections=['train_summary'])


def add_valid_summary(name, value):
    avg, update = tf.metrics.mean(value)
    tf.summary.scalar(name, avg, collections=['valid_summary'])
    return update
