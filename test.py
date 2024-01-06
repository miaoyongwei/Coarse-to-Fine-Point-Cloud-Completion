# import numpy as np
# test = np.load('./00019_00183_indoors_300_010_depth.npy',encoding="latin1")
# doc = open('2.txt','a')
# print(test,file=doc)


# import numpy as np
# input_data = np.load(r"./00019_00183_indoors_300_010_depth_mask.npy")
# print(input_data.shape)
# data = input_data.reshape(1,-1)
# print(data.shape)
# print(data)
# np.savetxt(r"test.txt",data,delimiter=',')

# import numpy as np
# import scipy.misc
# image = np.load('./00019_00183_indoors_300_010_depth.npy')
# print(image.shape)
# scipy.misc.imsave('test1.png',image)

# import numpy as np
# np.set_printoptions(suppress=True)
# # 作用是取消numpy默认的科学计数法，测试表明open3d点云读取函数没法读取科学计数法的表示
# import open3d as o3d
# data = np.load('00019_00183_indoors_300_010_depth.npy').reshape(1, -1)
# txt_data = np.savetxt('scene1.txt', data)
# pcd = o3d.read_point_cloud('scene1.txt', format='xyz')
# # 此处因为npy里面正好是 x y z r g b的数据排列形式，所以format='xyzrgb'
# print(pcd)
# o3d.visualization.draw_geometries([pcd], width=1200, height=600) # 可视化点云

import tensorflow as tf
import numpy as np
from tf_util import *
def mlp_conv(inputs, layer_dims, bn=None, bn_params=None):
    for i, num_out_channel in enumerate(layer_dims[:-1]):
        inputs = tf.contrib.layers.conv1d(
            inputs,
            num_out_channel,
            kernel_size=1,
            normalizer_fn=bn,
            normalizer_params=bn_params,
            scope='conv_%d' % i)
    outputs = tf.contrib.layers.conv1d(
        inputs,
        layer_dims[-1],
        kernel_size=1,
        activation_fn=None,
        scope='conv_%d' % (len(layer_dims) - 1))
    return outputs


def point_maxpool(inputs, npts, keepdims=False):
    outputs = [
        tf.reduce_mean(
            tf.transpose(
                tf.math.top_k(tf.transpose(f, perm=[0, 2, 1]), k=1).values,
                perm=[0, 2, 1]),
            axis=1,
            keepdims=True) for f in tf.split(inputs, npts, axis=1)
    ]
    if keepdims:
        # print(tf.concat(outputs, axis=0).shape)
        return tf.concat(outputs, axis=0)
    else:
        # print(tf.squeeze(tf.concat(outputs, axis=0), axis=1).shape)
        return tf.squeeze(tf.concat(outputs, axis=0), axis=1)

# def point_maxpool(inputs, npts, keepdims=False):
#     outputs = [tf.reduce_max(f, axis=1, keepdims=keepdims)
#         for f in tf.split(inputs, npts, axis=1)]
#     print(tf.concat(outputs, axis=0).shape)
#     return tf.concat(outputs, axis=0)

def point_unpool(inputs, npts):
    inputs = tf.split(inputs, inputs.shape[0], axis=0)
    outputs = [tf.tile(f, [1, npts[i], 1]) for i, f in enumerate(inputs)]
    # print(tf.concat(outputs, axis=0).shape)
    # print(tf.concat(outputs, axis=1).shape)
    # print(tf.concat(outputs, axis=2).shape)
    return tf.concat(outputs, axis=1)


def point_softpool(inputs, npts_output, orders):
    inputs = tf.nn.softmax(inputs, axis=-1)
    # inputs_ordered = []
    # for idx in range(orders):
    #     idx_reg = tf.math.top_k(
    #         inputs[:, :, idx], k=npts_output // orders).indices
    #     # print(idx_reg.shape)
    #     inputs_ordered.append(tf.gather(inputs, indices=idx_reg, axis=1))
    # # print(tf.concat(inputs_ordered[:], axis=1).shape)
    # inputs_ordered = tf.concat(inputs_ordered[:], axis=0)
    # inputs_ordered = tf.reduce_max(inputs_ordered, axis=1)
    # return inputs_ordered
    output = []
    for j in range(orders):
        idx = tf.math.top_k(inputs[:, :, j], k=npts_output // orders).indices
        for i in range(inputs.shape[0]):
            # print(sess.run(idx))
            # print(sess.run(idx[i]))
            #
            # print(sess.run(inputs[i]))
            output.append(tf.gather(inputs[i], indices=idx[i], axis=0))
            # output1.append(tf.gather(input))
            # print(sess.run(output))

        # print(output)
    outputs = tf.concat(output[:], axis=0)
    outputs = tf.reshape(outputs, [inputs.shape[0], -1, inputs.shape[2]])
    return outputs


def mlp(features, layer_dims, bn=None, bn_params=None):
    for i, num_outputs in enumerate(layer_dims[:-1]):
        features = tf.contrib.layers.fully_connected(
            features,
            num_outputs,
            normalizer_fn=bn,
            normalizer_params=bn_params,
            scope='fc_%d' % i)
    outputs = tf.contrib.layers.fully_connected(
        features,
        layer_dims[-1],
        activation_fn=None,
        scope='fc_%d' % (len(layer_dims) - 1))
    # print(outputs.shape)
    return outputs

def mlp_conv_act(inputs, layer_dims, act_dim=8, bn=None, bn_params=None):
    for i, num_out_channel in enumerate(layer_dims[:-1]):
        inputs = tf.contrib.layers.conv1d(
            inputs,
            num_out_channel,
            kernel_size=1,
            rate=2,
            normalizer_fn=bn,
            normalizer_params=bn_params,
            scope='conv_%d' % i)
    feature = tf.contrib.layers.conv1d(
        inputs,
        layer_dims[-1],
        kernel_size=1,
        activation_fn=None,
        scope='conv_%d' % (len(layer_dims) - 1))
    act = tf.contrib.layers.conv1d(
        inputs, act_dim, kernel_size=1, activation_fn=None, scope='conv_act')
    outputs = tf.concat([feature, act], axis=-1)
    # print(outputs.shape)
    return outputs
# print(npts.shape)
# s = point_maxpool(inputs, npts, keepdims=True)
# point_unpool(s, npts)
# for f in tf.split(inputs, npts, axis=1):
#     # print(tf.shape(f))
#     outputs = [tf.reduce_mean(tf.transpose(tf.math.top_k(tf.transpose(f, [0, 2, 1]), k=1).values, [0, 2, 1]), axis=1, keepdims=True)]


# input = tf.constant(np.random.rand(32, 540, 256))
# output = tf.math.top_k(input, 1)
# with tf.Session() as sess:
#     print(sess.run(input))
#     print(sess.run(output))

# input = tf.constant(np.random.rand(32, 540, 256))
# f1, f2= tf.split(input, [270, 270], axis=1)
# print(f1.shape)
# print(f2.shape)

def create_encoder(inputs, npts):
    with tf.variable_scope('encoder_0', reuse=tf.AUTO_REUSE):
        features = mlp_conv(inputs, [128, 256])  # (32, 540, 256)
        features_global = point_unpool(
            point_maxpool(features, npts, keepdims=True), npts)
        features = tf.concat([features, features_global], axis=2)
    with tf.variable_scope('encoder_1', reuse=tf.AUTO_REUSE):
        features = mlp_conv(features, [512, 1024])
        print(features.shape)
        features = point_maxpool(features, npts)
    print(features.shape)
    return features


def create_decoder(features):
    with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
        coarse = mlp(features,
                     [1024, 1024, 256 * 8])
        coarse = tf.reshape(coarse, [-1, 256, 8])
        print(coarse.shape)
        feat_spool = point_softpool(
            coarse, npts_output=256, orders=8)
        print(feat_spool.shape)
        coarse = mlp_conv_act(
            feat_spool, [512, 512, 3], act_dim=8)
        print(coarse.shape)
        coarse = tf.concat([coarse[:, :, :3], feat_spool[:, :, :]],
                           axis=-1)
        print(coarse.shape)

    with tf.variable_scope('folding', reuse=tf.AUTO_REUSE):
        grid = tf.meshgrid(
            tf.linspace(-0.05, 0.05, 2),
            tf.linspace(-0.05, 0.05, 2))
        # print(grid.shape)
        grid = tf.expand_dims(
            tf.reshape(tf.stack(grid, axis=2), [-1, 2]), 0)
        # print(grid.shape)
        grid_feat = tf.tile(grid, [features.shape[0], 256, 1])
        # print(grid_feat.shape)
        point_feat = tf.tile(
            tf.expand_dims(coarse, 2), [1, 1, 2**2, 1])
        # print(point_feat.shape)
        point_feat = tf.reshape(point_feat,
                                [-1, 1024, 3 + 8])
        # print(point_feat.shape)
        global_feat = tf.tile(
            tf.expand_dims(features, 1), [1, 1024, 1])
        # print(global_feat.shape)
        feat = tf.concat([grid_feat, global_feat, point_feat], axis=2)
        # print(feat.shape)
        feat_spool = tf.tile(
            tf.expand_dims(feat_spool, 2), [1, 1, 2**2, 1])
        print(feat_spool.shape)
        feat_spool = tf.reshape(feat_spool,
                                [-1, 1024, 8])
        print(feat_spool.shape)
        center = tf.tile(
            tf.expand_dims(coarse, 2), [1, 1, 2**2, 1])
        # print(center.shape)
        center = tf.reshape(center, [-1, 1024, 3 + 8])
        # print(center.shape)
        fine = mlp_conv_act(
            feat, [512, 512, 3], act_dim=8)  # + center
        print(fine.shape)
        fine = tf.concat([fine[:, :, :3], feat_spool], axis=-1)
        print(fine.shape)
        mesh = fine

    return coarse, fine, mesh

if __name__ == '__main__':
    # input = tf.random_uniform([16, 2048], 0, 1)
    # mlp(input, [1024, 1024, 2048])
    # coarse = tf.reshape(input, [-1, 256, 8])
    # print(coarse.shape)
    input = tf.random_uniform([16, 1024], 0, 1)
    npts = tf.placeholder(tf.int32, (16, ), 'num_points')
    # create_encoder(input, npts)
    create_decoder(input)
    # f = [tf.reduce_mean(tf.transpose(tf.math.top_k(tf.transpose(f, perm=[0, 2, 1]), k=1).values, perm=[0, 2, 1]), axis=1, keepdims=True) for f in tf.split(input, npts, axis=1)]
    # f = tf.concat(f, axis=1)
    # print(f.shape)
    # npts = tf.placeholder(tf.int32, (16,), 'num_points')
    # f1 = point_unpool(f, npts)
    # f1 = tf.concat([f, f1], axis=1)
    # print(f1.shape)
    # f1 = mlp_conv(f1, [512, 1024])
    # print(f1.shape)
    # f1 = point_maxpool(f1, npts)
    # print(f1.shape)
    # f1 = mlp(f1, [1024, 1024, 256 * 8])
    # print(f1.shape)
    # f1 = tf.reshape(f1, [-1, 256, 8])
    # print(f1.shape)
    # f1 = point_softpool(f1, 256, 8)
    # print(f1.shape)
    # f1 = mlp_conv_act(f1, [512, 512, 3], 8)
    # print(f1.shape)
    # point_softpool(coarse, 256, 8)
    # mlp_conv_act(input, [512, 512, 3], 8)

    # inputs = tf.random_uniform([3, 4, 5], 0, 1)
    # inputs = tf.constant([[[1, 2, 3, 4, 5], [2, 3, 4, 5, 1], [3, 4, 5, 1, 2], [4, 5, 1, 2, 3]], [[2, 1, 3, 4, 5], [4, 2, 4, 5, 1], [5, 3, 5, 1, 2], [2, 1, 1, 2, 3]]])
    inputs = tf.random_uniform([16, 256, 8], 0, 1)
    output = []
    output1 = []
    output2 = []
    # print(inputs.shape[0])
    # print(inputs.shape)
    with tf.Session() as sess:
        # print(sess.run(inputs))
        for j in range(8):
            idx = tf.math.top_k(inputs[:, :, j], k=32).indices
            for i in range(inputs.shape[0]):
                # print(sess.run(idx))
                # print(sess.run(idx[i]))
                #
                # print(sess.run(inputs[i]))
                output.append(tf.gather(inputs[i], indices=idx[i], axis=0))
                # output1.append(tf.gather(input))
                # print(sess.run(output))

            # print(output)
        outputs = tf.concat(output[:], axis=0)
        outputs = tf.reshape(outputs, [inputs.shape[0], -1, inputs.shape[2]])
        # print(sess.run(outputs))
        # print(outputs.shape)

        # outputs = tf.reduce_max(outputs, axis=2)
        # print(sess.run(outputs))
        # print(outputs.shape)