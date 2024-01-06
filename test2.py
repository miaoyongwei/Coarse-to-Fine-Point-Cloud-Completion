import argparse
import importlib
import models
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from data_util import resample_pcd
from mpl_toolkits.mplot3d import Axes3D
from open3d import *

def FPS(xyz, npoint, RAN = True):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
    B, N, C = xyz.shape
    # centroids = tf.Variable(tf.zeros([B, npoint],dtype=tf.int64))
    distance = tf.Variable(tf.ones([B, N]) * 1e10)
    # centroid = tf.Variable(tf.zeros([B, 1, 3], dtype=tf.float32))
    out = []
    if RAN:
        farthest = tf.zeros([B,], dtype=tf.int64)
    else:
        farthest = tf.ones([B,], dtype=tf.int64)

    # batch_indices = tf.range(B, dtype=tf.int64)
    for i in range(npoint):
        print("第%d次："% i)
        # centroids = tf.assign(centroids, farthest)
        # output = tf.add(output, centroids)
        out.append(farthest)
        centroid = []
        for j in range(B):
            # centroid = tf.assign(centroid, tf.reshape(xyz[j, farthest[j], :], [B, 1, 3]))
            # centroid = tf.assign(centroid, xyz[j, farthest[j], :])
            centroid.append(xyz[j, farthest[j], :])
        centroid1 = tf.convert_to_tensor(centroid)
        centroid1 = tf.reshape(centroid1, [B, 1, 3])
        dist = tf.reduce_sum((tf.abs(xyz - tf.tile(centroid1, (1, N, 1)))) ** 2, axis=-1)
        # dist1 = tf.reshape(tf.tile(centroid, (1, B, 1)),(B, N, C))
        mask = tf.to_int64(dist < distance)
        mask1 = tf.where(mask > 0)
        # distance = tf.reshape(tf.gather_nd(dist, mask))
        # distance = tf.gather_nd(dist, mask1)
        distance = tf.scatter_nd_update(distance, mask1, tf.gather_nd(dist, mask1))
        # distance = tf.reshape(distance, [B, 3])
        farthest = tf.argmax(distance, axis=1)
    print("convert_to_tensor")
    out = tf.convert_to_tensor(out)
    out = tf.transpose(out, [1, 0])
    return out



def index_points(points, idx):
    B = points.shape[0]
    output = []
    # view_shape = list(idx.shape)
    # view_shape[1:] = [1] * (len(view_shape) - 1)
    # repeat_shape = list(idx.shape)
    # repeat_shape[0] = 1
    # batch_indices = tf.reshape(tf.tile(tf.reshape(tf.range(B, dtype=tf.int64),[B, 1]), [1, 3]), [B, 3])
    index = idx
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    indexN = index.eval(session=sess)
    #sess.close()
    print(type(points))
    print(type(idx))
    for i in range(int(B)):
        for j in range(int(idx.shape[1])):
            # print(type(int(idx.shape[1])))
            index = idx[i, j]
            # print(type(index))
            #sess = tf.Session()
            #sess.run(tf.global_variables_initializer())
            #indexN = index.eval(session=sess)
            # print(type(indexN))
            # print(i)
            # print(indexN[i, j])
            # print(sess.run(points[i, indexN[i, j], :]))
            # index = tensor_to_np(idx)
            output.append(points[i, indexN[i, j], :])
        # output = []
        # output = [points[i, idx[i, j], :] for j in range(idx.shape[1])]
            # output.append(points[i, idx[i, j], :])
            # outputs = tf.concat([output, points[i, idx[i, j], :]], axis=0)
            # tf.assign(output[i, j, 0:4], points[i, idx[i, j], 0:4])
        #
        # outputs = tf.concat(output, axis=0)
        # output.clear()
    output = tf.convert_to_tensor(output)
    output = tf.reshape(output, [B, idx.shape[1], -1])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(output))
    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='demo_data/fig6_mis/lamp_0064_break2_2.pcd')
    
    args = parser.parse_args()
    partial = read_point_cloud(args.input_path)
    partial = np.array(partial.points,dtype=np.float32)
    print(type(partial))
    partial = np.expand_dims(partial,axis=0)
    partial = tf.convert_to_tensor(partial)
    print(partial.shape)
    idx = FPS(partial, 1024, False)
    print(idx.shape)
    output = index_points(partial, idx)
    print(output.shape)
    print(type(output))
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    complete = sess.run(output)
    print(complete[0])
 
    f = open('demo_data/fig6_mis/lamp_0064_break2_2_1024.pcd', 'a')
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