import tensorflow as tf
import numpy as np
from tf_util import chamfer,earth_mover, chamfer1
from open3d import *
from data_util import resample_pcd
 
def distance_matrix(array1, array2):
    """
    arguments:
        array1: the array, size: (num_point, num_feature)
        array2: the samples, size: (num_point, num_feature)
    returns:
        distances: each entry is the distance from a sample to array1
            , it's size: (num_point, num_point)
    """
    num_point, num_features = array1.shape
    expanded_array1 = tf.tile(array1, (num_point, 1))
    expanded_array2 = tf.reshape(
            tf.tile(tf.expand_dims(array2, 1),
                    (1, num_point, 1)),
            (-1, num_features))
    distances = tf.norm(expanded_array1-expanded_array2, axis=1)
    distances = tf.reshape(distances, (num_point, num_point))
    return distances
 
def av_dist(array1, array2):
    """
    arguments:
        array1, array2: both size: (num_points, num_feature)
    returns:
        distances: size: (1,)
    """
    distances = distance_matrix(array1, array2)
    distances = tf.reduce_min(distances, axis=1)
    distances = tf.reduce_mean(distances)
    return distances
 
def av_dist_sum(arrays):
    """
    arguments:
        arrays: array1, array2
    returns:
        sum of av_dist(array1, array2) and av_dist(array2, array1)
    """
    array1, array2 = arrays
    av_dist1 = av_dist(array1, array2)
    av_dist2 = av_dist(array2, array1)
    return av_dist1+av_dist2
 
def chamfer_distance_tf(array1, array2):
    batch_size, num_point, num_features = array1.shape
    dist = tf.reduce_mean(
               tf.map_fn(av_dist_sum, elems=(array1, array2), dtype=tf.float32)
           )
    return dist
 
def array2samples_distance(array1, array2):
    """
    arguments:
        array1: the array, size: (num_point, num_feature)
        array2: the samples, size: (num_point, num_feature)
    returns:
        distances: each entry is the distance from a sample to array1
    """
    num_point, num_features = array1.shape
    expanded_array1 = np.tile(array1, (num_point, 1))
    expanded_array2 = np.reshape(
            np.tile(np.expand_dims(array2, 1),
                    (1, num_point, 1)),
            (-1, num_features))
    distances = np.linalg.norm(expanded_array1-expanded_array2, axis=1)
    distances = np.reshape(distances, (num_point, num_point))
    distances = np.min(distances, axis=1)
    distances = np.mean(distances)
    return distances
 
def chamfer_distance_numpy(array1, array2):
    batch_size, num_point, num_features = array1.shape
    dist = 0
    for i in range(batch_size):
        av_dist1 = array2samples_distance(array1[i], array2[i])
        av_dist2 = array2samples_distance(array2[i], array1[i])
        dist = dist + (av_dist1+av_dist2)/batch_size
    return dist
 
if __name__=='__main__':
    np.random.seed(1)
    data1 = read_point_cloud('demo_data/saum_data/add/saum3/airplane_6_break_pcn.pcd')
    data1 = np.array(data1.points,dtype=np.float32)
    data1 = resample_pcd(data1,2048)
    #print(data1)
    data1 = np.expand_dims(data1,axis=0)
    #print(data1)

    data4 = read_point_cloud('demo_data/saum_data/add/saum3/airplane_6.pcd')
    data4 = np.array(data4.points,dtype=np.float32)
    #data4 = resample_pcd(data4,1000)
    data4 = np.expand_dims(data4,axis=0)
    #print (array1)
    #print(array2)
    print(chamfer_distance_numpy(data1, data4))
    with tf.Session() as sess:
        print(sess.run(earth_mover(data1, data4)))
