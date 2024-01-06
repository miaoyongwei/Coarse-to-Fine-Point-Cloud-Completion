# coding=utf-8
import os
import sys
import numpy as np
import h5py
import argparse
import scipy.sparse
from sklearn.neighbors import KDTree
import multiprocessing as multiproc
from functools import partial
#import glog as logger
from copy import deepcopy
import errno
#import gdown #https://github.com/wkentaro/gdown
#import torch
import tensorflow as tf

def edges2A(edges, n_nodes, mode='P', sparse_mat_type=scipy.sparse.csr_matrix):
    '''
    note: assume no (i,i)-like edge
    edges: <2xE>
    '''
    edges = np.array(edges).astype(int)
    #shape : (2, 222186)
    
    data_D = np.zeros(n_nodes, dtype=np.float32)
    #                 [2048]
    #[0. 0. 0. 0. ... 0. 0.]
    
    #对2048个进行循环
    for d in range(n_nodes):
        data_D[ d ] = len(np.where(edges[0] == d)[0])   # compute the number of node which pick node_i as their neighbor
    #计算第i个节点被作为邻居的数量  
    #20.0    第0个节点有20次被作为邻居
    #19.0
    #18.0
    #22.0
    #...
    #18.0
    #16.0
    #18.0
    
    if mode.upper() == 'M':  # 'M' means max pooling, which use the same graph matrix as the adjacency matrix
        data = np.ones(edges[0].shape[0], dtype=np.int32)
        #[1,1,1,1,...,1,1]  shape == (37354,)
        
    elif mode.upper() == 'P':
        data = 1. / data_D[ edges[0] ]
    else:
        raise NotImplementedError("edges2A with unknown mode=" + mode)

    return sparse_mat_type((data, edges), shape=(n_nodes, n_nodes))


def knn_search(data, knn=16, metric="euclidean", symmetric=True):
    """
    Args:
      data: Nx3 , 点云坐标点
      knn: default=16
    """
    
    assert(knn>0)
    n_data_i = data.shape[0]
    #12288
    
    
    #对点云坐标点建立KDTree 距离是欧里几德距离
    kdt = KDTree(data, leaf_size=30, metric=metric)
    
    
    #对其进行邻居查寻
    # nbs[0]:NN distance,N*17. nbs[1]:NN index,N*17
    nbs = kdt.query(data, k=knn+1, return_distance=True)
    
    #零矩阵(2048,9)
    cov = np.zeros((n_data_i,9), dtype=np.float32)
    #建立一个dict()
    adjdict = dict()
    # wadj = np.zeros((n_data_i, n_data_i), dtype=np.float32)

    #遍历每一个坐标,每个坐标有16个邻居
    for i in range(n_data_i):
        # nbsd = nbs[0][i]
        # 邻居的索引
        nbsi = nbs[1][i]    #index i, N*17 YW comment
        #nbsi is 第一个是0-16的索引,16个邻居 [0  639 1218 1512 1081 1112 1823 1348 1890 1372 1472  200  114 1116 1285 1041 1683]
        
    
        #计算本地的协防差矩阵
        #获取16个邻居的 data[nbsi[1:]] 坐标  shape=[16,3]
        # data[nbsi[1:]].T                  shape=[3,16]
        # np.conv(data[nbsi[1:]].T).reshape(-1)    先计算出的是[3,3] 经过reshape后是shape=[9,]
        cov[i] = np.cov(data[nbsi[1:]].T).reshape(-1) #compute local covariance matrix
        #shape = (9,)
        
        #遍历16个邻居
        for j in range(knn):
            #如果是对称的话
            if symmetric:
                adjdict[(i, nbsi[j+1])] = 1
                adjdict[(nbsi[j+1], i)] = 1
                # wadj[i, nbsi[j + 1]] = 1.0 / nbsd[j + 1]
                # wadj[nbsi[j + 1], i] = 1.0 / nbsd[j + 1]
            else:
                adjdict[(i, nbsi[j+1])] = 1
                # wadj[i, nbsi[j + 1]] = 1.0 / nbsd[j + 1]
        # adjdict.keys() i==0  将 [(1467, 0), (0, 33), (2023, 0), (0, 47), (47, 0), (1447, 0), (0, 1531), (0, 1223), (0, 939), (0, 1447), (0, 481), (0, 1374),...,] 32个
        #np.array(list(adjdict.keys()), dtype=int) is
        #(32,2)
        '''
        [
         [   0  870]
         [   0 1205]
         [   0  223]
         [   0  879]
         [1779    0]
         ...
         [1674    0]
         [ 223    0]
         [2033    0]
        ]
        '''
            
    edges = np.array(list(adjdict.keys()), dtype=int).T
    #(2,32)
    #返回 边,距离矩阵,cov协防差矩阵
    return edges, nbs[0], cov #, wadj



def build_graph_core(ith_datai):
    try:
        xyi = ith_datai
        #shape = [12288,3]
        
        n_data_i = xyi.shape[0]
        #shape = [12288]
        
        #将点云坐标点放入knn_search()函数中,knn=16   metric="欧里几德"
        #返回边矩阵/距离矩阵/协防差矩阵
        edges, nbsd, cov = knn_search(xyi, 16, 'euclidean')
        #(2,222248) (12288,17)  (12288)
        #返回的是边,距离,和协防差矩阵
        
        #将args.mode = 'M'   还有将非零矩阵进行压缩 ,将边矩阵压缩成2048X2048
        ith_graph = edges2A(edges, n_data_i,'M', sparse_mat_type=scipy.sparse.csr_matrix)
        #(12288,12288)

        nbsd=np.asarray(nbsd)[:, 1:]
        #(12288, 16)
        #存储了每个点和16个邻居的距离

        nbsd=np.reshape(nbsd, -1)
        #(196608,)

        #return ith, ith_graph, nbsd, cov
        return ith_graph, nbsd, cov
    except KeyboardInterrupt:
        exit(-1)
