import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl
import torch
import torch.nn as nn


class bn:
    def __init__(self, momentum, eps, num_features):
        self._running_mean = 0
        self._running_var = 1

        self._momentum = momentum
        self._eps = eps
        self._beta = np.zeros(shape=(num_features,))
        self._gamma = np.ones(shape=(num_features,))

    def batch_norm(self, x):
        # x_mean = tf.reduce_mean(x)
        # x_var = x.var(axis=0)
        x_mean, x_var = tf.nn.moments(x,axes=0)

        self._running_mean = (1 - self._momentum) * x_mean + self._momentum * self._running_mean
        self._running_var = (1 - self._momentum) * x_var + self._momentum * self._running_var

        x_hat = (x - x_mean) / tf.sqrt(x_var + self._eps)
        y = self._gamma * x_hat + self._beta
        return y
x = tf.random_normal([2,2,2])
conv1 = tc.layers.conv1d(
    x, 64, kernel_size=1,
    activation_fn=tf.identity
)
print(conv1.shape)
# data = np.array([[[1, 2],[2, 3]],[[3, 4],[4, 5]]]).astype(np.float32)
bn_torch = nn.BatchNorm1d(num_features=2)
# data_torch = torch.from_numpy(conv1)
# bn_output_torch = bn_torch(data_torch)
# print(bn_output_torch)
my_bn = bn(momentum=0.01, eps=0.001, num_features=2)
my_bn._beta = bn_torch.bias.detach().numpy()
my_bn._gamma = bn_torch.weight.detach().numpy()
bn_output = my_bn.batch_norm(conv1, )
print(bn_output)