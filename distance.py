from tf_util import chamfer,earth_mover
from open3d import *
import numpy as np
import tensorflow as tf



data1 = read_point_cloud('demo_data/fig6_mis/airplane_0111.pcd')
data1 = np.array(data1.points,dtype=np.float32)
data1 = np.expand_dims(data1,axis=0)

data2 = read_point_cloud('demo_data/fig6_mis/airplane_0111_break2_ours_1024.pcd')
data2 = np.array(data2.points,dtype=np.float32)
data2 = np.expand_dims(data2,axis=0)

data3 = read_point_cloud('demo_data/cdemd/lamp_0070_break5_earthdis.pcd')
data3 = np.array(data3.points,dtype=np.float32)
data3 = np.expand_dims(data3,axis=0)

data4 = read_point_cloud('demo_data/cdemd/airplane_0055_break4_ps.pcd')
data4 = np.array(data4.points,dtype=np.float32)
data4 = np.expand_dims(data4,axis=0)


distance_earthmove = earth_mover(data1, data2)
#distance_pcn = earth_mover(data1, data3)
#distance_ps = earth_mover(data1, data4)

sess = tf.Session()

distance_earthmove = sess.run(distance_earthmove)
#distance_pcn = sess.run(distance_pcn)
#distance_ps = sess.run(distance_ps)


print(distance_earthmove)
#print(distance_pcn)
#print(distance_ps)

