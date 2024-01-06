import h5py
import os


filename = '/root/autodl-tmp/wrq1_0527/demo_data/saum_data/add/h5'
filename1 = '/root/autodl-tmp/wrq1_0527/demo_data/saum_data/add'
dirList = os.listdir(filename)
i = 1
for name in dirList:
    print(i)
    i += 1
    name1, _ = name.split('.')
    name1 += '.pcd'
    filename2 = os.path.join(filename, name)
    f = h5py.File(filename2, 'r')
    data = f['data']
    # print(f.keys())
    # data = f['data']
    filename3 = os.path.join(filename1, name1)
    if os.path.exists(filename3):
        os.remove(filename3)
    #
    out = open(filename3, 'a')
    point_num = data.shape[0]
    # headers
    out.write(
        '# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1')
    string = '\nWIDTH ' + str(point_num)
    out.write(string)
    out.write('\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0')
    string = '\nPOINTS ' + str(point_num)
    out.write(string)
    out.write('\nDATA ascii')
    #
    # # datas
    for i in range(point_num):
        string = '\n' + str(data[i, 0]) + ' ' + str(data[i, 1]) + ' ' + str(data[i, 2])
        #print(str(data[i, 0][0]))
        out.write(string)

    out.close()

    # print(name1)
# f = h5py.File(filename, 'r')
#
# index_start = 0
# index_end = 1
# data = f['data']
#
#
#
# path = 'D:/data2048/shapenet/test/partial/seg1.pcd'
# if os.path.exists(path):
#     os.remove(path)
#
# out = open(path, 'a')
# point_num = data.shape[0]
# # headers
# out.write(
#     '# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1')
# string = '\nWIDTH ' + str(point_num)
# out.write(string)
# out.write('\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0')
# string = '\nPOINTS ' + str(point_num)
# out.write(string)
# out.write('\nDATA ascii')
#
# # datas
# for i in range(point_num):
#     string = '\n' + str(data[i, 0]) + ' ' + str(data[i, 1]) + ' ' + str(data[i, 2])
#     out.write(string)
#
# out.close()