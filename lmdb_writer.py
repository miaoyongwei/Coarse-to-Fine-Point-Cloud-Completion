# coding=utf-8
# Author: Liujazz

import argparse
import os
from io_util import read_pcd
from tensorpack import DataFlow, dataflow


class pcd_df(DataFlow):
    def __init__(self, model_list, num_scans, partial_dir, complete_dir):
        self.model_list = model_list     # ['gt/airplane_0014', 'gt/airplane_0019', 'gt/airplane_0021', 'gt/airplane_0022']
        self.num_scans = num_scans       # 1
        self.partial_dir = partial_dir   #data/train_pcds/misspcds/
        self.complete_dir = complete_dir  #data/train_pcds/gt/

    def size(self):
        return len(self.model_list) * self.num_scans
    
    
    def get_data(self):
        for model_id in model_list:
            complete = read_pcd(os.path.join(self.complete_dir, '%s.pcd' % model_id))
            for i in range(self.num_scans):
                partial = read_pcd(os.path.join(self.partial_dir, '%s_break%d.pcd'%(model_id,i+1)))
                yield model_id, partial, complete
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--list_path')
    parser.add_argument('--num_scans', type=int)
    parser.add_argument('--partial_dir')
    parser.add_argument('--complete_dir')
    parser.add_argument('--output_path')
    args = parser.parse_args()

    with open(args.list_path) as file:
        model_list = file.read().splitlines()
    
    df = pcd_df(model_list, args.num_scans, args.partial_dir, args.complete_dir)
    if os.path.exists(args.output_path):
        os.system('rm %s' % args.output_path)
    dataflow.LMDBSerializer.save(df, args.output_path)
