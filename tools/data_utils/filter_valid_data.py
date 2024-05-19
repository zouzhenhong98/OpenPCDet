import os
import os.path as osp
from glob import glob
from tqdm import tqdm
import shutil
import pdb
import logging
from pyntcloud import PyntCloud
import numpy as np
import random


military_mapping = {
    'Charlie': 'Soldier',
    'Delta': 'Soldier',
    'Foxtrot': 'Soldier',
    'Alfa': 'Soldier',
    'ZTZ99A': 'MBT',
    'PGZ09': 'FV',
    'T72B3': 'MBT',
    'Mar1a3': 'FV',
    'T90A': 'MBT',
    'M1A2SEP': 'MBT',
    'ZTL11': 'FV',
    'HMARS': 'FV',  
    'Car': 'Car',
    '2s3m': 'Artillery',
    'M2A3': 'FV',
    'ZTZ96A': 'MBT',
    'M109': 'Artillery'
}

# init folder
root = "/hy-tmp"
custom_root = osp.join(root, "data/custom")
index_folder = osp.join(custom_root, "ImageSets")
point_folder = osp.join(custom_root, "points")
label_folder = osp.join(custom_root, "labels")

def filter_names(txt_file):
    with open(txt_file, "r") as file:
        names = file.readlines()
    names = [name.strip("\n") for name in names]
    print(f"Get {len(names)} from {txt_file}")
    filter_names = []
    for name in tqdm(names):
        pcd_file = osp.join(point_folder, f"{name}.npy")
        try:
            np.load(pcd_file)
        except Exception as exp:
            print(exp)
            continue
        filter_names.append(name) 
    print(f"filter {len(filter_names)} files")
    with open(txt_file+'.new', "w", encoding="utf-8") as file:
        for name in filter_names:
            file.write(name + "\n")

src_point_root = "/hy-tmp/datasets/pcd"
src_label_root = "/hy-tmp/datasets/bb3d"
filter_names(osp.join(index_folder, "train.txt"))
filter_names(osp.join(index_folder, "val.txt"))