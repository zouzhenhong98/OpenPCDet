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
# root = "/home/OpenPCDet"
root = "/hy-tmp"
custom_root = osp.join(root, "data/custom")
index_folder = osp.join(custom_root, "ImageSets")
point_folder = osp.join(custom_root, "points")
label_folder = osp.join(custom_root, "labels")
os.makedirs(custom_root, exist_ok=True)
os.makedirs(index_folder, exist_ok=True)
os.makedirs(point_folder, exist_ok=True)
os.makedirs(label_folder, exist_ok=True)


# source data
src_point_root = "/hy-tmp/datasets/pcd"
src_label_root = "/hy-tmp/datasets/bb3d"
sample_num = 100000
point_files = sorted(glob(f"{src_point_root}/*.pcd"))
sample_step = int(len(point_files) / sample_num)
data_names = []
for pfile in tqdm(point_files[::max(sample_step, 1)]):
    pname = osp.basename(pfile).split(".")[0]
    label = osp.join(src_label_root, f"{pname}.txt")
    dst_pfile = osp.join(point_folder, osp.basename(pfile))
    dst_label = osp.join(label_folder, osp.basename(label))
    if osp.exists(dst_pfile.replace(".pcd", '.npy')) and osp.exists(dst_label):
        data_names.append(pname)
        continue
    with open(pfile, "rb") as file:
        lines = file.readlines()
    if lines[0].decode("utf-8") != 'VERSION 0.7\n':
        continue
    # process
    try:
        # label
        with open(label, "r", encoding="utf-8") as file:
            lines = file.readlines()
        new_boxes = []
        if len(lines) == 0:
            continue
        if len(lines) == 1 and lines[1].split() < 2:
            continue
        for line in lines:
            parts = line.split()
            parts[0] = military_mapping[parts[0]]
            # notice, original annotaion follows which coordinates
            new_line = " ".join([parts[i] for i in [1, 2, 3, 4, 5, 6, 9, 0]]) + "\n"
            new_boxes.append(new_line)
        with open(dst_label, "w", encoding="utf-8") as file:
            file.writelines(new_boxes)
        # pcd2npy
        # shutil.copyfile(pfile, dst_pfile)
        cloud = PyntCloud.from_file(pfile)
        points = cloud.points[['x', 'y', 'z', 'intensity']].values
        npy_path = dst_pfile.replace('.pcd', '.npy')
        np.save(npy_path, points)
    except Exception as exp:
        logging.error(label)
        continue
    data_names.append(pname)
data_index = list(range(len(data_names)))
random.shuffle(data_index)
train_num = int(len(data_index) * 0.8)
train_index = data_index[:train_num]
val_index = data_index[train_num:]
train_names = [data_names[idx] for idx in train_index] 
val_names = [data_names[idx] for idx in val_index] 
with open(osp.join(index_folder, "train.txt"), "w", encoding="utf-8") as file:
    for name in train_names:
        file.write(name + "\n")
with open(osp.join(index_folder, "val.txt"), "w", encoding="utf-8") as file:
    for name in val_names:
        file.write(name + "\n")
