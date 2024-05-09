import os
import os.path as osp
from glob import glob
from tqdm import tqdm
import shutil
import pdb
import numpy as np
import logging

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

src_label_root = "/hy-tmp/5165/bb3d"
label_files = sorted(glob(f"{src_label_root}/*.txt"))
names = []
xx = []
yy = []
zz = []
size_x = []
size_y = []
size_z = []
size_by_name = {}
for label in tqdm(label_files):
    with open(label, "r", encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:
            parts = line.split()
            if len(parts) == 1:
                continue
            names.append(parts[0])
            xx.append(float(parts[1]))
            yy.append(float(parts[2]))
            zz.append(float(parts[3]))
            cvt_name = military_mapping[parts[0]]
            if cvt_name not in size_by_name:
                size_by_name[cvt_name] = {'x':[],'y':[],'z':[],"h":[]}
            size_by_name[cvt_name]["x"].append(float(parts[4]))
            size_by_name[cvt_name]["y"].append(float(parts[5]))
            size_by_name[cvt_name]["z"].append(float(parts[6]))
            size_by_name[cvt_name]["h"].append(float(parts[3]) - float(parts[6])/2)
print(set(names))
xx = np.array(xx)
yy = np.array(yy)
zz = np.array(zz)
print(xx.max(), xx.min())
print(yy.max(), yy.min())
print(zz.max(), zz.min())
for key, val in size_by_name.items():
    print(key, np.mean(val["x"]), np.mean(val["y"]), np.mean(val["z"]), np.mean(val['h']))