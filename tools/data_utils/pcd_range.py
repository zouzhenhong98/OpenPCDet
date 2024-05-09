from glob import glob
import pyntcloud
from tqdm import tqdm
import numpy as np

root = "/hy-tmp/5165/pcd"
pcds = glob(f"{root}/*.pcd")
xx_max = []
xx_min = []
yy_max = []
yy_min = []
zz_max = []
zz_min = []
for pcd in tqdm(pcds[::10]):
    with open(pcd, "rb") as file:
        lines = file.readlines()
    if lines[0].decode("utf-8") != 'VERSION 0.7\n':
        continue
    pcd = pyntcloud.PyntCloud.from_file(pcd)
    xx = pcd.points["x"]
    yy = pcd.points["y"]
    zz = pcd.points["z"]
    xx_max.append(xx.max())
    xx_min.append(xx.min())
    yy_max.append(yy.max())
    yy_min.append(yy.min())
    zz_max.append(zz.max())
    zz_min.append(zz.min())
print(max(xx_max), min(xx_max), np.mean(xx_max))
print(max(xx_min), min(xx_min), np.mean(xx_min))
print(max(yy_max), min(yy_max), np.mean(yy_max))
print(max(yy_min), min(yy_min), np.mean(yy_min))
print(max(zz_max), min(zz_max), np.mean(zz_max))
print(max(zz_min), min(zz_min), np.mean(zz_min))
