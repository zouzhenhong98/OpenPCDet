import copy
import os
import os.path as osp
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from uuid import UUID

import argparse
import cv2
import numpy as np
from pyntcloud import PyntCloud

def normailize_color(
    color_mat: np.ndarray, balance_color: Optional[bool] = True
) -> np.ndarray:
    """Normalize color matrix to [0, 255] and balance the color

    Args:
        color_mat (np.ndarray): color matrix, shape is (N, 3)
        balance_color (Optional[bool], optional): balance color distribution.
            Defaults to True.

    Returns:
        np.ndarray: normalized color matrix, shape is (N, 3)
    """
    if color_mat.shape[0] == 0:
        return color_mat
    color_mat = np.copy(color_mat)
    color_mat -= color_mat.min()
    color_mat /= color_mat.max()
    if balance_color:  # balance distribution for color
        if np.mean(color_mat) > 0.5:
            color_mat = 1 - color_mat
        color_mat = np.sqrt(color_mat)
    color_mat *= 255
    return np.clip(color_mat, 0, 255)

def read_boxes(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        lines = file.readlines()
    boxes = []
    for line in lines:
        box = line.strip('\n').split(' ')
        if len(box) == 8:
            box = [float(elem) for elem in box[:-1]] + [box[-1]]
        elif len(box) == 10:
            new_box = [float(box[i]) for i in [1, 2, 3, 4, 5, 6, 9]] + [box[0]]
            box = new_box
        else:
            raise
        boxes.append(box)
    print(boxes[0], boxes[-1])
    return boxes

def rotate_point(point, center, angle_degrees):
    """旋转点"""
    angle_radians = np.radians(angle_degrees)
    c, s = np.cos(angle_radians), np.sin(angle_radians)
    R = np.array(((c, -s), (s, c)))
    point -= center
    rotated_point = np.dot(R, point)
    return rotated_point + center

def get_box_corners(cx, cy, dx, dy, yaw):
    """根据中心点、宽高和偏航角计算矩形四角坐标"""
    # 未旋转时的顶点相对于中心点的坐标
    corners_unrotated = np.array([
        [cx-dx/2, cy+dy/2],
        [cx+dx/2, cy+dy/2],
        [cx+dx/2, cy-dy/2],
        [cx-dx/2, cy-dy/2]
    ])
    # return corners_unrotated
    
    # 旋转每个顶点
    corners_rotated = np.array([rotate_point(corner, (cx, cy), yaw * 180 / np.pi) for corner in corners_unrotated])
    return corners_rotated
    

def box_to_xycoor(box3d):
    # box3d: cx cy cz dx dy dz yaw cls
    coors = get_box_corners(
        cx=box3d[0],
        cy=box3d[1],
        dx=box3d[3],
        dy=box3d[4],
        yaw=box3d[6],
    )
    return coors# + [box3d[-1]]



def save_pointxy_to_jpg(
    points: np.ndarray,
    save_path: str,
    resolution: Optional[float] = 0.1,
    fix_screen_size: Optional[float] = 1000,
    move_front_lidar_center: Optional[bool] = False,
    box_list: Optional[List] = None,
) -> None:
    """save pointxy to image format

    Args:
        points (np.ndarray): points with x, y, z
        save_path (str): save path
        resolution (float): image resolution ( the size of grid or pixel).
            Defaults to 0.1(m)
        fix_screen_size (Optional[float], optional): fix screen size.
            Defaults to 1000.
        move_front_lidar_center:: move front lidar center or not.
            Defaults to False. Use it for INNO
    """
    points[:, 0] /= resolution
    points[:, 1] /= resolution
    width, height = fix_screen_size, fix_screen_size
    half_width, half_height = width // 2, height // 2
    if not move_front_lidar_center:
        points[:, 0] += half_width
    points[:, 1] += half_height
    valid_x = (points[:, 0] >= 0) & (points[:, 0] < width)
    valid_y = (points[:, 1] >= 0) & (points[:, 1] < height)
    valid_indices = valid_x & valid_y
    points = points[valid_indices]
    xaxis = points[:, 0].astype(np.int32)
    yaxis = points[:, 1].astype(np.int32)
    img = np.zeros((height, width, 1), dtype=np.uint8)
    color = normailize_color(points[:, 2])
    img[yaxis, xaxis] = np.expand_dims(color, axis=1).astype(np.uint8)
    zero_mask = (img == 0)[:, :, 0]
    color = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    color[zero_mask] = [0, 0, 0]
    if box_list is not None:
        def adapt_coor(coor):
            coor /= resolution
            if not move_front_lidar_center:
                coor[0] += half_width
            coor[1] += half_height
            return tuple(coor.astype(int))
        
        box_color = (0, 0, 255)
        thickness = 2
        for box in box_list:
            if np.any(np.isinf(box)) or np.any(np.isnan(box)):
                continue
            invalid_flag = False
            for item in box:
                if abs(item) > 200:
                    invalid_flag = True
                    break
            if invalid_flag:
                continue
            corners = box_to_xycoor(box)
            # print(box, corners, [adapt_coor(copy.deepcopy(cor)) for cor in corners])
            for i in range(4):
                next_i = (i + 1) % 4
                cv2.line(
                    color, 
                    adapt_coor(copy.deepcopy(corners[i])),
                    adapt_coor(copy.deepcopy(corners[next_i])),
                    box_color, 
                    thickness
                )
    filename, _ = osp.splitext(osp.basename(save_path))
    webp_path = osp.join(osp.dirname(save_path), f"{filename}.webp")
    cv2.imwrite(
        webp_path, color, [cv2.IMWRITE_WEBP_QUALITY, 40]
    )


if __name__=="__main__":
    index = 0
    pcd_file = f"/hy-tmp/data/custom/points/{index}.npy"
    label_file = f"/hy-tmp/data/custom/labels/{index}.txt"
    # pcd_file = "/hy-tmp/datasets/交付数据_5165/pcd/175123.pcd"
    # label_file = "/hy-tmp/datasets/交付数据_5165/bb3d/175123.txt"
    # pcd_file = "/hy-tmp/datasets/SyntheticDataLogger08(12.023k)/pcd/803871.pcd"
    # label_file = "/hy-tmp/datasets/SyntheticDataLogger08(12.023k)/bb3d/803871.txt"
    # pcd_file = "/hy-tmp/datasets/SyntheticDataLogger07(8.223k)/pcd/10093.pcd"
    # label_file = "/hy-tmp/datasets/SyntheticDataLogger07(8.223k)/bb3d/10093.txt"
    if pcd_file.endswith("pcd"):
        cloud = PyntCloud.from_file(pcd_file)
        point = cloud.points[['x', 'y', 'z', 'intensity']].values
    else:
        point = np.load(pcd_file)
    save_path = f"vis_{index}.jpg"
    boxes = read_boxes(label_file)
    save_pointxy_to_jpg(point, save_path, move_front_lidar_center=True, box_list=boxes)