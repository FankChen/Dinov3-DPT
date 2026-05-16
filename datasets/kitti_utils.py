"""
KITTI 工具函数 —— 内参解析 + velodyne → depth 投影。

本文件移植自 monodepth2/kitti_utils.py(总 98 行),所有几何逻辑保持一致;
仅添加中文注释 + 类型标注,便于回溯。
原始参考:https://github.com/nianticlabs/monodepth2/blob/master/kitti_utils.py

对齐说明:
  - load_velodyne_points  ↔ monodepth2 L8-15
  - read_calib_file       ↔ monodepth2 L18-36
  - sub2ind               ↔ monodepth2 L39-43
  - generate_depth_map    ↔ monodepth2 L46-97   (Eigen split 用的 raw GT 来源)
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Dict

import numpy as np


# ----------------------------------------------------------------------------
# 1. velodyne 点云 + 标定文件读取
# ----------------------------------------------------------------------------
def load_velodyne_points(filename: str) -> np.ndarray:
    """读取一个 .bin velodyne 点云文件,返回 (N, 4) 齐次坐标。
    每行原始为 (x_forward, y_left, z_up, reflectance);我们把反射率位置改成 1
    以便后续 4×4 矩阵相乘。"""
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # 齐次
    return points


def read_calib_file(path: str) -> Dict[str, np.ndarray]:
    """读取 KITTI 标定文本(calib_cam_to_cam.txt / calib_velo_to_cam.txt)。
    每行格式: '<key>: <space-separated floats>'。能转成 float 数组的就转,否则保留字符串。"""
    float_chars = set("0123456789.e+- ")
    data: Dict[str, np.ndarray] = {}
    with open(path, "r") as f:
        for line in f.readlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                try:
                    data[key] = np.array(list(map(float, value.split(" "))))
                except ValueError:
                    pass
    return data


def sub2ind(matrix_size, row_sub, col_sub):
    """(行,列) 下标 → 一维线性下标。仅用于 generate_depth_map 内去重。"""
    m, n = matrix_size
    return row_sub * (n - 1) + col_sub - 1


# ----------------------------------------------------------------------------
# 2. velodyne → 图像平面投影,生成稀疏深度图(Eigen split 的原始 GT)
# ----------------------------------------------------------------------------
def generate_depth_map(
    calib_dir: str,
    velo_filename: str,
    cam: int = 2,
    vel_depth: bool = False,
) -> np.ndarray:
    """
    把 velodyne 点云投影到 image_0{cam} 上,得到与原图同 shape 的稀疏深度图 (H, W),
    缺失处为 0。

    Args:
        calib_dir: 形如 '/.../kitti/2011_09_26' 的目录(含 calib_cam_to_cam.txt + calib_velo_to_cam.txt)
        velo_filename: 形如 '.../velodyne_points/data/0000000005.bin'
        cam: 2 = image_02 (左 RGB),3 = image_03 (右 RGB)
        vel_depth: True 时返回 velodyne 自身的 x 距离;False(默认,Eigen 标准)返回相机系 Z

    几何流程(monodepth2 原样):
        velo  ── R|t (velo→cam0_unrect) ──> cam0
        cam0  ── R_rect_00 ───────────────> rectified cam0
        rect  ── P_rect_0{cam} ───────────> 像素 + 深度
    """
    cam2cam = read_calib_file(os.path.join(calib_dir, "calib_cam_to_cam.txt"))
    velo2cam = read_calib_file(os.path.join(calib_dir, "calib_velo_to_cam.txt"))
    velo2cam = np.hstack((velo2cam["R"].reshape(3, 3), velo2cam["T"][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

    # 原图 shape (H, W) —— S_rect_02 存的是 (W, H),所以反转
    im_shape = cam2cam["S_rect_02"][::-1].astype(np.int32)

    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam2cam["R_rect_00"].reshape(3, 3)
    P_rect = cam2cam["P_rect_0" + str(cam)].reshape(3, 4)
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    velo = load_velodyne_points(velo_filename)
    velo = velo[velo[:, 0] >= 0, :]  # 砍掉相机后方的点(近似)

    velo_pts_im = np.dot(P_velo2im, velo.T).T
    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis]

    if vel_depth:
        velo_pts_im[:, 2] = velo[:, 0]

    # -1 是为了与 KITTI 官方 matlab 代码完全对齐
    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1] < im_shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]

    depth = np.zeros(im_shape[:2])
    depth[velo_pts_im[:, 1].astype(int), velo_pts_im[:, 0].astype(int)] = velo_pts_im[:, 2]

    # 同一像素多 LiDAR 点 → 取最近(最小深度)
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    depth[depth < 0] = 0
    return depth


# ----------------------------------------------------------------------------
# 3. 从 calib_cam_to_cam.txt 取真正的相机内参 K (3×3)
#    monodepth2 用的是手写常量内参 (0.58, 1.92, 0.5, 0.5),那是 *归一化* 后的近似,
#    适合 self-sup 多分辨率训练。我们这里两套都提供:
#      - get_normalized_K()   ↔ monodepth2 风格(分辨率无关)
#      - get_real_K(date)     从标定文件读真值(评测、复现 paper 数字时用)
# ----------------------------------------------------------------------------
def get_normalized_K() -> np.ndarray:
    """monodepth2 默认归一化内参(每个 KITTI date 都共用),返回 4×4。
    使用时要再乘 (W, H) 把它放回当前分辨率。"""
    return np.array(
        [
            [0.58, 0.00, 0.5, 0.0],
            [0.00, 1.92, 0.5, 0.0],
            [0.00, 0.00, 1.0, 0.0],
            [0.00, 0.00, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def get_real_K(calib_dir: str, cam: int = 2) -> np.ndarray:
    """从 calib_cam_to_cam.txt 读 P_rect_0{cam} 的左 3×3 当作 K。
    返回的是 *原图分辨率* (1242×375) 下的真值像素内参 (3×3, float32)。"""
    cam2cam = read_calib_file(os.path.join(calib_dir, "calib_cam_to_cam.txt"))
    P_rect = cam2cam["P_rect_0" + str(cam)].reshape(3, 4)
    return P_rect[:3, :3].astype(np.float32)
