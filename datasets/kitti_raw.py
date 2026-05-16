"""
KITTI Raw 数据集 —— 同时支持 self-sup 训练 (eigen_zhou) 与评测 (eigen / eigen_benchmark)。

设计思路(与 monodepth2 的差异说明):
─────────────────────────────────────────────────────────────────
1. 我们只面向 DINOv3 DPT(没有多尺度金字塔需求),因此 *砍掉* 了 monodepth2 的
   num_scales 多尺度 resize,简化为单一目标尺寸 (H, W)。photometric loss 用
   原尺度即可。

2. 标准化策略:dataset 输出的 color tensor 仍是 [0, 1] 范围,*不* 在这里做
   ImageNet mean/std 归一化 —— 归一化放到 model 的 encoder 入口去做,这样
   photometric loss(SSIM + L1)用未归一化的图像,符合 monodepth2 习惯。
   而 DINOv3 backbone 内部再自行 ImageNet norm。

3. 评测用 GT:
   - 'eigen'           → 用 velodyne 投影出的稀疏 depth(garg_crop)
   - 'eigen_benchmark' → 用 data_depth_annotated/{val,train}/ 的 improved GT
   这两套数字 *都报*(见拍板项 ①)。

─────────────────────────────────────────────────────────────────
★ 拍板项(用户已确认,凡涉及之处用 [PIN-N] 注释明确标出)
   [PIN-1] test split 双口径:eigen(697,raw velodyne)+ eigen_benchmark(652, improved)
   [PIN-2] 训练分辨率分两阶段:先 192×640 通流程,再 768×1024 出真数字
            ↳ 由调用方传 (height, width) 参数控制;此 dataset 本身无偏好
   [PIN-3] scale align 双口径:self-sup 用 median;SYNTHMIX 复现用 least-squares
            ↳ 该选择发生在 evaluator,*不* 在 dataloader 里;此处仅备注
─────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import PIL.Image as pil
import torch
import torch.utils.data as data
from torchvision import transforms

from . import kitti_utils as ku


# 集群上数据的固定根目录(可被构造函数覆盖)
DEFAULT_KITTI_ROOT = "/home/izi2sgh/MYDATA/kitti"
DEFAULT_ANNO_ROOT = "/home/izi2sgh/MYDATA/kitti/data_depth_annotated"

# KITTI 原图分辨率(全部 5 个 date 都一样:1242×375)—— 用于 GT depth 上采样
KITTI_FULL_RES_WH = (1242, 375)  # (W, H),与 monodepth2 一致


def _pil_loader(path: str) -> "pil.Image":
    with open(path, "rb") as f:
        with pil.open(f) as img:
            return img.convert("RGB")


def read_split_file(split_path: str) -> List[str]:
    """读取一个 train_files.txt / val_files.txt / test_files.txt。
    每行三列:  <date>/<drive>  <frame_index>  <l|r>"""
    with open(split_path, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    return lines


# ============================================================================
# 主数据集
# ============================================================================
class KITTIRawDataset(data.Dataset):
    """
    输出字典 (与 monodepth2 keys 兼容,scale 维度恒为 0):
        ("color",     0, 0)       (3, H, W) float[0,1]    —— 当前帧
        ("color_aug", 0, 0)       (3, H, W) float[0,1]    —— 当前帧 + color jitter
        ("color",     -1, 0)      (3, H, W) float[0,1]    —— 前一帧 (仅 train)
        ("color",     +1, 0)      (3, H, W) float[0,1]    —— 后一帧 (仅 train)
        ("K", 0)                  (4, 4) float32          —— 缩放到 (H, W) 后的像素内参
        ("inv_K", 0)              (4, 4) float32
        "depth_gt"                (1, H_full, W_full)     —— eval 时才有;原图分辨率不下采
        "frame_meta"              str                     —— 调试用:'<date>/<drive>/<10d>/<side>'
    """

    def __init__(
        self,
        data_path: str = DEFAULT_KITTI_ROOT,
        anno_path: str = DEFAULT_ANNO_ROOT,
        split_file: str = "",
        height: int = 192,         # [PIN-2] stage-1 默认 192,stage-2 切到 768
        width: int = 640,          # [PIN-2] stage-1 默认 640,stage-2 切到 1024
        frame_idxs: Tuple[int, ...] = (0, -1, 1),  # monodepth2 默认 ±1 邻帧
        is_train: bool = True,
        gt_source: str = "none",   # 'none' | 'velodyne' | 'improved'  [PIN-1]
        img_ext: str = ".png",     # 集群上是 .png(monodepth2 默认 .jpg)
        color_jitter: bool = True,
    ):
        super().__init__()
        self.data_path = data_path
        self.anno_path = anno_path
        self.height = height
        self.width = width
        self.frame_idxs = tuple(frame_idxs)
        self.is_train = is_train
        self.gt_source = gt_source
        self.img_ext = img_ext
        self.color_jitter_enabled = color_jitter

        assert gt_source in ("none", "velodyne", "improved"), f"bad gt_source={gt_source}"
        assert os.path.isfile(split_file), f"split file not found: {split_file}"
        self.filenames = read_split_file(split_file)

        # 输入 resize (LANCZOS,monodepth2 同款)
        self.resize_color = transforms.Resize(
            (self.height, self.width), interpolation=transforms.InterpolationMode.LANCZOS
        )
        self.to_tensor = transforms.ToTensor()

        # color jitter 参数与 monodepth2 完全一致(见 mono_dataset.py L67-78)
        self._brightness = (0.8, 1.2)
        self._contrast = (0.8, 1.2)
        self._saturation = (0.8, 1.2)
        self._hue = (-0.1, 0.1)

        # KITTI 左 / 右 RGB camera index
        self.side_map = {"l": 2, "r": 3, "2": 2, "3": 3}

    # --------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.filenames)

    # --------------------------------------------------------------------
    # 路径拼装
    # --------------------------------------------------------------------
    def _image_path(self, folder: str, frame_index: int, side: str) -> str:
        """<data_path>/<date>/<drive>/image_0{2|3}/data/<10d>.png"""
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        return os.path.join(
            self.data_path, folder, "image_0{}".format(self.side_map[side]), "data", f_str
        )

    def _calib_dir(self, folder: str) -> str:
        """<data_path>/<date>/   (folder = '<date>/<drive>')"""
        return os.path.join(self.data_path, folder.split("/")[0])

    def _improved_gt_path(self, folder: str, frame_index: int, side: str) -> Optional[str]:
        """data_depth_annotated/{train,val}/<drive>/proj_depth/groundtruth/image_0{2|3}/<10d>.png
        注意:improved GT 目录里只有 drive 一级,没有 date 前缀。"""
        drive = folder.split("/")[1]
        f_str = "{:010d}.png".format(frame_index)
        for sub in ("train", "val"):
            p = os.path.join(
                self.anno_path, sub, drive,
                "proj_depth/groundtruth/image_0{}".format(self.side_map[side]),
                f_str,
            )
            if os.path.isfile(p):
                return p
        return None

    # --------------------------------------------------------------------
    # 单帧读取(原图分辨率)
    # --------------------------------------------------------------------
    def _load_color(self, folder: str, frame_index: int, side: str, do_flip: bool):
        img = _pil_loader(self._image_path(folder, frame_index, side))
        if do_flip:
            img = img.transpose(pil.FLIP_LEFT_RIGHT)
        return img

    def _load_depth_gt(self, folder: str, frame_index: int, side: str, do_flip: bool) -> Optional[np.ndarray]:
        """返回原图分辨率 (H_full, W_full) 的 float32 稀疏深度图,缺失为 0。"""
        if self.gt_source == "none":
            return None

        if self.gt_source == "velodyne":
            # [PIN-1] eigen split:用 velodyne 投影出的稀疏 GT
            velo_filename = os.path.join(
                self.data_path, folder,
                "velodyne_points/data/{:010d}.bin".format(int(frame_index)),
            )
            if not os.path.isfile(velo_filename):
                return None
            depth = ku.generate_depth_map(
                self._calib_dir(folder), velo_filename, cam=self.side_map[side]
            )
            # monodepth2 用 skimage.transform.resize(order=0);为避免依赖 skimage,
            # 这里改成纯 numpy 的最近邻 resize,效果等价(order=0 即 NN)。
            out_w, out_h = KITTI_FULL_RES_WH
            in_h, in_w = depth.shape
            if (in_h, in_w) != (out_h, out_w):
                y_idx = (np.arange(out_h) * in_h / out_h).astype(np.int64)
                x_idx = (np.arange(out_w) * in_w / out_w).astype(np.int64)
                depth = depth[y_idx[:, None], x_idx[None, :]]
            depth = depth.astype(np.float32)

        elif self.gt_source == "improved":
            # [PIN-1] eigen_benchmark split:用官方 data_depth_annotated improved GT
            p = self._improved_gt_path(folder, frame_index, side)
            if p is None:
                return None
            depth_png = pil.open(p)
            # 注意:improved GT 一般已经是 1216×352 之类,这里 resize 回 1242×375 保持口径一致
            depth_png = depth_png.resize(KITTI_FULL_RES_WH, pil.NEAREST)
            # KITTI 官方编码:uint16 / 256.0 = 米
            depth = np.array(depth_png).astype(np.float32) / 256.0

        if do_flip:
            depth = np.fliplr(depth).copy()
        return depth

    # --------------------------------------------------------------------
    # 内参缩放:把 *真值* 内参从原图 (W_full, H_full) 缩到 (self.width, self.height)
    # --------------------------------------------------------------------
    def _build_K(self, folder: str) -> Tuple[np.ndarray, np.ndarray]:
        K_full = ku.get_real_K(self._calib_dir(folder), cam=2).copy()  # (3,3) 原图像素
        sx = self.width / KITTI_FULL_RES_WH[0]
        sy = self.height / KITTI_FULL_RES_WH[1]
        K_full[0, :] *= sx  # fx, 0, cx 一起缩
        K_full[1, :] *= sy  # 0, fy, cy
        K4 = np.eye(4, dtype=np.float32)
        K4[:3, :3] = K_full
        inv_K4 = np.linalg.pinv(K4).astype(np.float32)
        return K4, inv_K4

    # --------------------------------------------------------------------
    # 主入口
    # --------------------------------------------------------------------
    def __getitem__(self, index: int) -> Dict:
        line = self.filenames[index].split()
        folder = line[0]                                # '<date>/<drive>'
        frame_index = int(line[1]) if len(line) >= 2 else 0
        side = line[2] if len(line) >= 3 else "l"

        # monodepth2 风格:训练时随机水平翻转 & 色彩增强(50% 概率)
        do_flip = self.is_train and random.random() > 0.5
        do_color_aug = self.is_train and self.color_jitter_enabled and random.random() > 0.5

        # ---- 1) 读邻帧 + 当前帧 ----------------------------------------
        pil_frames: Dict[int, object] = {}
        for fi in self.frame_idxs:
            try:
                pil_frames[fi] = self._load_color(folder, frame_index + fi, side, do_flip)
            except FileNotFoundError:
                # eval 模式下 ±1 邻帧可能不存在(序列首尾),复用当前帧避免崩
                pil_frames[fi] = self._load_color(folder, frame_index, side, do_flip)

        # ---- 2) resize 到目标分辨率 [PIN-2] ----------------------------
        resized = {fi: self.resize_color(im) for fi, im in pil_frames.items()}

        # ---- 3) color jitter:同一序列共用一个 jitter,保证邻帧光度一致 -
        if do_color_aug:
            color_aug = transforms.ColorJitter(
                self._brightness, self._contrast, self._saturation, self._hue
            )
        else:
            color_aug = (lambda x: x)

        # ---- 4) 转 tensor + 装字典 ------------------------------------
        inputs: Dict = {}
        for fi, im in resized.items():
            inputs[("color", fi, 0)] = self.to_tensor(im)              # [0,1]
            inputs[("color_aug", fi, 0)] = self.to_tensor(color_aug(im))

        # ---- 5) 内参(基于 self.height/self.width) --------------------
        K4, inv_K4 = self._build_K(folder)
        inputs[("K", 0)] = torch.from_numpy(K4)
        inputs[("inv_K", 0)] = torch.from_numpy(inv_K4)

        # ---- 6) eval 时附 GT ------------------------------------------
        if self.gt_source != "none":
            dgt = self._load_depth_gt(folder, frame_index, side, do_flip)
            if dgt is not None:
                inputs["depth_gt"] = torch.from_numpy(dgt[None, ...].astype(np.float32))

        inputs["frame_meta"] = "{}/{:010d}/{}".format(folder, frame_index, side)
        return inputs
