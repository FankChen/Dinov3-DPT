# Copyright (c) 2026
# 自定义 NYU dataset:适配 cluster 上现有的 NYU labeled subset(795 训 + 654 测)
#
# ============================================================
# 为什么有这个文件?(选择 Q1=a 的备注)
# ------------------------------------------------------------
# 官方 dinov3.data.datasets.NYU 期望 BTS 格式:
#     - split 文件每行: "<rgb_path> <depth_path> <focal>"
#     - 数据按 scene 组织: NYU/<scene>/<rgb>.jpg + sync_depth_*.png
#
# 但我们 cluster 上 ( /fs/scratch/.../nyu_depth_v2/nyuv2/ ) 只有
# labeled subset 格式:
#     - split 文件每行: 只有一个 ID,如 "0003"
#     - 文件: train/rgb/<id>.png + train/depth/<id>.png  (16-bit, 单位 mm)
#
# 两种适配方案:
#   方案 a: 写新 dataset 类,不改 split 文件                ← 我们用这个
#   方案 b: 预生成伪 BTS split,直接复用官方 NYU 类         ← 没用
#
# 选 a 是因为:数据格式扁平,新 dataset 类只需 ~50 行,代码清晰。
# ============================================================
#
# 输出格式与官方 NYU 一致:每个 sample 是 (image_bytes, depth_bytes),
# 由 ExtendedVisionDataset 框架的 image_decoder/target_decoder 解码成 (PIL.Image, PIL.Image)。
# 之后官方 transforms.py 会把 depth 当 16-bit Tensor 处理,除以 normalization_constant=1000
# 得到米单位深度。

from __future__ import annotations

import os
from enum import Enum
from typing import Any, Callable, Optional, Union

from PIL import Image

# 复用官方的基类和 decoder,避免重复造轮子
from dinov3.data.datasets.extended import ExtendedVisionDataset
from dinov3.data.datasets.decoders import (
    Decoder,
    DenseTargetDecoder,
    ImageDataDecoder,
)


class _Split(Enum):
    """与官方 NYU 一样的三档 split,但底层文件用我们的格式。"""
    TRAIN = "train"
    VAL = "val"   # NYU 没有独立 val,我们用 test 当 val(与官方一致)
    TEST = "test"

    @property
    def split_filename(self) -> str:
        # cluster 上的 split 文件:nyuv2/train.txt, nyuv2/test.txt
        _MAP = {
            _Split.TRAIN: "train.txt",
            _Split.VAL: "test.txt",
            _Split.TEST: "test.txt",
        }
        return _MAP[self]

    @property
    def subdir(self) -> str:
        # 文件实际存放的子目录
        _MAP = {
            _Split.TRAIN: "train",
            _Split.VAL: "test",
            _Split.TEST: "test",
        }
        return _MAP[self]


class NYULabeled(ExtendedVisionDataset):
    """NYU labeled subset (795 train + 654 test).

    目录结构(root 应为 ".../nyu_depth_v2/nyuv2/"):
        root/
          train.txt            一行一个 ID
          test.txt             一行一个 ID
          train/
            rgb/<id>.png       (480, 640, 3) uint8
            depth/<id>.png     (480, 640)    uint16 mm  ← inpainted
            depth_raw/<id>.png (480, 640)    uint16 mm, 0=invalid
          test/
            (同上)

    Args:
        split:        TRAIN / VAL / TEST
        root:         指向 nyuv2/ 目录
        use_raw_depth: True 用 depth_raw/(更接近 BTS),False 用 inpainted depth/
        其余参数与 ExtendedVisionDataset 相同
    """

    Split = Union[_Split]
    Labels = Union[Image.Image]

    def __init__(
        self,
        *,
        split: "NYULabeled.Split",
        root: Optional[str] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_decoder: Decoder = ImageDataDecoder,
        target_decoder: Decoder = DenseTargetDecoder,
        use_raw_depth: bool = False,
    ) -> None:
        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            image_decoder=image_decoder,
            target_decoder=target_decoder,
        )
        assert root is not None, "must provide root"
        self.split = split
        self.use_raw_depth = use_raw_depth
        depth_dir = "depth_raw" if use_raw_depth else "depth"

        split_path = os.path.join(root, split.split_filename)
        with open(split_path) as f:
            ids = [ln.strip() for ln in f if ln.strip()]

        self.image_paths = [
            os.path.join(split.subdir, "rgb", f"{idn}.png") for idn in ids
        ]
        self.target_paths = [
            os.path.join(split.subdir, depth_dir, f"{idn}.png") for idn in ids
        ]

    # ------------------------------------------------------------------
    # 下面 3 个方法签名必须与官方 NYU 类完全一致(被 ExtendedVisionDataset
    # 的 __getitem__ 调用)。
    # ------------------------------------------------------------------
    def get_image_data(self, index: int) -> bytes:
        full = os.path.join(self.root, self.image_paths[index])
        with open(full, "rb") as f:
            return f.read()

    def get_target(self, index: int) -> Any:
        full = os.path.join(self.root, self.target_paths[index])
        with open(full, "rb") as f:
            return f.read()

    def __len__(self) -> int:
        return len(self.image_paths)
