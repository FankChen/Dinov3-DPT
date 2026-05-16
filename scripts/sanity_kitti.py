"""
KITTI dataloader sanity check —— 独立可跑,打印一切供人工核对。

跑法:
    cd /home/izi2sgh/MYDATA/quanjie/liren/dinov3_baseline
    python -m my_baseline.scripts.sanity_kitti

输出会覆盖以下需要确认的事项:
  ① 三个 split 文件长度对不对(39810 / 697 / 652)
  ② train batch 邻帧 (-1, 0, +1) shape 一致
  ③ K(3,3) 数值合理(fx 约等于 width × 0.58 量级)
  ④ eigen GT (velodyne)   非零点 占比 ≈ 1-5%
  ⑤ eigen_benchmark GT    非零点 占比 ≈ 15-25%
  ⑥ 训练分辨率切到 768×1024 也能跑通(stage-2 烟雾测试)
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

# 确保以 -m 跑时能找到 my_baseline
sys.path.insert(0, "/home/izi2sgh/MYDATA/quanjie/liren/dinov3_baseline")

from my_baseline.datasets.kitti_raw import KITTIRawDataset  # noqa: E402

SPLITS_DIR = "/home/izi2sgh/MYDATA/quanjie/liren/dinov3_baseline/my_baseline/datasets/splits"


def banner(s: str):
    print("\n" + "=" * 70)
    print(s)
    print("=" * 70)


def describe_batch(batch, tag: str):
    print(f"\n--- {tag} ---")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {str(k):28s}  shape={tuple(v.shape)}  dtype={v.dtype}  "
                  f"min={v.float().min().item():.4f}  max={v.float().max().item():.4f}")
        else:
            print(f"  {str(k):28s}  {v}")


def check_K(K4: torch.Tensor, height: int, width: int):
    K = K4[0].numpy()  # 取 batch 第 0 个,仅打印
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    print(f"  K[0,0]=fx={fx:.2f}  K[1,1]=fy={fy:.2f}  cx={cx:.2f}  cy={cy:.2f}")
    print(f"  expected: cx≈{width/2:.1f}, cy≈{height/2:.1f}, fx 在 {0.5*width:.0f}~{0.9*width:.0f} 之间合理")


def describe_depth(d: torch.Tensor, tag: str):
    arr = d[0, 0].numpy()  # (H, W)
    valid = arr > 0
    pct = 100.0 * valid.sum() / arr.size
    if valid.any():
        vmin, vmax, vmean = arr[valid].min(), arr[valid].max(), arr[valid].mean()
    else:
        vmin = vmax = vmean = 0.0
    print(f"  {tag}: shape={arr.shape}  valid_px={pct:.2f}%  "
          f"depth(valid) min={vmin:.2f}m  max={vmax:.2f}m  mean={vmean:.2f}m")


# ============================================================================
def main():
    # ---- ① split 行数核对 ----
    banner("① split 文件行数(必须 = 39810 / 4424 / 697 / 652)")
    for rel, expect in [
        ("eigen_zhou/train_files.txt", 39810),
        ("eigen_zhou/val_files.txt", 4424),
        ("eigen/test_files.txt", 697),
        ("eigen_benchmark/test_files.txt", 652),
    ]:
        p = os.path.join(SPLITS_DIR, rel)
        n = sum(1 for _ in open(p))
        flag = "✅" if n == expect else "❌"
        print(f"  {flag}  {rel:36s}  {n}  (expect {expect})")

    # ---- ② train stage-1 (192×640) ----
    banner("② train (eigen_zhou, 192×640, frame_idxs=(0,-1,1), color jitter) [PIN-2 stage-1]")
    ds_train = KITTIRawDataset(
        split_file=os.path.join(SPLITS_DIR, "eigen_zhou/train_files.txt"),
        height=192, width=640,
        frame_idxs=(0, -1, 1),
        is_train=True,
        gt_source="none",
    )
    print(f"  len={len(ds_train)}")
    loader = DataLoader(ds_train, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    describe_batch(batch, "train batch[2]")
    check_K(batch[("K", 0)], 192, 640)

    # ---- ③ eval eigen (velodyne GT) ----
    banner("③ eval eigen (697, velodyne raw GT, single frame, 192×640) [PIN-1 raw]")
    ds_eig = KITTIRawDataset(
        split_file=os.path.join(SPLITS_DIR, "eigen/test_files.txt"),
        height=192, width=640,
        frame_idxs=(0,),
        is_train=False,
        gt_source="velodyne",
    )
    print(f"  len={len(ds_eig)}")
    sample = ds_eig[0]
    describe_batch({k: v for k, v in sample.items()}, "eigen[0]")
    if "depth_gt" in sample:
        describe_depth(sample["depth_gt"][None], "depth_gt (velodyne)")

    # ---- ④ eval eigen_benchmark (improved GT) ----
    banner("④ eval eigen_benchmark (652, improved GT, 192×640) [PIN-1 improved]")
    ds_eb = KITTIRawDataset(
        split_file=os.path.join(SPLITS_DIR, "eigen_benchmark/test_files.txt"),
        height=192, width=640,
        frame_idxs=(0,),
        is_train=False,
        gt_source="improved",
    )
    print(f"  len={len(ds_eb)}")
    # eigen_benchmark 的前几行可能 frame_index 找不到 improved GT(并非每帧都标注),
    # 所以我们顺序找到第一个带 GT 的样本
    hit = None
    for i in range(min(50, len(ds_eb))):
        s = ds_eb[i]
        if "depth_gt" in s:
            hit = (i, s)
            break
    if hit is None:
        print("  ❌  前 50 个样本都没有找到 improved GT —— 检查 anno_path")
    else:
        i, s = hit
        print(f"  第一个带 GT 的样本 index={i}")
        describe_batch(s, "eigen_benchmark[hit]")
        describe_depth(s["depth_gt"][None], "depth_gt (improved)")

    # ---- ⑤ stage-2 烟雾测试 (768×1024) ----
    banner("⑤ train stage-2 烟雾测试 (768×1024) [PIN-2 stage-2]")
    ds_hi = KITTIRawDataset(
        split_file=os.path.join(SPLITS_DIR, "eigen_zhou/train_files.txt"),
        height=768, width=1024,
        frame_idxs=(0, -1, 1),
        is_train=True,
        gt_source="none",
    )
    loader_hi = DataLoader(ds_hi, batch_size=1, shuffle=False, num_workers=0)
    batch_hi = next(iter(loader_hi))
    describe_batch(batch_hi, "train batch[1] 768×1024")
    check_K(batch_hi[("K", 0)], 768, 1024)

    banner("✅ sanity check finished —— 上面 ①~⑤ 都正常即代表 dataloader OK")
    print("\nNOTE for PIN-3 (scale align):")
    print("  本 dataloader 不做 scale alignment;evaluator 阶段决定:")
    print("    - self-sup 训练评估     → median scaling")
    print("    - SYNTHMIX zero-shot 复现 → least-squares (scale + shift, scale-invariant)")


if __name__ == "__main__":
    main()
