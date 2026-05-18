"""
KITTI 深度评测器 —— self-sup 训练侧 + zero-shot 复现侧共用。

协议来源(全部移植,不发明):
  - Eigen et al. 2014:           split + mask gt > 0
  - Garg et al. 2016:            center crop (用于 'eigen' raw GT split)
  - Monodepth2 evaluate_depth.py L160-218:  median scaling + 5 指标
  - MiDaS / DPT / DINOv3 paper:  least-squares affine alignment

─────────────────────────────────────────────────────────────────
★ 拍板项落地点
   [PIN-1] 双 split 都报:
              split='eigen'           → garg_crop + velodyne GT
              split='eigen_benchmark' → 无 crop,只 mask gt > 0 (improved GT 已稠密)
   [PIN-3] 双 scale align:
              align='median'        → s = median(gt)/median(pred), 只调 scale
                                       (self-sup mono 训练评估口径)
              align='least_squares' → 最小二乘解 (s, t):min ||s·p + t - g||²
                                       (MiDaS / DPT / DINOv3 zero-shot 复现口径)
              align='none'          → 不动 pred(stereo / metric-supervised 才用)
─────────────────────────────────────────────────────────────────

API:
    ev = KITTIEvaluator(split='eigen', align='median')
    for pred, gt in val_loader:           # pred/gt 都是 numpy 或 torch tensor
        ev.update(pred, gt)
    metrics = ev.compute()                # dict[str, float],对所有样本平均
    ev.print_table()                      # 打印漂亮表格
    ev.save_json(out_path)                # 保存详细结果(per-image + 平均)
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False

from .metrics import compute_depth_errors, DEPTH_METRIC_NAMES


# KITTI 标准深度区间(所有论文都用这个)
MIN_DEPTH = 1e-3
MAX_DEPTH = 80.0

# Garg 2016 crop(相对坐标),只对 'eigen' raw split 使用
GARG_CROP = (0.40810811, 0.99189189, 0.03594771, 0.96405229)  # (y0, y1, x0, x1)


# ============================================================================
# 工具:tensor → numpy
# ============================================================================
def _to_numpy(x):
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _resize_pred_to_gt(pred: np.ndarray, gt_shape: Tuple[int, int]) -> np.ndarray:
    """把 pred resize 到 GT 分辨率。pred 是稠密深度,用 bilinear 安全。
    不依赖 cv2 / skimage —— 用 PIL 即可。"""
    H, W = gt_shape
    if pred.shape == (H, W):
        return pred
    from PIL import Image
    # PIL.Image.resize 要 (W, H) 顺序
    img = Image.fromarray(pred.astype(np.float32), mode="F")
    img = img.resize((W, H), resample=Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)


# ============================================================================
# Scale alignment(PIN-3 的两个口径都在这)
# ============================================================================
def align_median(pred: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, float]:
    """monodepth2 风格:s = median(gt) / median(pred); pred *= s。
    只调 scale,不调 shift。
    Returns:
        aligned_pred, scale_ratio
    """
    s = float(np.median(gt) / np.median(pred))
    return pred * s, s


def align_least_squares(pred: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
    """affine-invariant alignment:解 min_{s,t} ||s·pred + t - gt||²。
    闭式解 = 一元线性回归。
    Returns:
        aligned_pred, (s, t)
    """
    p = pred.astype(np.float64)
    g = gt.astype(np.float64)
    p_mean, g_mean = p.mean(), g.mean()
    var_p = ((p - p_mean) ** 2).mean()
    if var_p < 1e-12:
        # pred 全是同一个值 → 无法解 affine,退化到 median
        return align_median(pred, gt)
    cov = ((p - p_mean) * (g - g_mean)).mean()
    s = cov / var_p
    t = g_mean - s * p_mean
    return (s * pred + t).astype(np.float32), (float(s), float(t))


# ============================================================================
# 主评测器
# ============================================================================
class KITTIEvaluator:
    """
    用法:
        ev = KITTIEvaluator(split='eigen', align='median')
        for pred_b, gt_b in loader:        # pred_b/gt_b: (B, 1, H, W) 或 (B, H, W)
            ev.update(pred_b, gt_b)
        ev.compute()  # → dict 平均指标
    """

    def __init__(
        self,
        split: str = "eigen",                           # [PIN-1]
        align: str = "median",                          # [PIN-3]
        min_depth: float = MIN_DEPTH,
        max_depth: float = MAX_DEPTH,
        garg_crop: Optional[bool] = None,               # None = 由 split 自动决定
        ckpt_name: str = "unknown",
    ):
        assert split in ("eigen", "eigen_benchmark"), f"bad split: {split}"
        assert align in ("median", "least_squares", "none"), f"bad align: {align}"
        self.split = split
        self.align = align
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.ckpt_name = ckpt_name

        # [PIN-1] 默认协议:eigen 用 garg_crop,eigen_benchmark 不用
        if garg_crop is None:
            self.garg_crop = (split == "eigen")
        else:
            self.garg_crop = garg_crop

        self._per_image: List[Dict[str, float]] = []
        self._scale_ratios: List[float] = []

    # ------------------------------------------------------------------
    def reset(self):
        self._per_image.clear()
        self._scale_ratios.clear()

    # ------------------------------------------------------------------
    def update(
        self,
        pred: Union["np.ndarray", "torch.Tensor"],
        gt: Union["np.ndarray", "torch.Tensor"],
    ):
        """累积一个 batch。pred 和 gt 都可以是 (H,W) / (1,H,W) / (B,1,H,W) / (B,H,W)。"""
        pred_np = _to_numpy(pred).astype(np.float32)
        gt_np = _to_numpy(gt).astype(np.float32)

        # 统一成 (B, H, W)
        if pred_np.ndim == 2:
            pred_np = pred_np[None]
        elif pred_np.ndim == 4:
            pred_np = pred_np[:, 0]
        if gt_np.ndim == 2:
            gt_np = gt_np[None]
        elif gt_np.ndim == 4:
            gt_np = gt_np[:, 0]

        assert pred_np.shape[0] == gt_np.shape[0], "batch size mismatch"

        for i in range(pred_np.shape[0]):
            self._update_one(pred_np[i], gt_np[i])

    # ------------------------------------------------------------------
    def _update_one(self, pred: np.ndarray, gt: np.ndarray):
        """单张图的完整 6 步流程。"""
        gt_h, gt_w = gt.shape

        # ---- step 1: pred resize 到 GT 分辨率(bilinear) -------------
        pred = _resize_pred_to_gt(pred, (gt_h, gt_w))

        # ---- step 2: 数值合法性 ---------------------------------------
        # pred 必须 > 0(后面要除 / log)。<=0 的位置后面会被 clip;
        # 但 align 之前要保证 median 有意义,这里先 clip 一下下限
        pred = np.clip(pred, self.min_depth, None)

        # ---- step 3: 构造有效像素 mask --------------------------------
        # 基础 mask: GT 在合理范围
        mask = (gt > self.min_depth) & (gt < self.max_depth)

        # [PIN-1] eigen split:再叠加 Garg crop
        if self.garg_crop:
            crop = (
                int(GARG_CROP[0] * gt_h), int(GARG_CROP[1] * gt_h),
                int(GARG_CROP[2] * gt_w), int(GARG_CROP[3] * gt_w),
            )
            crop_mask = np.zeros_like(mask)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = True
            mask &= crop_mask

        if not mask.any():
            # 极端情况:该图根本没有有效像素(garg_crop 之外都没标注)
            # 跳过,不计入统计
            return

        pred_v = pred[mask]
        gt_v = gt[mask]

        # ---- step 4: scale align [PIN-3] -----------------------------
        if self.align == "median":
            pred_v, ratio = align_median(pred_v, gt_v)
            self._scale_ratios.append(ratio)
        elif self.align == "least_squares":
            pred_v, (s, t) = align_least_squares(pred_v, gt_v)
            self._scale_ratios.append(s)  # 只记录 scale,shift 可在 json 里加
        # align == "none": 不动 pred

        # ---- step 5: clip 到合法范围 ---------------------------------
        pred_v = np.clip(pred_v, self.min_depth, self.max_depth)

        # ---- step 6: 算 7 个指标 -------------------------------------
        errs = compute_depth_errors(gt_v, pred_v)
        self._per_image.append(errs)

    # ------------------------------------------------------------------
    def compute(self) -> Dict[str, float]:
        """对所有样本取平均。"""
        if not self._per_image:
            raise RuntimeError("KITTIEvaluator.compute() called with no samples")
        agg = {name: float(np.mean([e[name] for e in self._per_image]))
               for name in DEPTH_METRIC_NAMES}
        agg["num_samples"] = len(self._per_image)
        if self._scale_ratios:
            agg["scale_ratio_median"] = float(np.median(self._scale_ratios))
            agg["scale_ratio_std"] = float(np.std(self._scale_ratios))
        return agg

    # ------------------------------------------------------------------
    def print_table(self, prefix: str = ""):
        """漂亮排版,贴汇报/paper 用。"""
        m = self.compute()
        head = "  " + " | ".join(f"{n:>9s}" for n in DEPTH_METRIC_NAMES)
        row = "  " + " | ".join(f"{m[n]:9.4f}" for n in DEPTH_METRIC_NAMES)
        bar = "=" * len(head)
        print(bar)
        print(f"{prefix}KITTI eval  split={self.split}  align={self.align}  "
              f"crop={'garg' if self.garg_crop else 'none'}  N={m['num_samples']}  "
              f"ckpt={self.ckpt_name}")
        print(bar)
        print(head)
        print(row)
        if "scale_ratio_median" in m:
            print(f"  scale ratio: median={m['scale_ratio_median']:.4f}  "
                  f"std={m['scale_ratio_std']:.4f}")
        print(bar)

    # ------------------------------------------------------------------
    def save_json(self, out_path: str):
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        payload = {
            "split": self.split,
            "align": self.align,
            "garg_crop": self.garg_crop,
            "min_depth": self.min_depth,
            "max_depth": self.max_depth,
            "ckpt": self.ckpt_name,
            "metrics": self.compute(),
            "per_image": self._per_image,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"-> eval results saved to {out_path}")
