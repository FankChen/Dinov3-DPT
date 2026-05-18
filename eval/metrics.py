"""
KITTI 标准深度指标 —— 6 个纯函数 + 1 个统一入口。

完全对齐 monodepth2/evaluate_depth.py 的 compute_errors() (L27-44),
口径与 Eigen 2014 / Garg 2016 / MiDaS / DPT 系列论文一致。

7 个指标:
  - abs_rel     mean(|p-g| / g)
  - sq_rel      mean((p-g)² / g)
  - rmse        sqrt(mean((p-g)²))
  - rmse_log    sqrt(mean((log p - log g)²))
  - a1          % of pixels where max(p/g, g/p) < 1.25
  - a2          ...                                 < 1.25²
  - a3          ...                                 < 1.25³

输入约定:
  - gt, pred 都是 1-D numpy array,*已经经过 mask + crop*,只剩有效像素
  - 单位:米
  - 调用前 caller 保证两者 shape 相同且 > 0
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


DEPTH_METRIC_NAMES: Tuple[str, ...] = (
    "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"
)


def compute_depth_errors(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """7 个标准指标。完全对齐 monodepth2 compute_errors。

    Args:
        gt:   (N,) float, > 0,单位米
        pred: (N,) float, > 0,单位米(已 clip 到 [MIN_DEPTH, MAX_DEPTH])
    Returns:
        dict[str, float]
    """
    assert gt.shape == pred.shape, f"shape mismatch: gt={gt.shape} pred={pred.shape}"
    assert gt.size > 0, "empty valid mask — check your crop / mask logic"

    thresh = np.maximum(gt / pred, pred / gt)
    a1 = float((thresh < 1.25     ).mean())
    a2 = float((thresh < 1.25 ** 2).mean())
    a3 = float((thresh < 1.25 ** 3).mean())

    rmse = float(np.sqrt(((gt - pred) ** 2).mean()))
    rmse_log = float(np.sqrt(((np.log(gt) - np.log(pred)) ** 2).mean()))

    abs_rel = float(np.mean(np.abs(gt - pred) / gt))
    sq_rel = float(np.mean(((gt - pred) ** 2) / gt))

    return {
        "abs_rel":  abs_rel,
        "sq_rel":   sq_rel,
        "rmse":     rmse,
        "rmse_log": rmse_log,
        "a1":       a1,
        "a2":       a2,
        "a3":       a3,
    }
