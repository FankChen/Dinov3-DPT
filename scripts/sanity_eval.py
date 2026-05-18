"""
KITTIEvaluator 的合成 sanity —— 不依赖任何 7B 权重。

3 个测试:
  ① ideal pred = gt        → 所有指标完美(abs_rel≈0, a1≈1.0)
  ② pred = constant 10m   → 程序不崩,abs_rel 很大但有限,a1 很小
  ③ pred = gt * 2.0       → median align 应能完全救回(abs_rel≈0)
                            → least_squares align 也应能完全救回
                            → align=none 应该烂(因为差 2 倍)

跑法:
    cd /home/izi2sgh/MYDATA/quanjie/liren/dinov3_baseline
    /home/izi2sgh/MYDATA/quanjie/liren/envs/dinov3_baseline/bin/python \
        -m my_baseline.scripts.sanity_eval
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, "/home/izi2sgh/MYDATA/quanjie/liren/dinov3_baseline")

from my_baseline.datasets.kitti_raw import KITTIRawDataset, DEFAULT_KITTI_ROOT  # noqa: E402
from my_baseline.eval.kitti_eval import KITTIEvaluator  # noqa: E402

SPLITS_DIR = "/home/izi2sgh/MYDATA/quanjie/liren/dinov3_baseline/my_baseline/datasets/splits"


def banner(s: str):
    print("\n" + "=" * 70)
    print(s)
    print("=" * 70)


def load_real_gt_samples(split: str, n: int = 20):
    """从真实 dataloader 抓 n 个 GT depth 出来当合成测试的素材。"""
    gt_source = "velodyne" if split == "eigen" else "improved"
    split_file = os.path.join(SPLITS_DIR, split, "test_files.txt")
    ds = KITTIRawDataset(
        split_file=split_file,
        height=192, width=640,
        frame_idxs=(0,),
        is_train=False,
        gt_source=gt_source,
    )
    gts = []
    i = 0
    while len(gts) < n and i < len(ds):
        sample = ds[i]
        if "depth_gt" in sample:
            gts.append(sample["depth_gt"][0].numpy())  # (H_full, W_full)
        i += 1
    return gts


def run_one(split: str, align: str, fake_kind: str, gts):
    """fake_kind: 'gt' | 'const10' | 'gt_times_2' | 'gt_plus_noise'"""
    ev = KITTIEvaluator(split=split, align=align, ckpt_name=f"fake/{fake_kind}")
    for gt in gts:
        if fake_kind == "gt":
            pred = gt.copy()
            # 把 0 位置改成一个合理值,免得后面 log 出问题
            pred[pred <= 0] = 10.0
        elif fake_kind == "const10":
            pred = np.full_like(gt, 10.0)
        elif fake_kind == "gt_times_2":
            pred = gt * 2.0
            pred[pred <= 0] = 10.0
        elif fake_kind == "gt_plus_noise":
            np.random.seed(0)
            pred = gt + np.random.randn(*gt.shape).astype(np.float32) * 0.5
            pred = np.clip(pred, 1e-3, None)
        else:
            raise ValueError(fake_kind)
        ev.update(pred, gt)
    ev.print_table(prefix=f"  [fake={fake_kind}]  ")
    return ev.compute()


def main():
    banner("准备:从真实 dataloader 抓 20 张 GT 当素材")
    gts_eig = load_real_gt_samples("eigen", n=20)
    gts_eb = load_real_gt_samples("eigen_benchmark", n=20)
    print(f"  eigen           : {len(gts_eig)} GT 张, shape={gts_eig[0].shape}")
    print(f"  eigen_benchmark : {len(gts_eb)} GT 张, shape={gts_eb[0].shape}")

    # ----------------------------------------------------------------
    banner("① pred = GT —— 所有指标应完美 (abs_rel≈0, a1≈1.0)")
    for split, gts in [("eigen", gts_eig), ("eigen_benchmark", gts_eb)]:
        m = run_one(split, "median", "gt", gts)
        ok = m["abs_rel"] < 1e-4 and m["a1"] > 0.999
        print(f"  ✅ pass" if ok else f"  ❌ FAIL  abs_rel={m['abs_rel']:.6f}  a1={m['a1']:.6f}")

    # ----------------------------------------------------------------
    banner("② pred = const 10m —— 程序应不崩,abs_rel 大但有限")
    for split, gts in [("eigen", gts_eig)]:
        m = run_one(split, "median", "const10", gts)
        ok = np.isfinite(m["abs_rel"]) and m["a1"] < 0.5
        print(f"  ✅ pass (abs_rel={m['abs_rel']:.3f}, a1={m['a1']:.3f})" if ok
              else f"  ❌ FAIL")

    # ----------------------------------------------------------------
    banner("③ pred = GT × 2 —— median 应救回,none 应烂")
    print("\n  -- align=median (应救回到完美) --")
    m1 = run_one("eigen", "median", "gt_times_2", gts_eig)
    ok1 = m1["abs_rel"] < 1e-3
    print(f"  {'✅ pass' if ok1 else '❌ FAIL'}  (abs_rel={m1['abs_rel']:.6f})")

    print("\n  -- align=least_squares (应救回到完美) --")
    m2 = run_one("eigen", "least_squares", "gt_times_2", gts_eig)
    ok2 = m2["abs_rel"] < 1e-3
    print(f"  {'✅ pass' if ok2 else '❌ FAIL'}  (abs_rel={m2['abs_rel']:.6f})")

    print("\n  -- align=none (不调,应该烂:abs_rel≈1.0) --")
    m3 = run_one("eigen", "none", "gt_times_2", gts_eig)
    ok3 = m3["abs_rel"] > 0.5
    print(f"  {'✅ pass' if ok3 else '❌ FAIL'}  (abs_rel={m3['abs_rel']:.6f})")

    # ----------------------------------------------------------------
    banner("④ pred = GT + 高斯噪声 σ=0.5m —— 应有不错的数字")
    for split, gts in [("eigen", gts_eig), ("eigen_benchmark", gts_eb)]:
        m = run_one(split, "median", "gt_plus_noise", gts)
        ok = m["abs_rel"] < 0.1 and m["a1"] > 0.85
        print(f"  {'✅ pass' if ok else '❌ FAIL'}  "
              f"(abs_rel={m['abs_rel']:.4f}, a1={m['a1']:.4f})")

    banner("✅ Evaluator sanity finished")
    print("\nNOTE: 真正的 7B SYNTHMIX 数字应该在:")
    print("  KITTI eigen        AbsRel ≈ 0.07-0.08, a1 ≈ 0.95+  (least_squares 口径)")
    print("  KITTI eigen_bench  AbsRel ≈ 0.05-0.07, a1 ≈ 0.96+  (least_squares 口径)")


if __name__ == "__main__":
    main()
