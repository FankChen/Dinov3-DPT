"""校验 NYU labeled subset 数据完整性,并准备一个干净的 root symlink。

DINOv3 的 dataset 类期望 root 直接指向"包含 train.txt + train/ + test/"的目录。
我们 cluster 上数据深埋在:
    /fs/scratch/datasets/cr_dlp_open_permissive/nyuv2/nyu_depth_v2/nyuv2/

为了 yaml 里 datasets.root 写起来短,我们建一个 symlink:
    my_baseline/data/nyu  →  /fs/scratch/.../nyu_depth_v2/nyuv2/

用法:
    python scripts/prepare_nyu_split.py
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]   # liren/dinov3_baseline/
TARGET = REPO_ROOT / "my_baseline" / "data" / "nyu"

NYU_REAL = Path(
    "/fs/scratch/datasets/cr_dlp_open_permissive/nyuv2/nyu_depth_v2/nyuv2"
)


def check_dir(p: Path, expected_n: int, label: str) -> None:
    if not p.is_dir():
        raise FileNotFoundError(f"[FAIL] {label} not a dir: {p}")
    n = len(list(p.iterdir()))
    if n < expected_n:
        raise RuntimeError(f"[FAIL] {label} has {n} files, expected ~{expected_n}: {p}")
    print(f"  [ok] {label:20s} N={n:>5d}  {p}")


def main() -> None:
    print(f"[check] real data root = {NYU_REAL}")
    if not NYU_REAL.is_dir():
        raise FileNotFoundError(f"NYU real root not found: {NYU_REAL}")

    # 校验关键文件 / 目录
    check_dir(NYU_REAL / "train" / "rgb", 795, "train/rgb")
    check_dir(NYU_REAL / "train" / "depth", 795, "train/depth")
    check_dir(NYU_REAL / "test" / "rgb", 654, "test/rgb")
    check_dir(NYU_REAL / "test" / "depth", 654, "test/depth")

    for split_name in ("train.txt", "test.txt"):
        p = NYU_REAL / split_name
        if not p.is_file():
            raise FileNotFoundError(f"split file missing: {p}")
        n = sum(1 for ln in p.read_text().splitlines() if ln.strip())
        print(f"  [ok] {split_name:20s} N={n:>5d}  {p}")

    # 建 symlink,让 yaml 里 root 路径短
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if TARGET.is_symlink() or TARGET.exists():
        existing = os.readlink(TARGET) if TARGET.is_symlink() else "(non-symlink)"
        if Path(existing).resolve() == NYU_REAL.resolve():
            print(f"\n[ok] symlink already correct: {TARGET} → {NYU_REAL}")
            return
        print(f"[overwrite] {TARGET} → {existing}, repointing")
        TARGET.unlink()
    os.symlink(NYU_REAL, TARGET)
    print(f"\n[ok] created symlink: {TARGET} → {NYU_REAL}")


if __name__ == "__main__":
    main()
