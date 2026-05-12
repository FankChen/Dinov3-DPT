"""我们的训练 launcher。在调官方 run.py 之前,monkey-patch 一件事:

把我们的 NYULabeled 注册进 dinov3.data.loaders 的 dataset registry,
让 yaml 里 'NYULabeled:split=TRAIN' 这个 dataset_str 能被解析。

为什么这样做(而不是直接改源码):
    所有改动都是增量的,DINOv3 源码完全不动,后续容易 PR 或同步。

backbone 权重不走 torch.hub(计算节点不联网),改走 config_file + pretrained_weights
路径,直接在 yaml 的 model: 段里指定本地路径,见 configs/nyu_dpt_vitl.yaml。

用法:
    python scripts/run_train.py config=my_baseline/configs/nyu_dpt_vitl.yaml \
        output_dir=my_baseline/outputs/nyu_dpt_vitl_v0
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# 路径配置
# 目录结构:
#   liren/dinov3_baseline/                <- REPO_ROOT
#     dinov3/                              <- DINOV3_PKG_PARENT (含 setup.py)
#       dinov3/                            <- 真正的 python 包
#     my_baseline/                         <- 我们的代码
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]                          # liren/dinov3_baseline/
DINOV3_PKG_PARENT = REPO_ROOT / "dinov3"             # 含 setup.py 的目录

# 让 import dinov3 / my_baseline 都能成功
sys.path.insert(0, str(DINOV3_PKG_PARENT))
sys.path.insert(0, str(REPO_ROOT))


# =============================================================================
# (1) 注册 NYULabeled 到 dataset registry
# =============================================================================
def _patch_dataset_registry() -> None:
    import dinov3.data.loaders as _loaders
    from my_baseline.datasets.nyu_labeled import NYULabeled

    _orig = _loaders._parse_dataset_str

    def _patched(dataset_str: str):
        # 先看看是不是我们的 dataset
        tokens = dataset_str.split(":")
        name = tokens[0]
        if name != "NYULabeled":
            return _orig(dataset_str)

        # 我们的 dataset:复用官方解析模式
        kwargs = {}
        for tok in tokens[1:]:
            key, value = tok.split("=", 1)
            assert key in ("root", "extra", "split", "use_raw_depth"), \
                f"unknown NYULabeled kwarg: {key}"
            kwargs[key] = value
        if "split" in kwargs:
            kwargs["split"] = NYULabeled.Split[kwargs["split"]]
        if "use_raw_depth" in kwargs:
            kwargs["use_raw_depth"] = kwargs["use_raw_depth"].lower() == "true"
        return NYULabeled, kwargs

    _loaders._parse_dataset_str = _patched
    print("[patch] registered NYULabeled in dataset registry")


# =============================================================================
# (2) 已删除:backbone 权重不再走 torch.hub,直接用 model.config_file + 
#     model.pretrained_weights 在本地加载,见 configs/nyu_dpt_vitl.yaml。
# =============================================================================


def main() -> None:
    _patch_dataset_registry()

    # 调用官方入口
    from dinov3.eval.depth.run import main as official_main
    official_main(sys.argv[1:])


if __name__ == "__main__":
    main()
