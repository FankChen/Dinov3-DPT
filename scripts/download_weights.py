"""下载 DINOv3 backbone 权重到本地,避免训练时计算节点拉不到。

用法(在 login 节点先跑):
    huggingface-cli login         # 一次性,接受 license
    python scripts/download_weights.py

会下载:
    facebook/dinov3-vitl16-pretrain-lvd1689m → checkpoints/dinov3_vitl16/
"""
from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parents[2]   # liren/dinov3_baseline/
CKPT_DIR = REPO_ROOT / "my_baseline" / "checkpoints"


WEIGHTS = [
    # (HuggingFace repo_id, 本地子目录名)
    ("facebook/dinov3-vitl16-pretrain-lvd1689m", "dinov3_vitl16"),
]


def _ensure_model_pth(local_dir: Path) -> Path | None:
    """下完之后,在 local_dir 里创建 / 更新一个稳定名字的 model.pth symlink,
    指向真正的权重文件。这样 yaml 里只用写 model.pth,不必每次手动改名。

    优先级: *.pth > *.safetensors > consolidated.* > pytorch_model.bin
    """
    candidates = (
        sorted(local_dir.glob("*.pth"))
        or sorted(local_dir.glob("*.safetensors"))
        or sorted(local_dir.glob("pytorch_model.bin"))
        or sorted(local_dir.glob("consolidated.*"))
    )
    if not candidates:
        print(f"[warn] no weight file found under {local_dir}")
        return None
    # 排除我们自己建的 symlink
    candidates = [c for c in candidates if c.name != "model.pth" or not c.is_symlink()]
    if not candidates:
        return None
    src = candidates[0]
    link = local_dir / "model.pth"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(src.name)  # 用相对路径,目录搬走也不会断
    print(f"[link] {link}  →  {src.name}")
    return link


def main():
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    for repo_id, local_name in WEIGHTS:
        local_dir = CKPT_DIR / local_name
        already_have = local_dir.exists() and (
            any(local_dir.glob("*.pth")) or any(local_dir.glob("*.safetensors"))
        )
        if already_have:
            print(f"[skip] {repo_id} already at {local_dir}")
        else:
            print(f"[download] {repo_id} → {local_dir}")
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
            )
            print(f"[ok] {local_dir}")
            for p in sorted(local_dir.iterdir()):
                if p.is_file():
                    print("  -", p.name, f"({p.stat().st_size / 1e6:.1f} MB)")

        # 不论是不是新下的,都确保 model.pth 这个稳定 symlink 存在
        _ensure_model_pth(local_dir)


if __name__ == "__main__":
    main()
