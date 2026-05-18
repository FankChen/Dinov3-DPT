"""Smoke test: HFDinov3Backbone + dinov3 DPTHead forward + backward.

Run:
    PYTHONPATH=/home/izi2sgh/MYDATA/quanjie/liren/dinov3_baseline/dinov3:\
/home/izi2sgh/MYDATA/quanjie/liren/dinov3_baseline \
        python my_baseline/scripts/sanity_dpt_forward.py
"""
import os
import sys

# Make both `dinov3` (upstream) and `my_baseline` importable
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))  # .../dinov3_baseline
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "dinov3"))

import torch  # noqa: E402

from my_baseline.models.backbone_hf import (  # noqa: E402
    DINOV3_VITL_FOUR_INTERVALS,
    HFDinov3Backbone,
)
from dinov3.eval.depth.models.dpt_head import DPTHead  # noqa: E402


def main():
    force_cpu = os.environ.get("FORCE_CPU", "0") == "1"
    if force_cpu:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # On LSF login nodes torch.cuda.is_available() may return True but
        # any allocation throws CUDA_ERROR_DEVICES_UNAVAILABLE. Probe once.
        if device == "cuda":
            try:
                _ = torch.empty(1, device="cuda")
            except Exception as e:
                print(f"[sanity] CUDA probe failed ({e.__class__.__name__}); falling back to CPU")
                device = "cpu"
    print(f"[sanity] device={device}")

    print("[sanity] loading backbone ...")
    backbone = HFDinov3Backbone(
        hf_model_id="facebook/dinov3-vitl16-pretrain-lvd1689m",
        freeze=True,
        dtype=torch.float32,
    ).to(device)
    print(
        f"[sanity] backbone embed_dim={backbone.embed_dim} "
        f"n_blocks={backbone.n_blocks} patch={backbone.patch_size} "
        f"reg={backbone.num_register_tokens}"
    )

    print("[sanity] building DPT head ...")
    head = DPTHead(
        in_channels=(backbone.embed_dim,) * 4,
        channels=256,
        post_process_channels=[128, 256, 512, 1024],
        readout_type="project",
        n_output_channels=1,  # depth = single channel
        n_hidden_channels=32,
    ).to(device)
    head.train()
    n_head_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    n_backbone_trainable = sum(
        p.numel() for p in backbone.parameters() if p.requires_grad
    )
    print(
        f"[sanity] head trainable params={n_head_params/1e6:.2f}M, "
        f"backbone trainable params={n_backbone_trainable}"
    )
    assert n_backbone_trainable == 0, "backbone should be frozen"

    # KITTI-like crop: 192 x 640, batch=2
    B, H, W = 2, 192, 640
    x = torch.randn(B, 3, H, W, device=device)
    print(f"[sanity] input shape={tuple(x.shape)}")

    print("[sanity] extracting intermediate layers ...")
    feats = backbone.get_intermediate_layers(
        x,
        n=DINOV3_VITL_FOUR_INTERVALS,
        reshape=True,
        return_class_token=True,
        norm=True,
    )
    assert len(feats) == 4, f"expected 4 stages got {len(feats)}"
    for i, (pf, cls) in enumerate(feats):
        print(f"  stage {i} block={DINOV3_VITL_FOUR_INTERVALS[i]}  patch={tuple(pf.shape)}  cls={tuple(cls.shape)}")
        expected_h, expected_w = H // backbone.patch_size, W // backbone.patch_size
        assert pf.shape == (B, backbone.embed_dim, expected_h, expected_w)
        assert cls.shape == (B, backbone.embed_dim)

    print("[sanity] running DPT head forward ...")
    # The HF backbone forward was run under no_grad in _forward_hidden_states.
    # For training, the head still needs grads w.r.t. its own params, but the
    # backbone features can be detached. Re-run with grads enabled for the
    # forward path so we can backprop into head.
    # NOTE: our backbone is frozen + eval, so detaching is correct.
    feats_detached = [(p.detach(), c.detach()) for (p, c) in feats]
    depth_pred = head(feats_detached)
    print(f"[sanity] depth_pred shape={tuple(depth_pred.shape)}")
    assert depth_pred.shape[0] == B
    assert depth_pred.shape[1] == 1

    # Fake target & loss
    target = torch.rand_like(depth_pred) * 80.0
    loss = torch.nn.functional.l1_loss(depth_pred, target)
    print(f"[sanity] dummy L1 loss={loss.item():.4f}")

    print("[sanity] running backward ...")
    loss.backward()
    grad_present = sum(
        1 for p in head.parameters() if p.grad is not None and p.grad.abs().sum() > 0
    )
    print(f"[sanity] head params with non-zero grad: {grad_present}")

    print("[sanity] ALL OK ✅")


if __name__ == "__main__":
    main()
