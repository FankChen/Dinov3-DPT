"""Sanity check for my_baseline.models — 跑一次空数据,确保 shape / 梯度 / device 都 OK。

Usage:
    python -m my_baseline.scripts.test_models
or:
    cd dinov3_baseline && python -c "from my_baseline.scripts.test_models import main; main()"
"""
from __future__ import annotations

import torch

from my_baseline.models import (
    InverseWarp, PoseNet, SSIM,
    transformation_from_parameters,
    photometric_reconstruction_loss,
    edge_aware_smoothness_loss,
    compute_depth_errors,
)


def main(device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    print(f"[test] device = {device}")
    B, H, W = 2, 192, 640

    # ---- 1. PoseNet: 输入 2 帧 cat 在 channel ----
    posenet = PoseNet(num_layers=18, pretrained=False, num_input_images=2).to(device)
    frames = torch.randn(B, 6, H, W, device=device)
    ax, tr = posenet(frames)
    print(f"[test] PoseNet axisangle  : {tuple(ax.shape)}  (expect (B,1,3))")
    print(f"[test] PoseNet translation: {tuple(tr.shape)}  (expect (B,1,3))")
    assert ax.shape == (B, 1, 3) and tr.shape == (B, 1, 3)

    # ---- 2. axis-angle + translation → 4×4 ----
    T = transformation_from_parameters(ax[:, 0], tr[:, 0], invert=False)  # (B,4,4)
    print(f"[test] SE(3) matrix      : {tuple(T.shape)}  (expect (B,4,4))")
    assert T.shape == (B, 4, 4)
    # bottom row 应是 [0,0,0,1]
    assert torch.allclose(T[:, 3, :], torch.tensor([0., 0., 0., 1.], device=device).expand(B, -1), atol=1e-5)

    # ---- 3. InverseWarp ----
    warp = InverseWarp(H, W).to(device)
    src_img = torch.randn(B, 3, H, W, device=device)
    tgt_img = torch.randn(B, 3, H, W, device=device)
    depth = torch.rand(B, 1, H, W, device=device) * 10 + 0.1            # [0.1, 10.1]

    # 简化的 K (focal=H/2 之类),实际训练用真 KITTI K
    K = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)
    K[:, 0, 0] = W / 2.0; K[:, 1, 1] = W / 2.0
    K[:, 0, 2] = W / 2.0; K[:, 1, 2] = H / 2.0
    inv_K = torch.linalg.inv(K)

    reproj, pix = warp(src_img, depth, T, K, inv_K)
    print(f"[test] reproj img        : {tuple(reproj.shape)}  (expect (B,3,H,W))")
    print(f"[test] pix coords        : {tuple(pix.shape)}      (expect (B,H,W,2))")
    assert reproj.shape == src_img.shape
    assert pix.shape == (B, H, W, 2)

    # ---- 4. Photometric loss ----
    pe = photometric_reconstruction_loss(tgt_img, reproj)
    print(f"[test] photo loss map    : {tuple(pe.shape)}  (expect (B,1,H,W))")
    print(f"[test] photo loss mean   : {pe.mean().item():.4f}")
    assert pe.shape == (B, 1, H, W)

    # ---- 5. Smoothness ----
    sm = edge_aware_smoothness_loss(1.0 / depth, tgt_img)
    print(f"[test] smoothness        : {sm.item():.4f}")

    # ---- 6. 反向传播 ----
    loss = pe.mean() + 0.001 * sm
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in posenet.parameters())
    print(f"[test] backward OK, posenet has grads = {has_grad}")
    assert has_grad

    # ---- 7. Depth metrics ----
    gt = torch.rand(1000, device=device) * 50 + 1
    pred = gt + torch.randn(1000, device=device) * 2
    pred = pred.clamp(min=0.1)
    err = compute_depth_errors(gt, pred)
    print(f"[test] compute_depth_errors = {err}")

    print("\n[test] ✅ ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
