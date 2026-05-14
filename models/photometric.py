"""几何 + 光度模块,从 monodepth2 抠出来,做轻度整理。

模块清单:
    BackprojectDepth  : 深度图 + K⁻¹ → 相机系 3D 点云 (B,4,H*W)
    Project3D         : 3D 点 + K + T → 目标视角的归一化像素坐标 (B,H,W,2)
    transformation_from_parameters : axis-angle + translation → 4×4 SE(3)
    inverse_warp      : 一步到位:src 图按 (depth, pose, K) 扭到 tgt 视角
    ssim, photometric_reconstruction_loss, edge_aware_smoothness_loss

用法:
    >>> warp = InverseWarp(H, W)
    >>> reproj = warp(src_img, depth_tgt, T_tgt2src, K, inv_K)
    >>> loss = photometric_reconstruction_loss(tgt_img, reproj)

为什么不直接 import monodepth2:
    - 那个 repo 是 non-commercial license,带进我们 repo 可能有版权问题
    - 也只用其中 200 行,自己重写一遍维护成本反而低
    - 我们的 batch 尺寸是动态的(DDP),BackprojectDepth 用 nn.Parameter 存固定 batch
      预生成坐标网格会出问题,这里改成 register_buffer 形式
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# (1) 几何 — 深度 + 相机内外参 → warp
# =============================================================================

class BackprojectDepth(nn.Module):
    """把 (B,1,H,W) 的 depth + K⁻¹ 投影到相机系 3D 点云。

    返回:cam_points (B, 4, H*W) — 齐次坐标
    """

    def __init__(self, height: int, width: int):
        super().__init__()
        self.height = height
        self.width = width

        # 像素网格 (homogeneous): (3, H*W)
        ys, xs = torch.meshgrid(
            torch.arange(height, dtype=torch.float32),
            torch.arange(width, dtype=torch.float32),
            indexing="ij",
        )
        ones = torch.ones_like(xs)
        pix_coords = torch.stack([xs, ys, ones], dim=0).reshape(3, -1)  # (3, H*W)
        self.register_buffer("pix_coords", pix_coords, persistent=False)
        self.register_buffer(
            "ones_row", torch.ones(1, height * width), persistent=False
        )

    def forward(self, depth: torch.Tensor, inv_K: torch.Tensor) -> torch.Tensor:
        # depth: (B, 1, H, W)   inv_K: (B, 4, 4)
        B = depth.shape[0]
        # (B, 3, H*W)
        pix = self.pix_coords.unsqueeze(0).expand(B, -1, -1)
        # cam_dir = K⁻¹ · pix
        cam_dir = torch.matmul(inv_K[:, :3, :3], pix)
        cam_points = depth.view(B, 1, -1) * cam_dir         # (B, 3, H*W)
        ones = self.ones_row.unsqueeze(0).expand(B, -1, -1)
        cam_points = torch.cat([cam_points, ones], dim=1)    # (B, 4, H*W)
        return cam_points


class Project3D(nn.Module):
    """把 3D 点云(齐次)按 K · T 投影成像素坐标(归一化到 [-1, 1] 给 grid_sample)。"""

    def __init__(self, height: int, width: int, eps: float = 1e-7):
        super().__init__()
        self.height = height
        self.width = width
        self.eps = eps

    def forward(
        self, points: torch.Tensor, K: torch.Tensor, T: torch.Tensor
    ) -> torch.Tensor:
        # points: (B, 4, H*W)   K: (B, 4, 4)   T: (B, 4, 4)
        B = points.shape[0]
        P = torch.matmul(K, T)[:, :3, :]                     # (B, 3, 4)
        cam_points = torch.matmul(P, points)                  # (B, 3, H*W)
        pix = cam_points[:, :2] / (cam_points[:, 2:3] + self.eps)
        pix = pix.view(B, 2, self.height, self.width).permute(0, 2, 3, 1)
        # 归一化到 [-1, 1](grid_sample 需要)
        pix[..., 0] = 2.0 * pix[..., 0] / (self.width - 1) - 1.0
        pix[..., 1] = 2.0 * pix[..., 1] / (self.height - 1) - 1.0
        return pix


# =============================================================================
# (2) Axis-angle + translation → 4×4 SE(3) 矩阵 (monodepth2 套路)
# =============================================================================


def _rot_from_axisangle(vec: torch.Tensor) -> torch.Tensor:
    """vec: (..., 3) axis-angle → 4×4 旋转矩阵(右下补 1)"""
    flat = vec.reshape(-1, 3)
    angle = flat.norm(dim=1, keepdim=True).clamp(min=1e-7)
    axis = flat / angle
    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca
    x, y, z = axis[:, 0:1], axis[:, 1:2], axis[:, 2:3]
    xs, ys, zs = x * sa, y * sa, z * sa
    xC, yC, zC = x * C, y * C, z * C
    xyC, yzC, zxC = x * y * C, y * z * C, z * x * C

    R = torch.zeros(flat.shape[0], 4, 4, device=vec.device, dtype=vec.dtype)
    R[:, 0, 0] = (x * xC + ca).squeeze(-1)
    R[:, 0, 1] = (xyC - zs).squeeze(-1)
    R[:, 0, 2] = (zxC + ys).squeeze(-1)
    R[:, 1, 0] = (xyC + zs).squeeze(-1)
    R[:, 1, 1] = (y * yC + ca).squeeze(-1)
    R[:, 1, 2] = (yzC - xs).squeeze(-1)
    R[:, 2, 0] = (zxC - ys).squeeze(-1)
    R[:, 2, 1] = (yzC + xs).squeeze(-1)
    R[:, 2, 2] = (z * zC + ca).squeeze(-1)
    R[:, 3, 3] = 1
    return R.view(*vec.shape[:-1], 4, 4)


def _translation_matrix(t: torch.Tensor) -> torch.Tensor:
    """t: (..., 3) → 4×4 平移矩阵"""
    flat = t.reshape(-1, 3)
    T = torch.zeros(flat.shape[0], 4, 4, device=t.device, dtype=t.dtype)
    T[:, 0, 0] = T[:, 1, 1] = T[:, 2, 2] = T[:, 3, 3] = 1
    T[:, :3, 3] = flat
    return T.view(*t.shape[:-1], 4, 4)


def transformation_from_parameters(
    axisangle: torch.Tensor, translation: torch.Tensor, invert: bool = False
) -> torch.Tensor:
    """(axis-angle, translation) → 4×4 SE(3)。invert=True 返回逆变换。"""
    R = _rot_from_axisangle(axisangle)
    t = translation.clone()
    if invert:
        R = R.transpose(-1, -2)
        t = -t
    T = _translation_matrix(t)
    if invert:
        return torch.matmul(R, T)
    return torch.matmul(T, R)


# =============================================================================
# (3) 一站式 inverse warping
# =============================================================================


class InverseWarp(nn.Module):
    """给定 (src_img, tgt_depth, T_tgt2src, K, inv_K),返回 src 图扭到 tgt 视角的结果。

    工作流:
        1. tgt depth + inv_K  → tgt 视角下的 3D 点 (相机系)
        2. T_tgt2src + K      → 投到 src 视角的像素坐标 (B,H,W,2) ∈ [-1,1]
        3. F.grid_sample(src) → 重建的 tgt 视角图
    """

    def __init__(self, height: int, width: int):
        super().__init__()
        self.backproject = BackprojectDepth(height, width)
        self.project = Project3D(height, width)

    def forward(
        self,
        src_img: torch.Tensor,    # (B, 3, H, W)
        tgt_depth: torch.Tensor,  # (B, 1, H, W)
        T_tgt2src: torch.Tensor,  # (B, 4, 4)
        K: torch.Tensor,          # (B, 4, 4)
        inv_K: torch.Tensor,      # (B, 4, 4)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cam_points = self.backproject(tgt_depth, inv_K)
        pix_coords = self.project(cam_points, K, T_tgt2src)
        warped = F.grid_sample(
            src_img, pix_coords, mode="bilinear",
            padding_mode="border", align_corners=True,
        )
        # 同时返回采样坐标(后面可用于 mask 越界像素)
        return warped, pix_coords


# =============================================================================
# (4) 光度损失:SSIM + L1 (monodepth2 公式)
# =============================================================================


class SSIM(nn.Module):
    """SSIM(I, J) → (B, 3, H, W),数值越小越相似(0=完全一致)"""

    def __init__(self):
        super().__init__()
        self.refl = nn.ReflectionPad2d(1)
        self.mu_x = nn.AvgPool2d(3, 1)
        self.mu_y = nn.AvgPool2d(3, 1)
        self.sig_x = nn.AvgPool2d(3, 1)
        self.sig_y = nn.AvgPool2d(3, 1)
        self.sig_xy = nn.AvgPool2d(3, 1)
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.refl(x)
        y = self.refl(y)
        mu_x = self.mu_x(x)
        mu_y = self.mu_y(y)
        s_x = self.sig_x(x ** 2) - mu_x ** 2
        s_y = self.sig_y(y ** 2) - mu_y ** 2
        s_xy = self.sig_xy(x * y) - mu_x * mu_y
        num = (2 * mu_x * mu_y + self.C1) * (2 * s_xy + self.C2)
        den = (mu_x ** 2 + mu_y ** 2 + self.C1) * (s_x + s_y + self.C2)
        return torch.clamp((1 - num / den) / 2, 0, 1)


def photometric_reconstruction_loss(
    tgt_img: torch.Tensor,
    reproj_img: torch.Tensor,
    ssim_module: SSIM | None = None,
    alpha: float = 0.85,
) -> torch.Tensor:
    """monodepth2 公式: α·SSIM + (1-α)·L1,先 per-pixel,再外面做 min-over-source / mean。"""
    l1 = (tgt_img - reproj_img).abs().mean(dim=1, keepdim=True)            # (B,1,H,W)
    if ssim_module is None:
        ssim_module = SSIM().to(tgt_img.device)
    ssim_val = ssim_module(tgt_img, reproj_img).mean(dim=1, keepdim=True)  # (B,1,H,W)
    return alpha * ssim_val + (1 - alpha) * l1


def edge_aware_smoothness_loss(disp: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
    """让 disp 在图像边缘处可以陡,在图像平坦处必须平滑(monodepth2 公式)。

    通常 disp 在 mean-normalize 之后用更稳:
        disp_norm = disp / (disp.mean([2,3], keepdim=True) + 1e-7)
    """
    grad_d_x = (disp[:, :, :, :-1] - disp[:, :, :, 1:]).abs()
    grad_d_y = (disp[:, :, :-1, :] - disp[:, :, 1:, :]).abs()
    grad_i_x = (img[:, :, :, :-1] - img[:, :, :, 1:]).abs().mean(1, keepdim=True)
    grad_i_y = (img[:, :, :-1, :] - img[:, :, 1:, :]).abs().mean(1, keepdim=True)
    grad_d_x = grad_d_x * torch.exp(-grad_i_x)
    grad_d_y = grad_d_y * torch.exp(-grad_i_y)
    return grad_d_x.mean() + grad_d_y.mean()


# =============================================================================
# (5) 评估指标(标准 KITTI / NYU 套路,跟 baseline 评估对齐)
# =============================================================================


def compute_depth_errors(gt: torch.Tensor, pred: torch.Tensor) -> dict[str, float]:
    """gt, pred: (N,) 1D tensors,已 mask 掉无效像素。"""
    thresh = torch.max(gt / pred, pred / gt)
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()
    rmse = torch.sqrt(((gt - pred) ** 2).mean())
    rmse_log = torch.sqrt(((gt.log() - pred.log()) ** 2).mean())
    abs_rel = ((gt - pred).abs() / gt).mean()
    sq_rel = (((gt - pred) ** 2) / gt).mean()
    return {
        "abs_rel": abs_rel.item(),
        "sq_rel": sq_rel.item(),
        "rmse": rmse.item(),
        "rmse_log": rmse_log.item(),
        "a1": a1.item(),
        "a2": a2.item(),
        "a3": a3.item(),
    }
