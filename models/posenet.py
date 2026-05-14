"""PoseNet — 输入相邻两帧,输出 6DoF 相对位姿。

架构(monodepth2 标准):
    [I_tgt, I_src] (B, 6, H, W)
      └─> ResNet18 encoder (conv1 权重 stride-concat 加倍处理多帧输入)
      └─> PoseDecoder: 4 个 conv 头
      └─> (axisangle (B,1,3), translation (B,1,3)),已乘 0.01 缩放

用法:
    >>> net = PoseNet(num_layers=18, pretrained=True)
    >>> ax, tr = net(torch.stack([img_tgt, img_src], dim=2))     # 也可以 cat 在 channel
    >>> T = transformation_from_parameters(ax[:,0], tr[:,0], invert=False)  # (B,4,4)

注意:
    - PoseNet 训练时通常预测 tgt → src 的位姿,inverse_warp 时填 T_tgt2src
    - monodepth2 用 ImageNet 预训练 ResNet18 初始化效果显著,建议保持 pretrained=True
    - 我们做的是 photometric refiner,跟 monodepth2 的 self-sup 完全一致
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm


# -----------------------------------------------------------------------------
# ResNet encoder,支持多帧拼 channel 输入(monodepth2 的小 trick)
# -----------------------------------------------------------------------------

class _MultiImageResNetEncoder(nn.Module):
    """ResNet18/50, 可选 num_input_images>1:把 conv1 的输入通道从 3 扩到 3*N。

    权重处理:把 ImageNet 预训练的 conv1.weight 复制 N 份再 / N(保持响应均值)。
    """

    def __init__(self, num_layers: int = 18, pretrained: bool = True,
                 num_input_images: int = 2):
        super().__init__()
        assert num_layers in (18, 50), "PoseNet 一般只用 ResNet18/50"
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        if num_layers == 50:
            self.num_ch_enc[1:] *= 4

        weights = (
            tvm.ResNet18_Weights.IMAGENET1K_V1 if num_layers == 18
            else tvm.ResNet50_Weights.IMAGENET1K_V1
        ) if pretrained else None
        builder = tvm.resnet18 if num_layers == 18 else tvm.resnet50
        backbone = builder(weights=weights)

        # 扩展 conv1 输入通道
        if num_input_images > 1:
            old = backbone.conv1
            new = nn.Conv2d(
                3 * num_input_images, old.out_channels,
                kernel_size=old.kernel_size, stride=old.stride,
                padding=old.padding, bias=False,
            )
            with torch.no_grad():
                new.weight.copy_(
                    old.weight.repeat(1, num_input_images, 1, 1) / num_input_images
                )
            backbone.conv1 = new

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        # 输入归一化到 ImageNet 统计(若上游没做)
        feats = []
        x = self.relu(self.bn1(self.conv1(x)))
        feats.append(x)
        x = self.layer1(self.maxpool(x))
        feats.append(x)
        x = self.layer2(x); feats.append(x)
        x = self.layer3(x); feats.append(x)
        x = self.layer4(x); feats.append(x)
        return feats


# -----------------------------------------------------------------------------
# PoseDecoder
# -----------------------------------------------------------------------------

class _PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc: np.ndarray,
                 num_input_features: int = 1,
                 num_frames_to_predict_for: int = 1,
                 stride: int = 1):
        super().__init__()
        self.num_input_features = num_input_features
        self.num_frames_to_predict_for = num_frames_to_predict_for
        self.squeeze = nn.Conv2d(num_ch_enc[-1], 256, 1)
        self.pose_0 = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.pose_1 = nn.Conv2d(256, 256, 3, stride, 1)
        self.pose_2 = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_features: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # 只用最后一层 feature
        last = input_features[-1]
        x = self.relu(self.squeeze(last))
        x = self.relu(self.pose_0(x))
        x = self.relu(self.pose_1(x))
        x = self.pose_2(x)
        x = x.mean(dim=(2, 3))                              # 全局平均池化
        x = 0.01 * x.view(-1, self.num_frames_to_predict_for, 1, 6)
        axisangle = x[..., :3]                              # (B, F, 1, 3)
        translation = x[..., 3:]                            # (B, F, 1, 3)
        return axisangle.squeeze(2), translation.squeeze(2)  # (B, F, 3) each


# -----------------------------------------------------------------------------
# 对外类
# -----------------------------------------------------------------------------

class PoseNet(nn.Module):
    """输入 (B, 3*N, H, W) 拼在 channel 维的 N 帧图,输出 N-1 段相对位姿。"""

    def __init__(self, num_layers: int = 18, pretrained: bool = True,
                 num_input_images: int = 2, num_frames_to_predict_for: int = 1):
        super().__init__()
        self.encoder = _MultiImageResNetEncoder(
            num_layers=num_layers, pretrained=pretrained,
            num_input_images=num_input_images,
        )
        self.decoder = _PoseDecoder(
            num_ch_enc=self.encoder.num_ch_enc,
            num_input_features=1,
            num_frames_to_predict_for=num_frames_to_predict_for,
        )

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """frames: (B, 3*N, H, W) — 把 N 帧 RGB cat 在 channel 维"""
        feats = self.encoder(frames)
        axisangle, translation = self.decoder(feats)
        return axisangle, translation
