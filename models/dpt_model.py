"""DPT depth model: DINOv3-L (frozen, HF) + DPT head (trainable).

This is the *complete* baseline model. The forward path is:

    image (B,3,H,W) in [0,1]  ->  ImageNet normalize  ->  HFDinov3Backbone
        ->  4 intermediate layers (block 4/11/17/23)
        ->  DPTHead  ->  raw (B,1,H,W)
        ->  sigmoid * max_depth  ->  depth (B,1,H,W) in [0, max_depth]

Only the DPT head is trainable. Backbone is frozen + eval().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from my_baseline.models.backbone_hf import (
    DINOV3_VITL_FOUR_INTERVALS,
    HFDinov3Backbone,
)


# ImageNet stats — DINOv3 HF preprocessor uses these.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class DPTModelConfig:
    hf_model_id: str = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    embed_dim: int = 1024
    backbone_out_layers: Tuple[int, ...] = DINOV3_VITL_FOUR_INTERVALS
    dpt_channels: int = 256
    post_process_channels: Tuple[int, ...] = (128, 256, 512, 1024)
    readout_type: str = "project"
    n_hidden_channels: int = 32
    max_depth: float = 80.0  # KITTI
    min_depth: float = 1e-3
    use_norm_for_layers: bool = True
    detach_backbone_features: bool = True


class DPTDepthModel(nn.Module):
    """Frozen DINOv3-L + trainable DPT head, with bounded depth output."""

    def __init__(self, cfg: DPTModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or DPTModelConfig()

        # ---- backbone ----------------------------------------------------
        self.backbone = HFDinov3Backbone(
            hf_model_id=self.cfg.hf_model_id,
            freeze=True,
            dtype=torch.float32,
        )
        assert self.backbone.embed_dim == self.cfg.embed_dim, (
            f"cfg embed_dim={self.cfg.embed_dim} but backbone gave "
            f"{self.backbone.embed_dim}"
        )

        # ---- DPT head (import here to keep dependency local) ------------
        from dinov3.eval.depth.models.dpt_head import DPTHead

        self.head = DPTHead(
            in_channels=(self.cfg.embed_dim,) * len(self.cfg.backbone_out_layers),
            channels=self.cfg.dpt_channels,
            post_process_channels=list(self.cfg.post_process_channels),
            readout_type=self.cfg.readout_type,
            n_output_channels=1,
            n_hidden_channels=self.cfg.n_hidden_channels,
        )

        # Normalization buffers
        mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("imagenet_mean", mean, persistent=False)
        self.register_buffer("imagenet_std", std, persistent=False)

    # ----- override .train() so backbone always stays in eval mode -------
    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        self.backbone.eval()
        return self

    # ----- helpers --------------------------------------------------------
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Input is [0, 1] RGB; convert to ImageNet-normalized."""
        return (x - self.imagenet_mean) / self.imagenet_std

    def trainable_parameters(self):
        """Returns only the DPT head parameters (the only trainable ones)."""
        return [p for p in self.head.parameters() if p.requires_grad]

    # ----- forward --------------------------------------------------------
    def forward_raw(self, x: torch.Tensor) -> torch.Tensor:
        """Forward returning raw (unsquashed) head output (B,1,H,W)."""
        x = self._normalize(x)

        # Backbone is frozen + eval. Pull features under no_grad to save
        # activation memory, since the head does not need backbone grads.
        if self.cfg.detach_backbone_features:
            with torch.no_grad():
                feats = self.backbone.get_intermediate_layers(
                    x,
                    n=list(self.cfg.backbone_out_layers),
                    reshape=True,
                    return_class_token=True,
                    norm=self.cfg.use_norm_for_layers,
                )
            feats = [(p.detach(), c.detach()) for (p, c) in feats]
        else:
            feats = self.backbone.get_intermediate_layers(
                x,
                n=list(self.cfg.backbone_out_layers),
                reshape=True,
                return_class_token=True,
                norm=self.cfg.use_norm_for_layers,
            )

        return self.head(feats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward returning bounded depth (B,1,H,W) in (0, max_depth)."""
        raw = self.forward_raw(x)
        # sigmoid * max_depth: smooth, bounded, no log explosion in SigLoss.
        depth = torch.sigmoid(raw) * self.cfg.max_depth
        # Lower bound to avoid log(0) in SigLoss.
        depth = depth.clamp(min=self.cfg.min_depth)
        return depth
