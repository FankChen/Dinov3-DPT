# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

# =============================================================================
# 【中文导读】encoder.py —— DINOv3 backbone 的"包装器"
# -----------------------------------------------------------------------------
# 本文件不做模型计算,只做两件事:
#   (1) 决定从 ViT 的哪几层取中间 token  → BackboneLayersSet
#   (2) 取出来后整理成 DPT 想要的格式    → DinoVisionTransformerWrapper.forward
#
# 输出一个 list,长度=取的层数(默认 4),每个元素是:
#   (patch_feat: (B, D, h, w),  cls_token: (B, D))
#   其中 h = H/patch_size, w = W/patch_size。
# DPT head (下一个文件) 接到这个 list,把 4 个层级融合成一张深度图。
#
# 【师兄方向的关键改动点】
#   forward() 现在是"一口气把 24 层全跑完,再统一取 4 层"。
#   将来要做"层间 photometric 反馈",需要改成分段跑:
#     0→4 → 取 token → 出小 depth → warp 算 photo error → 把 error 喂回去
#     4→11 → 取 token → 再算 → 再喂回去 → ...
# =============================================================================

import logging
from enum import Enum

from dinov3.eval.depth.models.embed import CenterPadding, StretchToMultiple
from torch import Tensor, nn

logger = logging.getLogger("dinov3")


class BackboneLayersSet(Enum):
    # 决定从 backbone 取哪些中间层的"策略枚举"。3 选 1。
    LAST = "LAST"                              # 只取最后一层(适合分类/分割,不适合 DPT)
    FOUR_LAST = "FOUR_LAST"                    # 取最后 4 层(都偏深层语义,缺浅层细节)
    FOUR_EVEN_INTERVALS = "FOUR_EVEN_INTERVALS"  # ⭐ DPT 默认:浅+中+深均匀采样,既懂细节又懂语义


def _get_backbone_out_indices(
    model: nn.Module,
    backbone_out_layers: list[int] | tuple[int, ...] | BackboneLayersSet = BackboneLayersSet.FOUR_EVEN_INTERVALS,
):
    """
    把"策略枚举"翻译成"具体的层 index list"。
    举例(FOUR_EVEN_INTERVALS):
      ViT-S/B (12 layers) → [2, 5, 8, 11]
      ViT-L   (24 layers) → [4, 11, 17, 23]   ← 注意!正确应是 [5,11,17,23],
                                                 但官方故意保留这个 off-by-one 错误
                                                 (旧 ckpt 都是按这个训的,改了就不能加载)
      ViT-g   (40 layers) → [9, 19, 29, 39]
    """
    n_blocks = getattr(model, "n_blocks", 1)
    out_indices: list[int]
    if isinstance(backbone_out_layers, (tuple, list)):
        # 用户直接给了 list,比如 [2, 5, 8, 11],就直接用
        out_indices = list(backbone_out_layers)
    elif backbone_out_layers == BackboneLayersSet.LAST:
        out_indices = [n_blocks - 1]
    elif backbone_out_layers == BackboneLayersSet.FOUR_LAST:
        out_indices = [i for i in range(n_blocks - 4, n_blocks)]
    elif backbone_out_layers == BackboneLayersSet.FOUR_EVEN_INTERVALS:
        # ⚠️ ViT-L 的特殊兼容性 hack:为了能加载历史 ckpt,保留(数学上不对的)[4,11,17,23]
        if n_blocks == 24:
            out_indices = [4, 11, 17, 23]
        else:
            # 通用公式:把 24 层 4 等分,取每 1/4 区间的最后一层
            out_indices = [i * (n_blocks // 4) - 1 for i in range(1, 5)]
    assert all([out_index < n_blocks for out_index in out_indices])
    return out_indices


class PatchSizeAdaptationStrategy(Enum):
    # 输入图尺寸不一定是 patch_size(16) 的倍数,这里决定怎么对齐
    CENTER_PADDING = "center_padding"  # 在四周补 0,让 H、W 都变成 16 的倍数(推荐)
    STRETCH = "stretch"                # 拉伸到 16 倍数(会改变长宽比,不推荐)
    NO_ADAPTATION = "never"            # 不动(要求输入已经对齐)


class DinoVisionTransformerWrapper(nn.Module):
    """
    DINOv3 backbone 的包装器。
    输入: image  (B, 3, H, W)
    输出: list[(patch_feat (B, D, h, w), cls_token (B, D))],长度 = len(backbone_out_indices)
    """

    def __init__(
        self,
        backbone_model: nn.Module,                                                       # DINOv3 实例
        backbone_out_layers: str | tuple[int, ...] | BackboneLayersSet,                  # 取哪几层
        use_backbone_norm: bool = False,                                                 # 取出来要不要过 LayerNorm
        adapt_to_patch_size: PatchSizeAdaptationStrategy = PatchSizeAdaptationStrategy.CENTER_PADDING,
    ):
        super().__init__()

        self.final_norm = use_backbone_norm
        self.backbone = backbone_model
        # 字符串("FOUR_EVEN_INTERVALS")也允许,转成枚举
        if isinstance(backbone_out_layers, str):
            backbone_out_layers = BackboneLayersSet(backbone_out_layers)
        # ⭐ 计算出具体取哪几层,例如 [4, 11, 17, 23]
        self.backbone_out_indices = _get_backbone_out_indices(self.backbone, backbone_out_layers=backbone_out_layers)

        # 记下每层 token 的维度,DPT head 后面要用
        # ViT-L: 每层 D=1024,所以 self.embed_dims = [1024, 1024, 1024, 1024]
        try:
            embed_dims: list[int] = getattr(self.backbone, "embed_dims")
        except AttributeError:
            embed_dim: int = getattr(self.backbone, "embed_dim")
            n_blocks: int = getattr(self.backbone, "n_blocks")
            logger.warning(f"Backbone does not define embed_dims, using {[embed_dim] * n_blocks} instead")
            embed_dims = [embed_dim] * n_blocks
        self.embed_dims = [embed_dims[idx] for idx in self.backbone_out_indices]

        # 准备一个 "patch size adapter":前向时先把图 pad/stretch 到 patch_size 的倍数
        try:
            input_pad_size = getattr(self.backbone, "input_pad_size")
        except AttributeError:
            patch_size = getattr(self.backbone, "patch_size")
            logger.warning(f"Backbone does not define input_pad_size, using {patch_size=} instead")
            input_pad_size = patch_size
        self.patch_size_adapter: nn.Module = nn.Identity()
        if adapt_to_patch_size is PatchSizeAdaptationStrategy.CENTER_PADDING:
            self.patch_size_adapter = CenterPadding(input_pad_size)
        elif adapt_to_patch_size is PatchSizeAdaptationStrategy.STRETCH:
            self.patch_size_adapter = StretchToMultiple(input_pad_size)

        # ⭐⭐ 冻结 backbone 全部参数 —— 整个 DINOv3 不再更新
        # 这是 self-supervised baseline 的标准玩法:特征已经预训练好,只训 head
        self.backbone.requires_grad_(False)

    def forward(
        self,
        x: Tensor,  # [B, 3, H, W] 输入 RGB
    ) -> list[tuple[Tensor, Tensor]]:
        # ① pad 到 patch_size 倍数(否则 ViT 切不齐)
        x = self.patch_size_adapter(x)
        # ② 一次跑完 ViT,取出指定层的 token + reshape 成 feature map
        #    这是 DINOv3 内置的工具方法,黑盒。
        #    返回值已经按 reshape=True 整理好:
        #       patch_feat:  (B, D, h, w)   ← 已 drop CLS 并 reshape
        #       class_token: (B, D)
        outputs = self.backbone.get_intermediate_layers(
            x,
            n=self.backbone_out_indices,    # 例 [4, 11, 17, 23]
            reshape=True,                    # token 序列 → 二维 feature map
            return_class_token=True,         # 同时把 CLS token 单独返回
            norm=self.final_norm,            # 取 LN 前 还是 LN 后
        )
        # ③ 直接返回给 DPT head
        # 【未来要改的地方】师兄方向需要把 forward 拆成"分段跑",
        #    不再一次性 get_intermediate_layers,而是每跑几层暂停一次,
        #    在中间插入 photometric 反馈模块。
        return outputs
