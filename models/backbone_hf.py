"""DINOv3 ViT backbone wrapper around HuggingFace `transformers` weights.

Why this file exists
--------------------
- The official DINOv3 release on HuggingFace
  (`facebook/dinov3-vitl16-pretrain-lvd1689m`) ships the ViT weights in the
  `transformers` `DINOv3ViTModel` format (token-sequence outputs).
- The DPT head we reuse (`dinov3.eval.depth.models.dpt_head.DPTHead`) and the
  encoder wrapper (`dinov3.eval.depth.models.encoder.DinoVisionTransformerWrapper`)
  expect a backbone that exposes
  `get_intermediate_layers(x, n, reshape=True, return_class_token=True, norm=...)`
  returning a `List[Tuple[(B,C,h,w), (B,C)]]`.

This module bridges the two: it loads the HF model and exposes the
DINOv3-native intermediate-layers API, so we can plug it straight into
`DinoVisionTransformerWrapper` and `DPTHead` without touching the upstream
`dinov3/` tree (read-only).

Notes
-----
- Token layout of HF `DINOv3ViTModel` for ViT-L/16 with default settings:
  ``[CLS] + 4 register tokens + (H/16 * W/16) patch tokens`` per layer.
  We slice the first ``1 + num_register_tokens`` tokens off and keep only the
  patch tokens, then reshape to ``(B, C, h, w)``.
- For ViT-L (24 blocks) the four-even-intervals indices from the official
  config are ``[4, 11, 17, 23]`` (kept for backward compatibility with the
  released DPT weights; the upstream encoder.py marks it as historically
  off-by-one but preserves it). We follow the same convention.
- All backbone params are frozen here. The DPT head is the only trainable
  component in the baseline.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import torch
import torch.nn as nn

try:
    from transformers import AutoModel
except ImportError as e:  # pragma: no cover - hard dep
    raise ImportError(
        "transformers is required for the HF DINOv3 backbone. "
        "Run: pip install transformers safetensors"
    ) from e


# ViT-L/16 four-even-intervals (matches dinov3 encoder.py FOUR_EVEN_INTERVALS
# for n_blocks=24). Used by the released DPT head.
DINOV3_VITL_FOUR_INTERVALS: Tuple[int, ...] = (4, 11, 17, 23)


class HFDinov3Backbone(nn.Module):
    """Wraps a HuggingFace `DINOv3ViTModel` to mimic the dinov3-native API.

    Exposed attributes / methods (the subset that
    `DinoVisionTransformerWrapper` and `DPTHead` rely on):

    - ``embed_dim`` (int): channel dim of the backbone features.
    - ``patch_size`` (int): patch side length in pixels.
    - ``n_blocks`` (int): total transformer blocks.
    - ``num_register_tokens`` (int): how many special non-CLS tokens precede
      the patch tokens (4 for DINOv3 ViT-L).
    - ``get_intermediate_layers(x, n, reshape, return_class_token, norm)``:
      returns `List[Tuple[(B,C,h,w), (B,C)]]` for the requested layer indices.
    """

    def __init__(
        self,
        hf_model_id: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        freeze: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.hf_model_id = hf_model_id
        self.model = AutoModel.from_pretrained(hf_model_id, torch_dtype=dtype)
        self.model.eval()

        cfg = self.model.config
        # transformers HF config naming for DINOv3 ViT
        self.embed_dim: int = int(getattr(cfg, "hidden_size"))
        self.patch_size: int = int(getattr(cfg, "patch_size", 16))
        self.n_blocks: int = int(getattr(cfg, "num_hidden_layers"))
        # num_register_tokens: HF config exposes this for DINOv3
        self.num_register_tokens: int = int(getattr(cfg, "num_register_tokens", 4))
        # Number of special tokens before patch tokens: 1 CLS + registers
        self._n_special: int = 1 + self.num_register_tokens

        if freeze:
            self.requires_grad_(False)

    # ---- API expected by dinov3 encoder.py / DPT head -------------------

    @torch.no_grad()
    def _forward_hidden_states(self, x: torch.Tensor):
        """Run the HF model and return all hidden_states (tuple of len n_blocks+1)."""
        out = self.model(pixel_values=x, output_hidden_states=True, return_dict=True)
        return out.hidden_states  # tuple: (embed_out, block1_out, ..., blockN_out)

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence[int]],
        reshape: bool = True,
        return_class_token: bool = True,
        norm: bool = True,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """DINOv3-compatible intermediate layers API.

        Args:
            x: image batch ``(B, 3, H, W)``, normalized in the same range the
               HF processor uses (ImageNet mean/std).
            n: either an int (take last ``n`` blocks) or a sequence of block
               indices (0-based, into the transformer blocks).
            reshape: if True, reshape patch tokens to ``(B, C, h, w)``.
            return_class_token: if True, also return CLS token per layer.
            norm: if True, apply the final layernorm to the selected layers.
                (HF DINOv3 only applies final norm to the *last* hidden state.
                For consistency with the dinov3-native behavior we apply the
                model's final ``layernorm`` to every selected layer when
                ``norm=True``.)

        Returns:
            list of length ``len(n)`` (or ``n`` if int). Each element is a
            tuple ``(patch_feats, cls_token)`` where
              - ``patch_feats``: ``(B, C, h, w)`` if reshape else ``(B, Np, C)``
              - ``cls_token``: ``(B, C)``
            If ``return_class_token=False``, the inner tuples become just
            ``patch_feats``.
        """
        # Run the HF model with hidden_states output
        with torch.set_grad_enabled(self.training and any(p.requires_grad for p in self.parameters())):
            hs = self._forward_hidden_states(x)

        # hs[0] = patch_embed output BEFORE block 1
        # hs[k] = output after block k (1-indexed in dinov3 convention).
        # Match dinov3 indexing: index 0 -> block 1 output, etc.
        # encoder.py uses `n=indices` where indices like [4,11,17,23] are
        # 0-based block indices for ViT-L (n_blocks=24). With dinov3 native
        # `get_intermediate_layers`, those correspond to outputs of blocks
        # 5, 12, 18, 24 (1-based) i.e. hs[5], hs[12], hs[18], hs[24].
        # So: hs_index = block_index + 1.
        if isinstance(n, int):
            block_indices = list(range(self.n_blocks - n, self.n_blocks))
        else:
            block_indices = list(n)

        # Optional final norm
        final_ln = getattr(self.model, "layernorm", None) or getattr(self.model, "norm", None)

        outputs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        H, W = x.shape[-2], x.shape[-1]
        h, w = H // self.patch_size, W // self.patch_size

        for bi in block_indices:
            tokens = hs[bi + 1]  # (B, 1 + nreg + Np, C)
            if norm and final_ln is not None:
                tokens = final_ln(tokens)
            cls_tok = tokens[:, 0]  # (B, C)
            patch_tok = tokens[:, self._n_special:]  # (B, Np, C)
            if reshape:
                B, Np, C = patch_tok.shape
                assert Np == h * w, (
                    f"Patch token count {Np} != h*w {h}*{w}={h * w}. "
                    f"Input H={H}, W={W}, patch_size={self.patch_size}."
                )
                patch_tok = patch_tok.transpose(1, 2).reshape(B, C, h, w).contiguous()
            if return_class_token:
                outputs.append((patch_tok, cls_tok))
            else:
                outputs.append(patch_tok)  # type: ignore[arg-type]

        return outputs

    # Some dinov3 code paths use `forward_features`; keep a stub for safety.
    def forward(self, x: torch.Tensor):  # pragma: no cover - not used in DPT path
        return self.get_intermediate_layers(x, n=[self.n_blocks - 1])
