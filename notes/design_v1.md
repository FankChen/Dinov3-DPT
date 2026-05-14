# Design Doc: GRU Iterative Refinement on DINOv3+DPT with Dense Photometric Feedback

**Author:** Liren  | **Advisor:** Prof. Wang  | **Date:** 2026-05-14  | **Status:** v1, draft

---

## TL;DR

Treat the frozen DINOv3+DPT baseline as an **initial depth predictor** `d₀ = DPT(DINOv3(I))`.
Append a lightweight **GRU-based iterative refiner** (inspired by FoundationStereo, but dense pixel-level instead of feature-level as in DROID-SLAM).
At each iteration `k`, use the **inverse-warping photometric residual** (monodepth2 style) as the GRU's update signal:
`d_{k+1} = d_k + GRU(d_k, residual_k, h_k)`.
Backbone and DPT head remain frozen; only PoseNet (~3M params) and refiner GRU (~10M params) are trained.

---

## 1. Background & Baseline Status

- DINOv3 ViT-L + DPT head reproduced on cluster.
- Trained on NYUv2 labeled subset (795/654) for 38 400 iter, 1×H200 MIG, 1h47min.
- Result: **AbsRel 0.0908, δ₁ 0.9329, RMSE 0.3362**.
- Codebase: https://github.com/FankChen/Dinov3-DPT (commit `a…`).
- Far from paper (BTS 24k → AbsRel ~0.04) but pipeline / loss / DDP / eval all clean.

---

## 2. Method (v1 design)

### 2.1 Architecture

```
                I_t  (target frame)         I_{t±1}  (source frame)
                 │                              │
       ┌─────────┴────────┐                     │
       │ DINOv3 backbone  │ frozen              │
       │ + DPT head       │                     │
       └─────────┬────────┘                     │
                 │  d₀  (H×W initial depth)     │
                 ▼                              │
       ┌──────────────────────────┐             │
       │   Iterative Refiner      │             │
       │                          │             │
       │   for k = 1..K:          │             │
       │     P = PoseNet(I_t,     │◄────────────┘
       │                I_{t±1})  │
       │     Î = warp(I_{t±1},    │
       │              d_k, P, K)  │  ← inverse warping (dense, pixel-level)
       │     r_k = I_t − Î        │  ← photometric residual (H×W×3)
       │     c_k = ConvEnc(r_k)   │
       │     h_{k+1}, Δd =        │
       │       ConvGRU(h_k,       │
       │         [c_k, d_k, F])   │  F = optional DPT mid-feature
       │     d_{k+1} = d_k + Δd   │
       │                          │
       └──────────────────────────┘
                 │
                 ▼
            d_K  (refined depth)
```

### 2.2 Why "dense" (not feature-level)
Advisor's note: unlike DROID-SLAM (sparse feature correspondence for SLAM optimisation),
our refinement operates on **full-resolution photometric residual maps**. This keeps
the geometric signal tied directly to image evidence rather than to learned tokens.

### 2.3 Components — sizes & sources

| Component | Source / Inspiration | Params | Trainable |
|-----------|---------------------|-------:|:---------:|
| DINOv3 ViT-L backbone | facebookresearch/dinov3 | ~300 M | ❌ |
| DPT head | dinov3 official | ~30 M | ❌ |
| PoseNet | monodepth2 (ResNet-18 enc + 4-conv pose head) | ~13 M | ✅ |
| ConvGRU refiner | FoundationStereo `update_block` adapted | ~10 M | ✅ |
| Photometric warp module | monodepth2 `BackprojectDepth` + `Project3D` | 0 | — |

Total trainable: **~23 M** (very small relative to frozen 330 M).

---

## 3. Data & Supervision

### 3.1 Dataset: **KITTI Eigen split**
- Already on cluster at `/home/izi2sgh/MYDATA/kitti/` (full raw + depth_annotated + Eigen split).
- Standard depth-SOTA benchmark; both stereo and temporal pairs available.

### 3.2 Loss (3-term, monodepth2 style)
```
L = α · L_photometric(I_t, Î_{t±1→t})        # SSIM + L1 reconstruction
  + β · L_smooth(d_K, I_t)                    # edge-aware smoothness
  + γ · L_sup(d_K, d_GT)        (optional)    # KITTI sparse GT (improved annotated)
```
- α=1.0, β=1e-3, γ=0 or 1 (ablation).
- Photometric loss applied to **all** intermediate `d_k`, k=1..K (RAFT-style supervision schedule).

---

## 4. Experiments Plan

### 4.1 Core comparison (the headline table)

| Method | AbsRel ↓ | δ₁ ↑ | RMSE ↓ |
|--------|---------:|-----:|-------:|
| DINOv3+DPT (frozen, our baseline) | — | — | — |
| + supervised fine-tune of DPT only | — | — | — |
| **+ ours (refiner, no GT)** | — | — | — |
| **+ ours (refiner, with GT)** | — | — | — |

### 4.2 Ablations

1. **K** (iterations): 1, 3, 5, 10, 20.
2. **GRU input**: residual only / residual + DPT feature / + cost volume.
3. **Refiner resolution**: ¼, ½, full.
4. **Loss weights**: pure self-sup vs supervised vs hybrid.
5. **Backbone**: DINOv3-L / DINOv2-L / DepthAnythingV2-L (frozen).
6. **Generalization**: train KITTI, test NYU / ScanNet (zero-shot).

Total ≈ 20 runs × ~3 h each ≈ 60 H200-hours. Budget OK.

---

## 5. Risks & mitigations

| Risk | Mitigation |
|------|-----------|
| Refiner overfits to small KITTI Eigen train set | self-sup loss has no overfitting target; smoothness + photometric act as regulariser |
| PoseNet bad → residual noisy → refiner unstable | warmup: train PoseNet alone with monodepth2 default for 3 epoch first |
| GRU on full resolution = OOM | start at ¼ res (RAFT default), upsample at end |
| Frozen backbone too rigid (no info flow back) | this is by design; if numbers stall, ablation (5) tries adapter / LoRA |

---

## 6. Timeline (rough, 4 weeks)

| Week | Goal |
|------|------|
| 1 | implement PoseNet + warp + photometric loss (verify on KITTI w/ GT depth only) |
| 2 | plug in ConvGRU refiner, sanity check K=3 |
| 3 | full sweep K / input ablations + headline table |
| 4 | generalization tests + writeup |

---

## 7. Open questions for advisor

1. **Pose source**: stick with PoseNet self-sup, or use KITTI GT pose (cleaner but less general)?
2. **Backbone unfreeze**: would you object to a late-stage adapter / LoRA experiment on the backbone?
3. **Contribution framing**: position as "iterative refinement on a frozen foundation depth model" or "photometric self-supervised refiner that works on any monocular depth"?
4. **Reference comparison**: which prior work do you most want us to outperform — monodepth2 era (FeatDepth/ManyDepth), or recent supervised SOTA (ZoeDepth, DepthAnythingV2)?

---

*References:*
[1] Godard et al., *Digging Into Self-Supervised Monocular Depth Estimation* (monodepth2), ICCV 2019.
[2] Wen et al., *FoundationStereo: Zero-Shot Stereo Matching*, 2024.
[3] Teed & Deng, *RAFT: Recurrent All-Pairs Field Transforms*, ECCV 2020.
[4] Lipson et al., *RAFT-Stereo*, 3DV 2021.
[5] Ranftl et al., *Vision Transformers for Dense Prediction* (DPT), ICCV 2021.
[6] Siméoni et al., *DINOv3*, 2024.
