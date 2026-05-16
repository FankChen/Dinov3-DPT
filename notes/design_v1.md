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

---

## Appendix B — DINOv3 ViT-7B + SYNTHMIX DPT Depther 拆解(refiner 接口选址)

> 目的:在 7B 权重到手前,先把 baseline 内部结构搞清,确定 refiner 怎么接最合适。
> 全部信息从 dinov3 源码直接读出,**不靠猜**。

### B.1 关键参数(7B SYNTHMIX,出自 `dinov3/hub/depthers.py`)

| 项 | 值 | 出处 |
|---|---|---|
| backbone | `dinov3_vit7b16`,40 blocks,patch=16,embed_dim=4096 | depthers.py + arXiv Tab 2 |
| 取层 index | **[9, 19, 29, 39]**(均匀 4 段,0-based) | `_get_out_layers` |
| 取出后每层通道 | 4096(ViT-7B 原始) | backbone embed_dim |
| Reassemble 后通道 | **[2048, 2048, 2048, 2048]** | `_get_post_process_channels` |
| Fusion 隐藏通道 | 512 | `head_kwargs.channels` |
| DPT 输出 bins | 256 | `n_output_channels` |
| Depth 范围(权重训练时) | (0.001, 100.0) m | `_get_depth_range(SYNTHMIX)` |
| Depth 范围(NYU eval 时) | **覆盖为 (0.001, 10.0)** m | `config-nyu-synthmix-dpt-inference.yaml` |
| Bins strategy / norm | linear / linear | DecoderConfig 默认 |
| use_batchnorm | **True** | `_DPT_HEAD_CONFIG_DICT` |
| use_backbone_norm | True(取层后过 LayerNorm) | 同上 |
| use_cls_token | False → readout_type="ignore" | 同上 |
| Inference img_size | 768 | inference yaml |
| Inference 其他 | use_tta=True, align_least_squares=True | inference yaml |

⚠️ **scale-invariant eval**:`align_least_squares=True` 意味着 paper Tab 12 的数字是和 GT 做 per-image scale align 后才报的(AbsRel / δ1)。**refiner 训完之后必须用同一套 align 才能跟 paper 比**,否则数字没意义。

### B.2 Tensor shape 链(假设 768×1024 输入)

```
INPUT  RGB image
       (B, 3, 768, 1024)
         │ ImageNet normalize + CenterPadding(到 patch_size 倍数,768/1024 已是 16 倍,无 pad)
         ▼
─── ENCODER: DinoVisionTransformerWrapper (frozen ViT-7B) ─────────────
       backbone.get_intermediate_layers(x, n=[9,19,29,39], reshape=True, return_class_token=True, norm=True)
       4 个层级,每个返回 (patch_feat, cls_token):
            L9  : ( (B, 4096, 48, 64),  (B, 4096) )       # 浅
            L19 : ( (B, 4096, 48, 64),  (B, 4096) )
            L29 : ( (B, 4096, 48, 64),  (B, 4096) )
            L39 : ( (B, 4096, 48, 64),  (B, 4096) )       # 深
         │ ★ 4 层 features 都是 48×64,占显存最大的位置在这里
         ▼
─── DECODER: DPTHead ────────────────────────────────────────────────
  STEP 1  ReassembleBlocks
       L9  ── BN(4096) → 1×1 Conv(4096→2048) → ConvTr stride=4 → (B, 2048, 192, 256)   ← 浅, 放大 4×
       L19 ── BN(4096) → 1×1 Conv(4096→2048) → ConvTr stride=2 → (B, 2048,  96, 128)   ← 放大 2×
       L29 ── BN(4096) → 1×1 Conv(4096→2048) → Identity        → (B, 2048,  48,  64)   ← 不变
       L39 ── BN(4096) → 1×1 Conv(4096→2048) → Conv  stride=2  → (B, 2048,  24,  32)   ← 深, 缩小 2×
       (cls_token 走 readout="ignore",**不参与**;7B 这里就忽略了)

  STEP 2  per-stage 3×3 Conv,4096 路通道全部 → 512(=head_kwargs.channels)
       → (B, 512, 192, 256), (B, 512, 96, 128), (B, 512, 48, 64), (B, 512, 24, 32)

  STEP 3  FeatureFusionBlock × 4,自顶向下,每步 ×2 上采(U-Net 风)
       fusion[0]  L39 单输入 → (B, 512, 48, 64)
       fusion[1]  + L29 skip → (B, 512, 96, 128)
       fusion[2]  + L19 skip → (B, 512, 192, 256)
       fusion[3]  + L9  skip → (B, 512, 384, 512)         ← ★ 这里是 1/2 输入分辨率
         │ project (Conv3×3 512→512)
         ▼
  STEP 4  UpConvHead   (B, 512, 384, 512)
              Conv3×3 512→256
              ↑2× bilinear
              Conv3×3 256→32, ReLU
              Conv1×1 32→256                              ← n_output_channels=256(bin 个数)
       → (B, 256, 768, 1024)                              ← ★ 跟原图分辨率一致

─── FeaturesToDepth (训练时 loss 在 bin logits 之前;推理时这一步出 depth) ────
       linear bins[256]: torch.linspace(0.001, 10.0, 256)
       logits → relu+eps → 按 bin 求 softmax-like 加权和
       → (B, 1, 768, 1024)  metric depth in meters
```

### B.3 显存粗算(7B,frozen backbone forward,bf16)

仅 forward(只算 4 层 features 不算 transformer 中间 KV):
- Backbone 权重 ~13.4 GB(bf16)
- 4 层 features 都是 (B, 4096, 48, 64) ≈ B × 50 MB,**B=2 时 ~400 MB**
- DPT decoder 权重 ~200 MB(估算)
- DPT 中间 activation 最大那张 (B, 512, 384, 512) ≈ B × 400 MB
- **总计 B=2 推理 ≈ 15 GB**(只 forward,frozen)

训练时(refiner 反传到 DPT 输出,backbone 仍 frozen):
- 需要保留 DPT 中间 activation → 再 ×2~3
- KITTI photometric 一次要 forward 2-3 帧(target + source(s))→ 再 ×2~3
- **粗估 30-45 GB**,H200 单卡(80GB)稳,mig 切片要 40GB+ 才行

### B.4 Refiner 4 个候选接入点

| 编号 | 接哪 | tensor shape (B,C,H,W) | 优点 | 缺点 | 工程风险 |
|---|---|---|---|---|---|
| **P1** | DPT 最终深度图后 | (B, 1, 768, 1024) | 最简单,1 通道,跟 monodepth2 完全对齐 | 信息只剩 1 维深度,refiner 没"context"可用 | 低 |
| **P2** | UpConvHead 之前(`forward_features` 返回) | (B, 512, 384, 512) | 有 512 维 context 给 GRU 当 state;还有 2× 提升的余地 | 自己要重新接 UpConvHead;1/2 分辨率 | 中 |
| **P3** | 4 个 fusion outputs 各一个 | 4 张多尺度 | RAFT/RAFT-Stereo 风格 hierarchical refine | 4 路 refiner,设计复杂 | 高 |
| **P4** | 4 路 reassemble 输出(L9/19/29/39 重排后) | 4 张 (B, 2048, ...) | 最深的 context,可直接换 decoder | **退化成"重训 head"**,不再叫 refiner | 不推荐 |

**初判建议:P1 + P2 二选一**:
- 想跟 paper 强对比 → **P1**(refiner 只调整深度,baseline DPT 输出可直接报 Tab 12 数字,refiner 后报"+refiner")
- 想 contribution 强一点 → **P2**(用 512 维 feature 当 GRU 隐藏状态,refiner 真正在 feature space 工作)

**这个选择不急,等师兄回 design_v1 时一起拍**。

### B.5 已确认的"等 7B 期间不需要权重就能写"的代码

| 模块 | 状态 | 备注 |
|---|---|---|
| `my_baseline/models/photometric.py` | ✅ Plan C 已完成 | 输入 (B,1,H,W) 深度图 + 相邻帧,输出 photometric loss |
| `my_baseline/models/posenet.py` | ✅ Plan C 已完成 | 标准 monodepth2 PoseCNN |
| KITTI dataloader (Eigen split) | ❌ 待写 | 数据已在 `/home/izi2sgh/MYDATA/kitti/`,直接抄 monodepth2 |
| Refiner 主体(ConvGRU based) | ⏳ 等 P1/P2 决定再写 | 但接口可以先按 P1 写好 |
| 训练循环 + smoke test(临时用 ViT-L) | ❌ 待写 | 验证 photometric loss 反传通即可,不报数字 |

### B.6 Eval protocol 必须先定死(避免事后口径变)

paper Tab 12 (7B SYNTHMIX) 的协议是:
- **zero-shot** — 模型从未见过 NYU/KITTI 训练图
- **scale-invariant** — 每张图都跟 GT 做最小二乘 scale align
- **TTA on** — 水平翻转 + 多尺度推理

我们的 refiner 协议**至少给出两套数字**才有说服力:
1. **复现 baseline**:7B SYNTHMIX 直接在 NYU eval(同 paper 协议),对齐 Tab 12 数字
2. **加 refiner 后**:refiner 在 KITTI 视频上 self-sup 训,**在 NYU 上 eval 时仍 zero-shot**(KITTI 不算训过 NYU);如果在 KITTI 上 eval 则是"in-domain self-sup"
   - 必须给师兄讲清楚:**这两种是不同的 claim**,不能混着报

---

*Appendix B 写于 2026-05-16,源码读取自 `dinov3/dinov3/eval/depth/` 与 `dinov3/dinov3/hub/depthers.py`。7B 权重到手前所有架构判断都已闭环,无需再猜。*

