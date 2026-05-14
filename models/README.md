# my_baseline/models

Plan C — 提前抠出来的 monodepth2 公共模块,给后面的 GRU iterative refiner 用。

## 文件

| 文件 | 内容 |
|---|---|
| [photometric.py](photometric.py) | `BackprojectDepth`, `Project3D`, `InverseWarp`, `SSIM`, `transformation_from_parameters`, `photometric_reconstruction_loss`, `edge_aware_smoothness_loss`, `compute_depth_errors` |
| [posenet.py](posenet.py) | `PoseNet` — ResNet18 编码器 (`num_input_images=2`) + monodepth2 PoseDecoder,输出 (axisangle, translation),已乘 0.01 缩放 |
| [__init__.py](__init__.py) | 统一对外接口 |

## 典型用法(给后面的 refiner)

```python
from my_baseline.models import (
    PoseNet, InverseWarp, transformation_from_parameters,
    photometric_reconstruction_loss, edge_aware_smoothness_loss,
)

posenet = PoseNet(num_layers=18, pretrained=True).cuda()
warp    = InverseWarp(H, W).cuda()

# I_tgt, I_src: (B, 3, H, W);depth_tgt: (B, 1, H, W) 来自 DINOv3+DPT (frozen) 或 refiner 当前 d_k
ax, tr = posenet(torch.cat([I_tgt, I_src], dim=1))
T      = transformation_from_parameters(ax[:, 0], tr[:, 0], invert=False)
reproj, _ = warp(I_src, depth_tgt, T, K, inv_K)
photo  = photometric_reconstruction_loss(I_tgt, reproj).mean()
smooth = edge_aware_smoothness_loss(1.0 / depth_tgt, I_tgt)
loss   = photo + 1e-3 * smooth
```

## 设计要点

1. **零外部依赖** — 没 import monodepth2,200 行自己重写,license 没问题。
2. **DDP-friendly** — `BackprojectDepth` 用 `register_buffer` 而非 `nn.Parameter`,batch 大小动态。
3. **可微** — 所有几何操作走 `torch.matmul / F.grid_sample`,depth & pose 都可以拿梯度。
4. **跟 monodepth2 公式一致** — 0.85·SSIM + 0.15·L1,edge-aware smoothness,axis-angle 0.01 缩放。

## 来源 attribution

部分代码思路来自 [monodepth2](https://github.com/nianticlabs/monodepth2) (Niantic, ICCV 2019)。
重写而非复制源码,但行为和公式保持一致以便实验复现。

## Sanity 测试

```bash
cd dinov3_baseline
python -m my_baseline.scripts.test_models
```

输出形状 / 反向传播 / metric 三件套验证。
