# DINOv3 Depth Baseline 学习笔记

> 项目目标:跑通 DINOv3 + DPT 的 monocular depth baseline,为后续 "层间 photometric 反馈" 改造做准备。
> 仓库位置:[liren/dinov3_baseline/dinov3/](../dinov3_baseline/dinov3/)
> 起始日期:2026-05-07

---

## 0. 路线图

```
读源码理解 baseline ──► 跑通官方 baseline (NYU训 DPT head) ──► 改造 forward 加 photometric 反馈
        ▲                          ▲                                  ▲
        当前位置                    待开始                             师兄方向
```

读源码顺序(按对师兄方向的重要度):
1. ✅ `eval/depth/run.py` —— 总入口
2. ✅ `eval/depth/models/encoder.py` —— DINOv3 backbone 包装器 ⭐
3. ✅ `eval/depth/models/dpt_head.py` —— DPT decoder ⭐⭐
4. ⏳ `eval/depth/train.py` —— 训练主循环
5. ⏳ `eval/depth/loss.py` / `metrics.py` —— 监督信号

---

## 1. 概念基础(读所有代码都要用)

### 1.1 backbone / head / depther 三段式

```
image → [backbone (DINOv3 ViT, 冻结)] → 多层 token → [head (DPT, 训练)] → depth map
                                          ↑
                                          整个 backbone+head 合起来叫 depther
```

- **backbone**:大,通用,自监督预训练好,**冻结不训**
- **head**:小,任务专属,**只训这部分**
- 一个 backbone 可接不同 head 做不同任务(分类 / 分割 / 深度)

### 1.2 张量形状语言

| 形状 | 含义 |
|---|---|
| `(B, 3, H, W)` | 一批 RGB 图。B=batch,3=RGB |
| `(B, N, D)` | ViT 内部 token 序列。N = 1(CLS) + h·w(patch) |
| `(B, D, h, w)` | reshape 回 feature map。h = H / patch_size |
| `(B, 1, H, W)` 或 `(B, H, W)` | 深度图 |

**ViT-L:patch_size=16, D=1024, n_blocks=24**

### 1.3 patch / token / CLS

ViT 把图切成 16×16 像素方块,每块 → 1 个向量 token。
- 512×512 图 → (512/16)² = 1024 个 patch token + 1 个 CLS token = 1025 个 token
- **CLS token**:全局语义,1 个向量代表整张图。深度任务**通常不用**(只用 patch token)

### 1.4 ckpt(checkpoint)

训练过程中保存的权重快照,通常是 `.pth` 或 `.pt` 文件。
- 整套 depther ckpt = backbone 权重 + head 权重(打包)
- 只 head ckpt = 自己训的部分,几十 MB,backbone 单独从 hub 加载

### 1.5 冻结 = `requires_grad_(False)`

每个参数有梯度开关:
- `True` → 训练时被更新
- `False` → 反向传播跳过

`encoder.py` 里 `self.backbone.requires_grad_(False)` 让 24 层 ViT 全部不学。

---

## 2. 已读文件笔记

### 2.1 `run.py` —— 总入口

**它不做计算,只调度**。核心是 `benchmark_launcher()`,根据 `config.load_from` 走 3 条路径:

| `load_from` 值 | 行为 | 用途 |
|---|---|---|
| `"dinov3_vit7b16_dd"` | hub 一次性加载完整官方 depther(7B+DPT) | 5 分钟跑通验证环境 |
| `/path/to/xxx.pth` | 加载 backbone + 加载自己训好的 head | 训完后 eval |
| 不设置 | 加载 backbone + **从头训 head** | ⭐ 师兄要的 baseline |

**配置优先级**:python dataclass 默认值 < yaml < 命令行 `key=value`

### 2.2 `models/encoder.py` —— DINOv3 backbone 包装器 ⭐

**作用**:告诉 DINOv3 "前向时把第几层的 token 留给我",并整理成 DPT 想要的格式。

**核心数据流**(`forward()`):

```
image (B, 3, H, W)
    │
    ▼ patch_size_adapter (pad 到 16 倍数)
    │
    ▼ backbone.get_intermediate_layers(n=[4, 11, 17, 23])  ← 一次跑完 24 层,取 4 层
    │
    ▼
[
  (patch_feat_L4,  cls_L4 ),    # patch_feat: (B, 1024, h, w),  cls: (B, 1024)
  (patch_feat_L11, cls_L11),
  (patch_feat_L17, cls_L17),
  (patch_feat_L23, cls_L23),
]
    │
    ▼ 给 DPT head
```

**3 种取层策略**(`BackboneLayersSet`):
- `LAST` —— 只取最后一层(分类、简单分割用)
- `FOUR_LAST` —— 最后 4 层(都偏深层语义)
- `FOUR_EVEN_INTERVALS` —— **DPT 默认**:浅+中+深均匀分布,既懂细节又懂语义

**ViT-L 的兼容性 hack** ⚠️:`FOUR_EVEN_INTERVALS` 应该是 `[5, 11, 17, 23]`,但官方早期算错成 `[4, 11, 17, 23]`,旧 ckpt 都按错的训了,所以代码里**故意保留这个错误**。

**师兄方向的入口**:
- 现在 forward 用 `get_intermediate_layers` "一次跑完 24 层"
- 将来要改成"分段跑":第 0→4 层 → 取 token → 出小 depth → warp 算 photometric error → 把 error 喂回去 → 4→11 层 → ...
- **要重写的核心就是 `forward()` 这几行**

---

### 2.3 `models/dpt_head.py` —— DPT decoder ⭐⭐

#### 一句话
**把 ViT 给的 4 个一样大的 token map,做成金字塔,自顶向下融合上采,最后翻译成深度图。**

#### 用做菜比喻理解全流程

ViT 给了 4 锅一样大(都是 32×32)的"原料":浅层(L4)细节多,深层(L23)语义强。
DPT 像厨子一样把它们炖成一张深度图,**4 步**:

```
STEP 1  Reassemble       人为造金字塔(浅放大、深缩小)
   L4  →  128×128  (放大 4×)  保留细节
   L11 →   64×64
   L17 →   32×32
   L23 →   16×16  (缩小 2×)   语义指挥官
   同时:把 CLS token 信息融合进 patch token,通道数也调成不同

STEP 2  Convs            把所有层级的通道数对齐到 256

STEP 3  FeatureFusion    自顶向下逐层融合 + 上采 (U-Net 套路)
   out = 锅4(16×16)
   out 上采 2× → 加上 锅3(32×32) → conv  → 32×32
   out 上采 2× → 加上 锅2(64×64) → conv  → 64×64
   out 上采 2× → 加上 锅1(128×128) → conv → 128×128
   out 上采 2× → conv                       → 256×256

STEP 4  UpConvHead       conv + 上采 + 1×1conv → (B, 1, 512, 512) 深度图
```

#### 4 个新概念(我之前没讲过)

| 词 | 大白话 |
|---|---|
| **分辨率 / feature map 大小** | feature map 的 H×W。越大 = 信息越细 |
| **上采样 upsample** | 把 feature map 变大(32→64),用插值或 ConvTranspose |
| **下采样 downsample** | 把 feature map 变小(32→16),用 stride=2 的 Conv |
| **通道数 channels** | (B, C, H, W) 里的 C。每个像素携带的信息量 |

ConvTranspose2d:**反向卷积**,会让 feature map 变大(stride=4 就放大 4 倍)。

#### 为什么"浅放大、深缩小"?

- 浅层 token 本身就懂细节(边缘、纹理) → 放大让这些细节占更多空间
- 深层 token 本身就懂全局语义(物体、场景) → 缩小到瓶颈处当"指挥"
- 多尺度组合 > 单尺度,这是 U-Net 论证过的经验

#### 5 个类的角色

| 类 | 行 | 角色 |
|---|---|---|
| `ConvModule` | 35-228 | 工具:封装 conv+norm+act。当标准 PyTorch Conv 用即可 |
| `Interpolate` | 230 | 工具:把上采样函数包成 nn.Module |
| `ReassembleBlocks` | 278 | ⭐ STEP 1:造金字塔 + 处理 CLS |
| `FeatureFusionBlock` | 404 | ⭐ STEP 3 单元:融合 + 上采 2× |
| `UpConvHead` | 243 | ⭐ STEP 4:特征 → 深度图 |
| `DPTHead` | 455 | ⭐⭐ 主类,串起所有组件 |

#### `forward_features` 主逻辑(521 行,本文件最关键的 10 行)

```python
def forward_features(self, inputs):                  # 4 个 (patch_feat, cls_token)
    x = self.reassemble_blocks(x)                    # STEP 1: 4 个不同分辨率
    x = [self.convs[i](f) for i, f in enumerate(x)]  # STEP 2: 通道对齐到 256

    # STEP 3: 从最深开始(x[-1] = L23,最小),自顶向下融合
    out = self.fusion_blocks[0](x[-1])
    for i in range(1, len(self.fusion_blocks)):
        out = self.fusion_blocks[i](out, x[-(i+1)])  # 每次加上更浅的 skip
    out = self.project(out)
    return out
```

#### 师兄方向的"插入点"

```
encoder.py:
   ViT 24 层 → 取 [4, 11, 17, 23] 4 层 token
                              │
                              ▼
dpt_head.py:
   ReassembleBlocks                    ← 入口 1:这里插
        │
        ▼
   ConvModule × 4                      ← 入口 2:在通道对齐后加分支算 photo error
        │
        ▼
   FeatureFusionBlock × 4              ← 入口 3:每个 fusion 后出 mid-depth + 反馈
        │
        ▼
   UpConvHead → depth map
```

**两种方案**:
- **方案 A(改 encoder)**:在 ViT 第 4/11/17 层后立刻出小 depth + photometric error,信号反馈进 backbone 后续层。**激进**。
- **方案 B(改 dpt_head)**:每个 FeatureFusion 后接小 head 出 mid-depth,error 当 channel concat 进下一个 fusion。**保守**。

师兄说"反馈给 transformer 下一层" = **方案 A**。

---

## 3. 三条路径对应的命令(README 摘录)

### 路径 1:推理官方 depther(快速验证)
```bash
PYTHONPATH=. python dinov3/eval/depth/run.py \
    config=dinov3/eval/depth/configs/config-nyu-synthmix-dpt-inference.yaml \
    datasets.root=<PATH/TO/DATASET> \
    load_from=dinov3_vit7b16_dd \
    output_dir=<PATH/TO/OUTPUT/DIR>
```

### 路径 3:从头训 DPT head(师兄要的 baseline)
```bash
PYTHONPATH=. python dinov3/eval/depth/run.py \
    model.dino_hub=dinov3_vit7b16 \
    config=dinov3/eval/depth/configs/config-nyu.yaml \
    datasets.root=<PATH/TO/DATASET> \
    --output-dir <PATH/TO/OUTPUT/DIR>
```
训完产出:
- `depth_config.yaml` —— 训练时的完整 config
- `model_final.pth` —— 训好的 head 权重
- `results-depth.csv` —— 测试集指标

---

## 4. 工程小知识

| 名词 | 一句话解释 |
|---|---|
| `omegaconf` / yaml | 分层配置系统,`key=value` 命令行可以临时覆盖 yaml |
| `distributed.is_main_process()` | 多 GPU 时只让 0 号卡写文件,单卡可忽略 |
| `autocast_dtype` (bf16/fp32) | 自动混合精度。H200 默认 bf16,显存减半速度翻倍 |
| `logger.info()` | 比 `print` 强的输出,能写日志文件 |
| `nn.Identity()` | "什么都不做"的网络层,占位用 |

---

## 5. 待回答的问题(下次见师兄前梳理)

- [ ] 监督信号是 GT depth 还是纯 photometric(自监督)?
- [ ] photometric warp 需要两帧,KITTI 用 stereo pair,那 NYU 怎么办?
- [ ] photometric error 怎么注入 ViT 中间层?(concat token / cross-attn / FiLM)
- [ ] backbone 用多大?(7B 太重,可能 ViT-L 起步)

---

## 6. 个人 TODO

- [x] clone DINOv3 repo
- [x] 读 `run.py` + 注释
- [x] 读 `encoder.py` + 注释
- [x] 读 `dpt_head.py` + 注释
- [ ] 读 `train.py`
- [ ] 跑通路径 1(推理官方 depther,验证环境)
- [ ] 跑通路径 3(NYU 自训 DPT head)
- [ ] 设计第一个改造实验

---

## 7. 集群上的 DINOv3 权重位置(以后好找)

> 不需要走 HF gated repo。所有 .pth 都是 official native 格式
> (`cls_token`, `patch_embed.proj.weight`, `blocks.X....`),可直接 `torch.load`。

**ViT-L (1.2 GB,本 baseline 用)**

- `/fs/scratch/rb-bd-dlp-rng-dl01-cr-tfx/special/tfx-901/luy1syv/dinov3/pretrained/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`
- `/fs/scratch/rb-bd-dlp-rng-dl01-cr-tfx/common-machinery/tfx-103/gnn-topo-ssl/gnn_ssl_data_copy/gnn-topo-ssl-attribute/dinov3_cache/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`

**其他尺寸(同一目录 `luy1syv/dinov3/pretrained/`)**

- `dinov3_vits16_pretrain_lvd1689m-08c60483.pth`     — Small
- `dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth` — Small+
- `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`     — Base
- `dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth` — Huge+ (在 `xiz1syv/baselines/dinov3/ckpts/`)

**我们项目里的 symlink**

```
my_baseline/checkpoints/dinov3_vitl16/model.pth
    → /fs/scratch/.../luy1syv/dinov3/pretrained/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

> 备注:HF repo `facebook/dinov3-vitl16-pretrain-lvd1689m` 申请被拒(Bosch
> corporate email 大概率被自动拒)。fbaipublicfiles 直链也 403。所以**集群
> 上的现成权重是唯一可行路径**。
