# DINOv3 + DPT Depth Baseline (NYU)

> 本项目目标:在 DINOv3 官方 repo 基础上 **增量** 搭一个 DPT depth baseline,
> 用于后续 "层间 photometric 反馈" 实验改造的对照组。
>
> 本目录里所有文件都是新增的,**不修改 `../dinov3/` 官方源码**(只在源码里加了中文注释)。
> 后续如果需要 PR 给官方,可以从本目录提取干净 patch。

---

## 1. 目录结构

```
my_baseline/
├── README.md                       本文档
├── envs/
│   └── env_dinov3.yaml             conda 环境定义
├── scripts/
│   ├── env_setup.sh                一键建 conda 环境
│   ├── prepare_nyu_split.py        校验 NYU 数据 + 建 symlink
│   ├── download_weights.py         从 HuggingFace 拉 DINOv3 backbone (备用)
│   ├── run_train.py                训练 launcher,monkey-patch dataset registry
│   ├── train_nyu.bsub              LSF 提交脚本(batch_h200,full GPU)
│   └── train_nyu_mig.bsub          LSF 提交脚本(batch_h200_mig,1/7 切片,排队快)
├── datasets/
│   └── nyu_labeled.py              cluster 上 795/654 NYU labeled subset 的 dataset 类
├── configs/
│   └── nyu_dpt_vitl.yaml           DPT + ViT-L 训练配置
├── notes/
│   ├── note4dinov3.md              DINOv3 源码精读笔记 + 集群权重位置
│   └── dinov3_annotations/         上游三个关键源文件的"加中文注释版"
├── checkpoints/                    backbone 权重(.gitignore,集群 symlink)
├── data/                           数据集 symlink(.gitignore)
└── outputs/                        训练产物 ckpt / log / csv(.gitignore)
```

---

## 2. 实验设定

| 项 | 值 | 说明 |
|---|---|---|
| Backbone | `dinov3_vitl16` (~300M) | DINOv3 ViT-L,patch=16,24 层,冻结 |
| Head | DPT | 不是 yaml 默认的 linear |
| 数据 | NYU labeled subset (795 / 654) | cluster 上现有,**不是** BTS 完整 NYU(~24k) |
| 评测 | NYU Eigen crop, max_depth=10m | 与 BTS / DA2 / DA3 一致 |
| 集群 | LSF, queue=`batch_h200`, 1× H200 | bf16 训练 |

**注意**: NYU 训练集只有 795 张(BTS 完整版的 1/30),数字会比 paper 差很多。这是 **预期** 行为,我们这阶段的目的是 **跑通 baseline + 拿到对照数字**,不是冲 SOTA。

---

## 3. 运行步骤

### 3.1 建 conda 环境(只做一次)
```bash
bash scripts/env_setup.sh
```

### 3.2 预下载 backbone 权重(只做一次)
```bash
# 先 huggingface-cli login 接受 license,然后:
python scripts/download_weights.py
```

### 3.3 准备 split 文件(只做一次)
```bash
python scripts/prepare_nyu_split.py
```

### 3.4 提交训练 job
```bash
bsub < scripts/train_nyu.bsub
```

---

## 4. Results

### v0 baseline (38400 iter, bs=2, single H200 mig slice, ~1h47min)

| Metric | Ours (NYU 795 训) | DINOv3 paper (BTS 24k 训) |
|--------|------------------:|--------------------------:|
| AbsRel ↓ | **0.0908** | ~0.040 |
| δ1 (a1) ↑ | **0.9329** | ~0.97 |
| RMSE ↓ | **0.3362** | ~0.16 |

> 数字差 ~2x 的主要原因:训练数据缺一个数量级(795 vs 24k)。
> Pipeline / loss / decoder 都按官方 NYU config,可作 photometric 改造的对照。

---

## 5. Changelog

- 2026-05-07: 项目初始化,目录骨架建立
- 2026-05-12: 跑通 v0 baseline,得到首个 NYU 数字
