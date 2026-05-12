# DINOv3 源码精读笔记(带中文注释)

> 这三个文件是从官方 DINOv3 repo (`facebookresearch/dinov3`) 的对应位置 copy 过来,
> 在原内容上**只加了中文注释**(以 `# 【中文导读】` 开头的代码块),代码逻辑零修改。
>
> 如果想看 unmodified 上游源码:`git clone https://github.com/facebookresearch/dinov3`

| 文件 | 上游路径 | 说明 |
|------|---------|------|
| `run.py` | `dinov3/eval/depth/run.py` | depth baseline 总入口 |
| `encoder.py` | `dinov3/eval/depth/models/encoder.py` | DINOv3 backbone 包装,FOUR_EVEN_INTERVALS 取 [4,11,17,23] 层 |
| `dpt_head.py` | `dinov3/eval/depth/models/dpt_head.py` | DPT decoder(reassemble → fusion×4 → upconv) |

详细解读见 [../../../note/note4dinov3.md](../../../note/note4dinov3.md)。
