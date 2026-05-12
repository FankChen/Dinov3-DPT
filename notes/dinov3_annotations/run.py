# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

# =============================================================================
# 【中文导读】run.py —— 整个 depth baseline 的总入口
# -----------------------------------------------------------------------------
# 本文件做三件事:
#   1) main()              : 命令行入口,解析参数 → 进 benchmark_launcher
#   2) benchmark_launcher(): 构造 depther(backbone + head),决定走"训练"还是"加载已训权重"
#   3) eval_depther_with_model(): 在测试集上跑评估,把指标写 csv
#
# 关键分支(在 benchmark_launcher 里):
#   - 如果 load_from == "dinov3_vit7b16_dd"  → 走 hub 加载官方 SYNTHMIX 预训练 depther(纯推理)
#   - 否则 + load_from 有值                  → 加载自己训过的 ckpt 评估
#   - 否则                                   → 进入 train_model_with_backbone 从头训 head
# =============================================================================

import json
import logging
import os
import sys
from typing import Any, Dict

import torch
from omegaconf import OmegaConf  # OmegaConf:用 yaml 做配置管理(类似 hydra)

import dinov3.distributed as distributed
# 找输出目录里最新的 checkpoint(断点续训 / 续评估用)
from dinov3.eval.depth.checkpoint_utils import find_latest_checkpoint
# DepthConfig:整个 depth pipeline 的"配置 schema",见 config.py
from dinov3.eval.depth.config import DepthConfig
# 评估器:给定 depther + 测试集 → 算 AbsRel / δ1 / RMSE 等
from dinov3.eval.depth.eval import evaluate_depther_with_config
# 工厂函数:根据 config 把 (backbone, head) 拼成完整的 depther 模型
from dinov3.eval.depth.models import make_depther_from_config
# 训练器:冻结 backbone,只训 head(linear 或 dpt),返回训完的 depther
from dinov3.eval.depth.train import train_model_with_backbone

# 通用工具(跨任务共享,不只是 depth):cli 解析、结果写盘
from dinov3.eval.helpers import args_dict_to_dataclass, cli_parser, write_results
# 加载 DINOv3 backbone(从 hub 名或 ckpt 路径)
from dinov3.eval.setup import load_model_and_context
# 分布式 / logging / 输出目录的初始化上下文
from dinov3.run.init import job_context
# 官方在 SYNTHMIX 上预训练好的 depther(7B backbone + DPT head),可直接 hub.load
from dinov3.hub.depthers import _get_depther_config, dinov3_vit7b16_dd

RESULTS_FILENAME = "results-depth.csv"             # 最终结果文件名
MAIN_METRICS = [".*_abs_rel", ".*_a1", ".*_rmse"]  # 写 csv 时主要 highlight 的 3 个指标(正则匹配)


logger = logging.getLogger("dinov3")


def _add_dataset_prefix_to_results(results_dict: Dict[str, float], dataset_name: str):
    # 把 {"abs_rel": 0.08, ...} 改成 {"nyu_abs_rel": 0.08, ...},
    # 多个测试集汇总时不会撞 key
    final_dict = {dataset_name + "_" + k: v for k, v in results_dict.items()}
    return final_dict


def eval_depther_with_model(*, depther: torch.nn.Module, config: DepthConfig):
    """在测试集上评估一个 depther(已经拿到模型对象,只负责跑测 + 汇总)。"""
    # 如果没指定具体 ckpt,就到 output_dir 里找最新的(自动续评估)
    if config.load_from is None:
        config.load_from = find_latest_checkpoint(config.output_dir)

    logger.info(f"Using config: \n {OmegaConf.to_yaml(config)}")

    # 真正跑 inference + 算指标的地方,详见 eval.py。
    # reduce_results=False:这里先不对多 GPU 的结果做平均,后面手动 nanmean。
    results_dict, _, _ = evaluate_depther_with_config(
        config=config,
        depther=depther,
        device=distributed.get_rank(),
        reduce_results=False,
    )

    # 测试集名字(yaml 里写的形如 "nyu:test"),取冒号前那段做子目录名
    test_config_name = config.datasets.test.split(":", 1)[0]
    test_save_dir = os.path.join(config.output_dir, test_config_name)

    # 只在 rank0 写 results.json(避免多 GPU 重复写文件)
    if distributed.is_main_process():
        if not os.path.exists(test_save_dir):
            os.makedirs(test_save_dir)
        with open(os.path.join(test_save_dir, "results.json"), "w") as f:
            json.dump(results_dict, f, indent=4)

    # 把每个指标在所有图上做 nanmean(GT 全 mask 时会得到 NaN,要忽略)
    for metric, values in results_dict.items():
        results_dict[metric] = float(torch.Tensor(values).nanmean())

    # 打印一行漂亮的 summary 到日志
    summary = " \n====== Summary ======\n"
    summary += (
        f"{test_config_name:<10} "
        + " ".join([f"{metric}: {value:.3f}" for metric, value in results_dict.items()])
        + "\n"
    )
    results_dict = _add_dataset_prefix_to_results(results_dict, test_config_name)
    summary += "====================="
    logger.info(summary)
    return results_dict


def benchmark_launcher(eval_args: dict[str, Any]) -> dict[str, Any]:
    """
    【核心调度函数】
    Initialization of distributed and logging are preconditions for this method
    (调用前必须先 init 好 distributed/logging,见 main() 里的 job_context)。

    干两件事:
      A. 把 cmdline + yaml 合并成一个完整的 DepthConfig
      B. 根据 config.load_from 决定走 3 条路径之一,产出 depther,然后送去 eval
    """

    # ---------- A. 装配配置 ----------
    if "config" in eval_args:
        # 走法 1:命令行里通过 config=xxx.yaml 指定了 base config
        base_config_path = eval_args.pop("config")
        output_dir = eval_args["output_dir"]
        base_config = OmegaConf.load(base_config_path)        # 读 yaml
        structured_config = OmegaConf.structured(DepthConfig) # 拿 schema(类型校验)
        # 三层 merge 优先级:default schema  <  yaml  <  cmdline
        depth_config: DepthConfig = OmegaConf.to_object(
            OmegaConf.merge(
                structured_config,
                base_config,
                OmegaConf.create(eval_args),
            )
        )
    else:
        # 走法 2:没给 yaml,纯靠 cmdline kwargs 拼一个 DepthConfig
        depth_config, output_dir = args_dict_to_dataclass(
            eval_args=eval_args, config_dataclass=DepthConfig, save_config=False
        )

    # 把最终 merge 出的 config 落盘,方便复现实验
    OmegaConf.save(config=depth_config, f=os.path.join(output_dir, "depth_config.yaml"))

    # autocast 精度(fp16 / bf16 / fp32),如果没指定就跟着 backbone 走
    config_autocast_dtype = depth_config.model_dtype.autocast_dtype if depth_config.model_dtype is not None else None

    # ---------- B. 三条路径产出 depther ----------

    if depth_config.load_from == "dinov3_vit7b16_dd":
        # 【路径 1】纯推理:用官方在 SYNTHMIX 上训好的 7B+DPT depther
        #   - 整个 depther(backbone + DPT head 全套权重)从 hub 一次性加载
        #   - 不训练,只评估
        with torch.device("cuda" if torch.cuda.is_available() else "cpu"):
            autocast_dtype = config_autocast_dtype or torch.float32
            # 用 hub 自带的 depther 配置(head 结构必须对齐它训练时的设置)覆盖用户配置
            depther_config = _get_depther_config("dinov3_vit7b16")
            depth_config.decoder_head = OmegaConf.to_object(
                OmegaConf.merge(
                    depth_config.decoder_head,
                    depther_config,
                )
            )
            # 这一行下载 + 实例化 backbone+DPT head,载入预训练权重
            depther = dinov3_vit7b16_dd(
                pretrained=True,
                autocast_dtype=autocast_dtype,
            )
    else:
        # 【路径 2 / 3】共同前置:先把 DINOv3 backbone 加载好
        with torch.device("cuda" if torch.cuda.is_available() else "cpu"):
            assert depth_config.model is not None
            # backbone 来自 cfg.model(可以是 hub 名,也可以是本地 ckpt 路径)
            model, model_context = load_model_and_context(depth_config.model, output_dir=output_dir)
            autocast_dtype = config_autocast_dtype or model_context["autocast_dtype"]

        if depth_config.load_from:
            # 【路径 2】backbone + 自己之前训好的 head ckpt → 直接评估
            depther = make_depther_from_config(
                backbone=model,
                config=depth_config.decoder_head,
                checkpoint_path=depth_config.load_from,
                autocast_dtype=autocast_dtype,
            )
            logger.info(f"Depth config:\n {OmegaConf.to_yaml(depth_config)}")
        else:
            # 【路径 3】load_from 为空 → 从头训 head(backbone 通常冻结)
            # 训完返回带训好 head 的 depther,继续往下做 eval
            depther = train_model_with_backbone(depth_config, model, autocast_dtype)

    # ---------- C. 不论哪条路径,最后都跑 eval + 写 csv ----------
    results_dict = eval_depther_with_model(depther=depther, config=depth_config)
    write_results(results_dict, output_dir, RESULTS_FILENAME)
    return results_dict


def main(argv=None):
    """命令行入口。典型用法见 README:
        python dinov3/eval/depth/run.py \
            config=dinov3/eval/depth/configs/config-nyu.yaml \
            datasets.root=<...> output_dir=<...>
    """
    if argv is None:
        argv = sys.argv[1:]
    # cli_parser:把 "key=value" 形式的 cmdline 解析成 dict
    eval_args = cli_parser(argv)
    # job_context:统一处理 distributed init、log 配置、output_dir 创建等
    with job_context(output_dir=eval_args["output_dir"]):
        benchmark_launcher(eval_args=eval_args)
    return 0


if __name__ == "__main__":
    main()
