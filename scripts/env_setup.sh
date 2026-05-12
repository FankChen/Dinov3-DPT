#!/bin/bash
# 一键建 conda 环境。在 login 节点跑(计算节点没 internet)。
# 用法:bash scripts/env_setup.sh
set -e

# 加载集群 conda
module purge
module load conda/25.1.1

ENV_PREFIX=/home/izi2sgh/MYDATA/quanjie/liren/envs/dinov3_baseline
YAML=$(dirname "$0")/../envs/env_dinov3.yaml

if [ -d "$ENV_PREFIX" ]; then
  echo "[env_setup] env already exists at $ENV_PREFIX"
  echo "[env_setup] to recreate: rm -rf $ENV_PREFIX && bash $0"
  exit 0
fi

echo "[env_setup] creating env at $ENV_PREFIX from $YAML"
conda env create -p "$ENV_PREFIX" -f "$YAML"

echo "[env_setup] done. activate with:"
echo "    conda activate $ENV_PREFIX"
