#!/usr/bin/env bash
# [#2 过拟合扫描] 一键：训练 V0/V1/V2 + 按"水位线px"逐epoch评分选模型。
# 前提：一块空闲 GPU(≥8G)。当前阻塞=本地/AutoDL 均被其他比赛训练占满(见报告§3.13)。
# 用法： bash run_sweep_overfit.sh          # 训练+评分全跑
#        bash run_sweep_overfit.sh score    # 只评分(训练已完成时)
set -e
PY=/home/jiale/sam3_test_venv/bin/python
HERE="$(cd "$(dirname "$0")" && pwd)"
CLOUD="$HERE/../cloud_server"
cd "$HERE"

VARIANTS="v0_baseline v1_lowcap v2_strongreg v3_encoder"

train_all() {
  for v in $VARIANTS; do
    echo "==================== 训练 $v ===================="
    $PY train_sam3_lora_native.py --config "configs/sweep_overfit/$v.yaml"
  done
}

# 按 px 给每个 epoch checkpoint 打分(val31+test16)。复用已验证的 eval_reading.py。
# 注：eval_reading 每次会重载 gauge 模型(sam3.pt 3.3G)→单次约1-2min,一次性选模型可接受。
score_all() {
  cd "$CLOUD"
  for v in $VARIANTS; do
    case $v in
      v0_baseline) outdir="SAM3_LoRa_Waterline_sweep_v0";;
      v1_lowcap)   outdir="SAM3_LoRa_Waterline_sweep_v1";;
      v2_strongreg)outdir="SAM3_LoRa_Waterline_sweep_v2";;
      v3_encoder)  outdir="SAM3_LoRa_Waterline_sweep_v3";;
    esac
    cfg="$HERE/configs/sweep_overfit/$v.yaml"
    echo "==================== 评分 $v (按水位线px) ===================="
    for w in "$HERE/$outdir"/epoch*_lora_weights.pt; do
      [ -e "$w" ] || { echo "  无 checkpoint($outdir),先训练"; break; }
      echo "---- $v $(basename "$w") ----"
      AIR_W="$w" AIR_CFG="$cfg" PYTHONPATH=. $PY eval_reading.py 2>&1 | grep -E "水位线像素误差|①纯水位线|端到端读数|未检出" || true
    done
  done
  echo "==================== 当前部署权重(对照基线) ===================="
  PYTHONPATH=. $PY eval_reading.py 2>&1 | grep -E "水位线像素误差|①纯水位线|端到端读数" || true
}

# SWA：px最优epoch邻域权重平均(小数据常更鲁棒)。需先确定每变体px最优epoch后手动指定,见报告§3.13。

case "${1:-all}" in
  score) score_all;;
  all)   train_all; score_all;;
  *) echo "用法: bash run_sweep_overfit.sh [all|score]"; exit 1;;
esac
echo "★ 选模型铁律：只有明确赢当前部署权重(重点看 max/离群 px,不只中位)才上线。"
