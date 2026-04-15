#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

# =========================================================
# 基础配置
# =========================================================
dataset='CHN6-CUG'
method='aprmatch'
exp='apr'
split='20%'

config=configs/${dataset}.yaml
labeled_id_path=splits/$dataset/$split/labeled.txt
unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
save_path=exp/$dataset/$method/$exp/$split

mkdir -p $save_path

# =========================================================
# 运行参数
# =========================================================
GPUS=${1:-1}
PORT=${2:-29500}

# APR
APR_RATIO_FACTOR=${3:-0.2}
APR_PROB=${4:-0.5}
APR_PATCH_SIZE=${5:-64}

# Boundary schedule
BDY_START_EPOCH=${6:-10}
BDY_RAMP_EPOCHS=${7:-10}
GATE_SCALE_FINAL=${8:-0.5}

# =========================================================
# 启动训练
# =========================================================
echo "============================================"
echo "实验: $exp | 数据集: $dataset | Split: $split"
echo "GPU: $GPUS | 端口: $PORT"
echo "APR: ratio_factor=$APR_RATIO_FACTOR, prob=$APR_PROB, patch_size=$APR_PATCH_SIZE"
echo "BDY: start=$BDY_START_EPOCH, ramp=$BDY_RAMP_EPOCHS, gate_scale=$GATE_SCALE_FINAL"
echo "保存路径: $save_path"
echo "============================================"

torchrun \
    --nproc_per_node=$GPUS \
    --master_addr=localhost \
    --master_port=$PORT \
    aprmatch.py \
    --config=$config \
    --labeled-id-path $labeled_id_path \
    --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path \
    --port $PORT \
    --apr-ratio-factor $APR_RATIO_FACTOR \
    --apr-prob $APR_PROB \
    --apr-patch-size $APR_PATCH_SIZE \
    --bdy-start-epoch $BDY_START_EPOCH \
    --bdy-ramp-epochs $BDY_RAMP_EPOCHS \
    --gate-scale-final $GATE_SCALE_FINAL \
    2>&1 | tee $save_path/$now.log
