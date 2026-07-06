#!/bin/bash
# NSP 超参消融启动脚本
# 用法: bash scripts/launch_nsp_ablation.sh <wave>
#   wave 1: nsp_eps sweep {0.02, 0.08, 0.12, 0.20}
#   wave 2: nsp_weight sweep {0, 0.01, 0.05, 0.10}
#   wave 3: soft projection variants
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate raoxuan
PROJ=/home/raoxuan/projects/project_clip_continual_learning
cd $PROJ
mkdir -p logs experiments

WAVE=${1:-1}

# 公共参数（6-task 筛选，复用 Combo 1 基线配置）
COMMON="--root /data1/open_datasets/X-TAIL"
COMMON="$COMMON --dataset_sequence aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397"
COMMON="$COMMON --num_shots 16 --batch_size 128"
COMMON="$COMMON --lora_type lora_nsp --use_dora false --lora_rank 4"
COMMON="$COMMON --lora_target_modules q_proj,k_proj,v_proj,out_proj,fc1,fc2"
COMMON="$COMMON --projection_param_mode full --null_init_mode none"
COMMON="$COMMON --optimizer adamw --lr 1e-4 --weight_decay 3e-5"
COMMON="$COMMON --iterations 800 --scheduler cosine_with_warmup"
COMMON="$COMMON --cd_weight 2.0 --fd_weight 1.0 --aux_weight 0.0"
COMMON="$COMMON --cd_divergence kl_forward --cd_temperature 4.0"
COMMON="$COMMON --seed 43 --eval_batch_size 128"
COMMON="$COMMON --output_dir experiments"

export PYTHONUNBUFFERED=1

case $WAVE in
  1)
    echo "=== Wave 1: nsp_eps sweep ==="
    # GPU 0: eps=0.02 (tighter)
    CUDA_VISIBLE_DEVICES=0 python -u main_incremental.py $COMMON \
        --nsp_eps 0.02 --nsp_weight 0.02 \
        --experiment_name nsp_eps_002 \
        > logs/nsp_eps_002.log 2>&1 &

    # GPU 1: eps=0.08
    CUDA_VISIBLE_DEVICES=1 python -u main_incremental.py $COMMON \
        --nsp_eps 0.08 --nsp_weight 0.02 \
        --experiment_name nsp_eps_008 \
        > logs/nsp_eps_008.log 2>&1 &

    # GPU 2: eps=0.12
    CUDA_VISIBLE_DEVICES=2 python -u main_incremental.py $COMMON \
        --nsp_eps 0.12 --nsp_weight 0.02 \
        --experiment_name nsp_eps_012 \
        > logs/nsp_eps_012.log 2>&1 &

    # GPU 3: eps=0.20
    CUDA_VISIBLE_DEVICES=3 python -u main_incremental.py $COMMON \
        --nsp_eps 0.20 --nsp_weight 0.02 \
        --experiment_name nsp_eps_020 \
        > logs/nsp_eps_020.log 2>&1 &
    ;;

  2)
    echo "=== Wave 2: nsp_weight sweep ==="
    # GPU 0: weight=0 (pure hard projection)
    CUDA_VISIBLE_DEVICES=0 python -u main_incremental.py $COMMON \
        --nsp_eps 0.05 --nsp_weight 0 \
        --experiment_name nsp_w_000 \
        > logs/nsp_w_000.log 2>&1 &

    # GPU 1: weight=0.01
    CUDA_VISIBLE_DEVICES=1 python -u main_incremental.py $COMMON \
        --nsp_eps 0.05 --nsp_weight 0.01 \
        --experiment_name nsp_w_001 \
        > logs/nsp_w_001.log 2>&1 &

    # GPU 2: weight=0.05
    CUDA_VISIBLE_DEVICES=2 python -u main_incremental.py $COMMON \
        --nsp_eps 0.05 --nsp_weight 0.05 \
        --experiment_name nsp_w_005 \
        > logs/nsp_w_005.log 2>&1 &

    # GPU 3: weight=0.10
    CUDA_VISIBLE_DEVICES=3 python -u main_incremental.py $COMMON \
        --nsp_eps 0.05 --nsp_weight 0.10 \
        --experiment_name nsp_w_010 \
        > logs/nsp_w_010.log 2>&1 &
    ;;

  3)
    echo "=== Wave 3: soft projection ==="
    # GPU 0: soft projection with default weight params
    CUDA_VISIBLE_DEVICES=0 python -u main_incremental.py $COMMON \
        --nsp_eps 0.05 --nsp_weight 0.02 --use_soft_projection true \
        --experiment_name nsp_soft_default \
        > logs/nsp_soft_default.log 2>&1 &

    # GPU 1: soft projection + weight_temp=5.0
    CUDA_VISIBLE_DEVICES=1 python -u main_incremental.py $COMMON \
        --nsp_eps 0.05 --nsp_weight 0.02 --use_soft_projection true \
        --weight_temp 5.0 \
        --experiment_name nsp_soft_temp5 \
        > logs/nsp_soft_temp5.log 2>&1 &

    # GPU 2: soft + weight_temp=0.5
    CUDA_VISIBLE_DEVICES=2 python -u main_incremental.py $COMMON \
        --nsp_eps 0.05 --nsp_weight 0.02 --use_soft_projection true \
        --weight_temp 0.5 \
        --experiment_name nsp_soft_temp05 \
        > logs/nsp_soft_temp05.log 2>&1 &

    # GPU 3: soft + weight_kind=linear
    CUDA_VISIBLE_DEVICES=3 python -u main_incremental.py $COMMON \
        --nsp_eps 0.05 --nsp_weight 0.02 --use_soft_projection true \
        --weight_kind linear \
        --experiment_name nsp_soft_linear \
        > logs/nsp_soft_linear.log 2>&1 &
    ;;

  *)
    echo "Usage: bash launch_nsp_ablation.sh {1|2|3}"
    echo "  1: nsp_eps sweep {0.02, 0.08, 0.12, 0.20}"
    echo "  2: nsp_weight sweep {0, 0.01, 0.05, 0.10}"
    echo "  3: soft projection variants"
    exit 1
    ;;
esac

echo "Wave $WAVE launched. PIDs: $(jobs -p)"
wait
