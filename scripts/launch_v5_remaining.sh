#!/bin/bash
# ============================================================
# V5 剩余实验启动脚本 (Phase 11-15)
# 基于 chat-history/2026-07-08-v5-plan.md
#
# 运行实验：
#   Phase 11: iterations=1600
#   Phase 12: warmup_ratio ∈ {0.0, 0.05, 0.20}
#   Phase 13: fd_weight ∈ {2.0, 4.0}
#   Phase 14: weight_decay ∈ {1e-5, 1e-4}
#   Phase 15: rank=8 + nsp_eps=0.20 交互
#
# 使用 3 张 GPU: 0, 1, 3 (GPU 2 被其他项目占用)
# ============================================================
set -euo pipefail

PROJECT=/home/raoxuan/projects/project_clip_continual_learning
PYTHON=/home/raoxuan/ENTER/envs/raoxuan/bin/python
LOGDIR="$PROJECT/artifacts/logs/10task_reablation"
OUTDIR="$PROJECT/experiments/10task_reablation"
mkdir -p "$LOGDIR" "$OUTDIR"

DATASET_SEQUENCE="aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397"

# 基线配置：所有会被实际读取的参数显式传入
# 注意：text_tuning_schedule=always 时，text_schedule_switch_task / text_lr_scale_after_task
# 不会被代码使用，因此不传入，避免“传了但无用”。
BASE_ARGS=(
    --root /data1/open_datasets/X-TAIL
    --dataset_sequence $DATASET_SEQUENCE
    --num_shots 16
    --batch_size 32
    --scheduler cosine_with_warmup
    --warmup_ratio 0.1
    --eta_min 0.0
    --iterations 800
    --lora_type lora_nsp
    --use_dora false
    --lora_rank 4
    --lora_target_modules q_proj,k_proj,v_proj,out_proj,fc1,fc2
    --projection_param_mode full
    --null_init_mode none
    --nsp_eps 0.20
    # --use_soft_projection 不传入：默认 False，对应 hard NSP
    --optimizer adamw
    --lr 1e-4
    --weight_decay 3e-5
    --cd_weight 2.0
    --fd_weight 1.0
    --aux_weight 0.0
    --cd_divergence kl_forward
    --cd_temperature 4.0
    --text_tuning_schedule always
    --reference_batch_size 32
    --num_centers 4
    --rgda_train_iter 200
    --rgda_train_lr 0.01
    --rgda_fit_source gmm_sample
    --ensemble_normalize maxshift
    --alpha_sweep_batch_size 512
    --seed 43
    --eval_batch_size 128
    --num_workers 2
    --output_dir "$OUTDIR"
)

launch_one() {
    local gpu=$1
    local name=$2
    shift 2
    local log="$LOGDIR/${name}.log"

    # 如果结果已存在则跳过
    if [ -s "$OUTDIR/${name}_zs_results.json" ] && [ -s "$OUTDIR/${name}_ens_results.json" ]; then
        echo "[skip] $name (results exist)"
        return 0
    fi

    echo "[$(date '+%H:%M:%S')] Launching $name on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu nohup "$PYTHON" -u "$PROJECT/main_incremental.py" \
        "${BASE_ARGS[@]}" \
        --experiment_name "$name" \
        "$@" \
        > "$log" 2>&1 &
}

echo "=== V5 Remaining Phases (3 GPUs: 0,1,3) ==="
echo "Baseline: LoRA+NSP hard eps=0.20 text=always lr=1e-4 bs=32 cd=2.0 fd=1.0 temp=4.0 mc4ft200"

# ---- Round 1 ----
echo ""
echo "--- Round 1/4 ---"
launch_one 0 iter1600          --iterations 1600
launch_one 1 warmup_ratio_0    --warmup_ratio 0.0
launch_one 3 warmup_ratio_5    --warmup_ratio 0.05
wait

# ---- Round 2 ----
echo ""
echo "--- Round 2/4 ---"
launch_one 0 warmup_ratio_20   --warmup_ratio 0.20
launch_one 1 fd_weight_2       --fd_weight 2.0
launch_one 3 fd_weight_4       --fd_weight 4.0
wait

# ---- Round 3 ----
echo ""
echo "--- Round 3/4 ---"
launch_one 0 weight_decay_1e-5 --weight_decay 1e-5
launch_one 1 weight_decay_1e-4 --weight_decay 1e-4
launch_one 3 rank8_eps0.20     --lora_rank 8 --nsp_eps 0.20
wait

echo ""
echo "=== All V5 remaining experiments submitted ==="
echo "Logs:   $LOGDIR"
echo "Results: $OUTDIR"
echo "Monitor: watch -n 30 'tail -n 5 $LOGDIR/*.log'"
