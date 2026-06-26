#!/bin/bash
cd /home/raoxuan/projects/project_clip_continual_learning
mkdir -p experiments/null_basis_ablation/iter800
PY=/home/raoxuan/miniconda3/bin/python
DATASETS="aircraft caltech101 dtd eurosat"
ROOT=/data1/open_datasets/X-TAIL
OUT=experiments/null_basis_ablation/iter800
COMMON="--dataset_sequence $DATASETS --root $ROOT --num_shots 16 --iterations 800 --batch_size 64 --lora_type lora_nsp --classifier_feature_transform test --seed 42"

echo "=== Launching 4 experiments ==="

$PY main_incremental.py $COMMON --projection_param_mode full --null_init_mode none --fd_weight 0 --cd_weight 0 --aux_weight 0 --gpu 0 --output_dir $OUT --experiment_name current_nsp_vision_only > $OUT/current_nsp_vision_only.log 2>&1 &
PID1=$!
echo "Exp1 (current_nsp vision_only) PID=$PID1 GPU=0"

$PY main_incremental.py $COMMON --projection_param_mode full --null_init_mode none --fd_weight 1.0 --cd_weight 1.0 --aux_weight 0 --gpu 1 --output_dir $OUT --experiment_name current_nsp_full_recipe > $OUT/current_nsp_full_recipe.log 2>&1 &
PID2=$!
echo "Exp2 (current_nsp full_recipe) PID=$PID2 GPU=1"

$PY main_incremental.py $COMMON --projection_param_mode full --null_init_mode history_init_only --fd_weight 0 --cd_weight 0 --aux_weight 0 --gpu 2 --output_dir $OUT --experiment_name hist_null_init_vision_only > $OUT/hist_null_init_vision_only.log 2>&1 &
PID3=$!
echo "Exp3 (hist_null_init vision_only) PID=$PID3 GPU=2"

$PY main_incremental.py $COMMON --projection_param_mode full --null_init_mode history_init_only --fd_weight 1.0 --cd_weight 1.0 --aux_weight 0 --gpu 3 --output_dir $OUT --experiment_name hist_null_init_full_recipe > $OUT/hist_null_init_full_recipe.log 2>&1 &
PID4=$!
echo "Exp4 (hist_null_init full_recipe) PID=$PID4 GPU=3"

echo "All launched. Waiting for all to finish..."
wait $PID1 $PID2 $PID3 $PID4
echo "=== All 4 experiments complete ==="
