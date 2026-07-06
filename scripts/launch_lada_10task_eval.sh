#!/bin/bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh && conda activate raoxuan
export HF_ENDPOINT=https://hf-mirror.com
cd /home/raoxuan/projects/project_clip_continual_learning
mkdir -p experiments/lada_10task_eval
rm -rf experiments/lada_10task_eval/combo*_async experiments/lada_10task_eval/combo*_eval experiments/lada_10task_eval/combo*_train.log experiments/lada_10task_eval/combo*.json

common="--dataset_sequence aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397 --num_shots 16 --batch_size 128 --eval_batch_size 256 --num_workers 6 --optimizer adamw --weight_decay 3e-5 --lora_type lora_nsp --lora_rank 4 --use_dora false --init_mode lora_nsp --train_budget_mode uniform --cd_weight 2.0 --fd_weight 1.0 --aux_weight 0.0 --cd_divergence kl_forward --cd_temperature 4.0 --alpha_sensitivity --output_dir experiments/lada_10task_eval"

CUDA_VISIBLE_DEVICES=0 nohup python main_incremental.py $common --iterations 800 --scheduler cosine_with_warmup --lr 1e-4 --null_init_mode none --experiment_name combo1_balanced_10task_train > experiments/lada_10task_eval/combo1_balanced_10task_train.log 2>&1 < /dev/null &
PID1=$!

CUDA_VISIBLE_DEVICES=2 nohup python main_incremental.py $common --iterations 1600 --scheduler linear --lr 3e-4 --null_init_mode none --experiment_name combo2_lastmax_10task_train > experiments/lada_10task_eval/combo2_lastmax_10task_train.log 2>&1 < /dev/null &
PID2=$!

CUDA_VISIBLE_DEVICES=3 nohup python main_incremental.py $common --iterations 800 --scheduler cosine --lr 1e-4 --null_init_mode history_init_runtime --experiment_name combo3_transfermax_10task_train > experiments/lada_10task_eval/combo3_transfermax_10task_train.log 2>&1 < /dev/null &
PID3=$!

CUDA_VISIBLE_DEVICES=4 nohup python main_incremental.py $common --iterations 800 --scheduler cosine_with_warmup --lr 1e-4 --null_init_mode history_init_only --experiment_name combo4_initonly_10task_train > experiments/lada_10task_eval/combo4_initonly_10task_train.log 2>&1 < /dev/null &
PID4=$!

echo "Training PIDs: $PID1 $PID2 $PID3 $PID4"
nohup bash scripts/monitor_lada_10task_eval.sh $PID1 $PID2 $PID3 $PID4 > experiments/lada_10task_eval/monitor.log 2>&1 < /dev/null &
echo "Monitor PID: $!"
sleep 3
ps aux | grep -E "main_incremental.py.*lada_10task_eval|monitor_lada_10task_eval.sh" | grep -v grep
