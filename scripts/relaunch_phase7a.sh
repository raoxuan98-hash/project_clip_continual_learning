#!/bin/bash
set -e
cd /home/raoxuan/projects/project_clip_continual_learning
mkdir -p artifacts/logs/10task_reablation

pkill -f "soft_nsp_w0.01" 2>/dev/null || true
pkill -f "soft_nsp_w0.02" 2>/dev/null || true
pkill -f "soft_nsp_w0.05" 2>/dev/null || true
sleep 3

CMD="/home/raoxuan/ENTER/envs/raoxuan/bin/python -u main_incremental.py \
  --root /data1/open_datasets/X-TAIL \
  --dataset_sequence aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397 \
  --num_shots 16 --batch_size 32 --iterations 800 \
  --scheduler cosine_with_warmup --lora_type lora_nsp --use_dora false --lora_rank 4 \
  --lora_target_modules q_proj,k_proj,v_proj,out_proj,fc1,fc2 \
  --projection_param_mode full --null_init_mode none \
  --optimizer adamw --lr 1e-4 --weight_decay 3e-5 \
  --cd_weight 2.0 --fd_weight 1.0 --aux_weight 0.0 \
  --cd_divergence kl_forward --cd_temperature 4.0 \
  --text_tuning_schedule always \
  --num_centers 4 --rgda_train_iter 200 --rgda_train_lr 0.01 --rgda_fit_source gmm_sample \
  --use_soft_projection --seed 43 --eval_batch_size 128 \
  --output_dir experiments/10task_reablation"

CUDA_VISIBLE_DEVICES=1 nohup $CMD --nsp_weight 0.01 --experiment_name soft_nsp_w0.01 \
  > artifacts/logs/10task_reablation/soft_nsp_w0.01.log 2>&1 &
echo "w0.01 on GPU1 PID=$!"

CUDA_VISIBLE_DEVICES=2 nohup $CMD --nsp_weight 0.02 --experiment_name soft_nsp_w0.02 \
  > artifacts/logs/10task_reablation/soft_nsp_w0.02.log 2>&1 &
echo "w0.02 on GPU2 PID=$!"

CUDA_VISIBLE_DEVICES=4 nohup $CMD --nsp_weight 0.05 --experiment_name soft_nsp_w0.05 \
  > artifacts/logs/10task_reablation/soft_nsp_w0.05.log 2>&1 &
echo "w0.05 on GPU4 PID=$!"

sleep 5
echo "GPU status:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
