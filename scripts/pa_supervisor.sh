#!/bin/bash
# pa_supervisor.sh — PA 重跑计划服务器端自守护循环
# 每 10 分钟唤醒一次，依次推进四个波次（主队列 → SigLIP2 → full-shot → B0）。
# 注意：同 backbone 换 LADA 分类器的“LADA 复现”已按用户指示跳过，不再计入总数。
# 各接力器幂等、自愈（崩溃 run 无输出文件会被重启）。全局 ≤2 个实验进程。
# 启动: setsid nohup bash scripts/pa_supervisor.sh >/dev/null 2>&1 &
set -u
cd /home/raoxuan/projects/project_clip_continual_learning
OUT=experiments/paper_formal/PA_main
LOG=artifacts/launch_logs/supervisor.log
PIDFILE=artifacts/launch_logs/supervisor.pid
INTERVAL=600
MAIN_TOTAL=27      # 12 个 16-shot + 15 个敏感性
SIGLIP2_TOTAL=4
FS_TOTAL=4         # LoRA-NF seed42/43 + LoRA-only seed42/43（已去除 LADA 分类器 full-shot）

mkdir -p artifacts/launch_logs
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "[supervisor] already running (pid $(cat "$PIDFILE"))" >> "$LOG"
  exit 0
fi
echo $$ > "$PIDFILE"

echo "[supervisor] start $(date '+%F %T') pid=$$" >> "$LOG"
while true; do
  main_done=$(ls "$OUT"/PA__*_ens_results.json 2>/dev/null | grep -v 'siglip2\|PA__fs__' | wc -l)
  sig_done=$(ls "$OUT"/PA__siglip2_*_ens_results.json 2>/dev/null | wc -l)
  fs_done=$(ls "$OUT"/PA__fs__*_ens_results.json 2>/dev/null | wc -l)
  echo "[supervisor] $(date '+%F %T') main=$main_done/$MAIN_TOTAL siglip2=$sig_done/$SIGLIP2_TOTAL fs=$fs_done/$FS_TOTAL" >> "$LOG"

  if [ "$main_done" -lt "$MAIN_TOTAL" ]; then
    bash scripts/pa_wave_launcher.sh >> "$LOG" 2>&1
  elif [ "$sig_done" -lt "$SIGLIP2_TOTAL" ]; then
    bash scripts/pa_siglip2_launcher.sh >> "$LOG" 2>&1
  elif [ "$fs_done" -lt "$FS_TOTAL" ]; then
    bash scripts/pa_fullshot_launcher.sh >> "$LOG" 2>&1
  elif [ ! -f "$OUT/PA__b0_preserve_aspect_ens_results.json" ]; then
    if ! pgrep -f 'eval_frozen_zeroshot|eval_frozen_retrieval_baseline' > /dev/null; then
      gpu=$(bash -c 'for g in 0 1 2 3 4 5; do if ! nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $g 2>/dev/null | grep -q .; then echo $g; break; fi; done')
      echo "[supervisor] start B0 eval on gpu ${gpu:-0} $(date '+%F %T')" >> "$LOG"
      CUDA_VISIBLE_DEVICES=${gpu:-0} nohup bash scripts/pa_b0_zeroshot.sh >> "$LOG" 2>&1 &
    fi
  else
    echo "[supervisor] ALL DONE $(date '+%F %T')" >> "$LOG"
    rm -f "$PIDFILE"
    exit 0
  fi
  sleep "$INTERVAL"
done
