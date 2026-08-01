#!/bin/bash
# 轻量监控 full-shot + B0 完成状态，每 5 分钟写一次状态文件
set -e
PROJECT=/home/raoxuan/projects/project_clip_continual_learning
STATUS_FILE=$PROJECT/artifacts/launch_logs/fullshot_b0_monitor.log
DIR=$PROJECT/experiments/paper_formal/PA_main

mkdir -p $(dirname $STATUS_FILE)
echo "[monitor] started at $(date +%Y-%m-%d\ %H:%M:%S) CST" > $STATUS_FILE

while true; do
  NOW=$(date +%Y-%m-%d\ %H:%M:%S)
  FS_LORANF=$(ls $DIR/PA__fs__loranf_i600__seed*_ens_results.json 2>/dev/null | wc -l)
  FS_LORAONLY=$(ls $DIR/PA__fs__loraonly_i600__seed*_ens_results.json 2>/dev/null | wc -l)
  B0=$(ls $DIR/PA__b0_*_ens_results.json 2>/dev/null | wc -l)
  echo "[$NOW] fs_loranf=$FS_LORANF/2 fs_loraonly=$FS_LORAONLY/2 b0=$B0" >> $STATUS_FILE
  if [ "$FS_LORANF" -eq 2 ] && [ "$FS_LORAONLY" -eq 2 ] && [ "$B0" -ge 1 ]; then
    echo "[$NOW] all required JSON present, running aggregation..." >> $STATUS_FILE
    /home/raoxuan/ENTER/envs/raoxuan/bin/python $PROJECT/scripts/aggregate_pa_results.py --dir $DIR --per-dataset >> $STATUS_FILE 2>&1 || true
    echo "[$NOW] aggregation done." >> $STATUS_FILE
    break
  fi
  sleep 300
done
