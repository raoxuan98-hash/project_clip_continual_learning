#!/bin/bash
# run_lada_official_campaign.sh — 官方 LADA 基线战役启动器（Wave C）
#
# 用法: bash scripts/run_lada_official_campaign.sh <16shot|fullshot> <gpu> <seed>
# 协议与 per-dataset epochs 原样取自 LADA_official/scripts/run_TAIL_{16shot,fullshot}.sh
# （官方论文超参数）；num_shots=-1 即全量（官方 generate_fewshot_dataset 原生支持）。
# 结果: LADA_official/output/<NAME>/result.txt 由 result_process.py 生成，
#       拷贝到 project 的 experiments/paper_formal/WaveC_lada/ 供 Wave F 汇总。
set -u
SHOT=${1:?usage: run_lada_official_campaign.sh <16shot|fullshot> <gpu> <seed>}
GPU=${2:?}
SEED=${3:?}

LADA_DIR=/home/raoxuan/projects/LADA_official
CAMPAIGN_DIR=/home/raoxuan/projects/project_clip_continual_learning/experiments/paper_formal/WaveC_lada
PY=/home/raoxuan/ENTER/envs/raoxuan/bin/python
ROOT=/data1/open_datasets/X-TAIL
NAME=waveC_TAIL_${SHOT}_seed${SEED}

mkdir -p "$CAMPAIGN_DIR"
if [ -f "$CAMPAIGN_DIR/${NAME}_result.txt" ]; then
  echo "[skip] $NAME (result exists)"
  exit 0
fi

if [ "$SHOT" = "16shot" ]; then
  SHOTS=16
  PAIRS="aircraft:40 caltech101:10 dtd:30 eurosat:100 flowers:30 food101:5 mnist:200 oxford_pets:10 stanford_cars:30 sun397:10"
elif [ "$SHOT" = "fullshot" ]; then
  SHOTS=-1
  PAIRS="aircraft:60 caltech101:20 dtd:30 eurosat:20 flowers:40 food101:10 mnist:20 oxford_pets:10 stanford_cars:20 sun397:2"
else
  echo "unknown shot: $SHOT"; exit 1
fi

cd "$LADA_DIR"
FIRST=1
for pair in $PAIRS; do
  ds=${pair%%:*}; ep=${pair##*:}
  extra=""
  if [ $FIRST -eq 1 ]; then extra="continue_train_first True"; FIRST=0; fi
  echo "[run ] gpu=$GPU $NAME $ds epochs=$ep ($(date '+%F %T'))"
  CUDA_VISIBLE_DEVICES=$GPU $PY main.py -d TAIL -m clip_vit_b16 \
    num_shots $SHOTS dataset $ds num_epochs $ep $extra \
    root $ROOT seed $SEED output_dir $NAME \
    > run_logs/${NAME}_${ds}.log 2>&1
  rc=$?
  echo "[done] gpu=$GPU $NAME $ds rc=$rc ($(date '+%F %T'))"
  if [ $rc -ne 0 ]; then
    echo "[fail] $NAME aborted at $ds (see run_logs/${NAME}_${ds}.log)"
    exit $rc
  fi
done
$PY result_process.py -d TAIL --output_dir $NAME
cp output/$NAME/result.txt "$CAMPAIGN_DIR/${NAME}_result.txt"
echo "[done] $NAME all tasks finished ($(date '+%F %T'))"
