# PA 新协议全量重跑进度 + 补增常规 LADA 16-shot

**日期**: 2026-08-01
**会话概况**: 检查服务器自治进度，发现主表已跑完但缺少常规 LADA 16-shot 复现；补增 LADA 波次到 supervisor 并提交到 average-push 分支。

---

## 1. 关键决策 / 变更

- **常规 LADA 16-shot 必须复现**: 原 `pa_wave_launcher.sh` 队列只包含 LoRA-NF / LoRA-only / 消融 / 敏感性，缺少常规 CLIP + LADA 分类器的 16-shot 复现。已新增 `scripts/pa_lada_launcher.sh`，并在 `scripts/pa_supervisor.sh` 中插入 LADA 波次（在 SigLIP2 之前）。
- **修正 MAIN_TOTAL**: 敏感性实际有 15 个（不是 14 个），`MAIN_TOTAL` 由 26 改为 27。
- **LADA 配置**: 与 SigLIP2/full-shot 中的 LADA 条目一致，使用常规 CLIP 模型、`--lora_type lora_nsp --init_mode lora_nsp --null_init_mode none`，**不传递 `--disable_lada`** 以启用 LADA 分类器评估。

## 2. 当前进度

- **16-shot 主表 + 消融 + 敏感性**: 27/27 完成，`experiments/paper_formal/PA_main/` 下五类 JSON（zs/rgda/ens/retrieval/main）齐全。
- **SigLIP2 LoRA-NF seed42/43**: 运行中，正在做最后阶段的图像特征/协方差收集；seed43 领先 seed42 约 50%。
- **LADA 16-shot**: 已加入队列，supervisor 进入 LADA 阶段，待 SigLIP2 LoRA-NF 释放 GPU 后自动启动。
- **SigLIP2 LADA / full-shot / B0**: 排队中，按 supervisor 四阶段顺序执行。

## 3. 注意事项 / 待确认

- 本地 Mac 工作目录的 `.git` 目录为空/损坏，`git pull` 失败；本次变更直接在服务器上提交到 `average-push` 分支并 push。
- SigLIP2 LADA 之前在 14:42 因 `is_siglip2_model_name` 缺失崩溃；当前 `src/models/backbone_utils.py` 已包含该函数，待 supervisor 调度时观察是否再崩溃。
- full-shot 仍按用户要求排在最后。

## 4. 相关文件

- `scripts/pa_supervisor.sh`: 新增 LADA 阶段，修正 MAIN_TOTAL=27。
- `scripts/pa_lada_launcher.sh`: 新增常规 LADA 16-shot 接力器。
- `experiments/paper_formal/PA_main/`: 当前 27 个 run 的 JSON 输出。
- `artifacts/launch_logs/supervisor.log`: supervisor 运行日志。
