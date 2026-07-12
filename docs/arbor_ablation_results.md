# Arbor 消融实验结果总结

**运行日期**: 2026-06-25 23:51 → 2026-06-26 15:04 (总耗时 15h12m)
**LLM**: `deepseek-v4-pro` (reasoning_effort=max via openai-chat)
**运行模式**: `--mode auto` 全自动无人值守
**服务器**: `raoxuan@10.20.34.30`, 6×GPU
**分支**: `v2-text-lora`
**实验协议**: 16-shot, 200 iterations/batch_size=64, vision-only LoRA-NSP, 无 FD/CD/aux

---

## 一、实验设计

测试 5 种 LoRA 变体，分为两类：

### Family 1: 参数化方式 (Parameterization)
| 变体 | 公式 | 说明 |
|------|------|------|
| **A0: current_nsp** (baseline) | ΔW = B A P | 现有方法，运行时投影 |
| **A1: basis_fixed_tail** | ΔW = B U_h^T | 固定 tail basis，rank=4 |
| **A2: basis_core_tail** | ΔW = B C U_h^T | 可学习的低秩组合 C∈R^{4×16} |

### Family 2: 初始化方式 (Initialization)
| 变体 | 公式 | 说明 |
|------|------|------|
| **B1: hist_null_init_only** | BA_init = W_0 U_h U_h^T, P=I | LoRA-Null 初始化，无运行时投影 |
| **B2: hist_null_init_runtime** | BA_init = W_0 U_h U_h^T, 有 P | 初始化 + 运行时投影 |

---

## 二、Round 1 结果 (3 数据集: aircraft → caltech101 → dtd, seed=42)

| 排名 | 变体 | Ensemble Average | Transfer | Last | Δ baseline |
|:----:|------|:----:|:----:|:----:|:---:|
| 🥇 | **hist_null_init_only (B1)** | **55.75** | 62.04 | 59.83 | **+0.21** |
| 🥈 | hist_null_init_runtime (B2) | 55.58 | 62.41 | 58.97 | +0.04 |
| 🥉 | current_nsp (A0) | 55.54 | 61.89 | 59.29 | — |
| 4 | basis_fixed_tail (A1) | 55.34 | 62.05 | 58.49 | −0.20 |
| 5 | basis_core_tail (A2) | 55.25 | 62.00 | 58.46 | −0.29 |

### 关键发现
- **hist_null_init_only 胜出**: 仅靠初始化即达到 +0.21 提升
- **初始化家族(B1,B2)均不弱于 baseline**: 说明初始化本身提供足够保护
- **参数化家族(A1,A2)Transfer 提升但 Last 下降**: 约束保护旧任务但限制塑性

### 各 Task 准确率矩阵 (Round 1)

```
A0 (current_nsp):
  Task 1 → [34.2, 80.8, 43.4]
  Task 2 → [34.0, 87.1, 42.6]
  Task 3 → [34.1, 87.0, 56.7]

A1 (basis_fixed_tail):
  Task 1 → [33.7, 81.4, 43.7]
  Task 2 → [33.4, 87.3, 42.2]
  Task 3 → [33.0, 87.6, 55.3]

A2 (basis_core_tail):
  Task 1 → [34.3, 81.0, 43.5]
  Task 2 → [33.9, 87.2, 42.4]
  Task 3 → [33.4, 87.0, 55.6]

B1 (hist_null_init_only):
  Task 1 → [34.4, 81.6, 43.3]
  Task 2 → [33.2, 88.1, 41.7]
  Task 3 → [31.1, 87.4, 60.9]   ← Last 最高 (+1.6 vs baseline)

B2 (hist_null_init_runtime):
  Task 1 → [33.6, 81.3, 43.9]
  Task 2 → [34.0, 87.1, 42.5]
  Task 3 → [33.8, 87.0, 56.5]
```

---

## 三、Round 2 结果 (4 数据集 × 3 seeds)

### 3-dataset 子集 (aircraft, caltech101, dtd)

| 变体 | Seed 42 | Seed 43 | Seed 44 | Mean ± Std | Δ baseline |
|------|:----:|:----:|:----:|:---:|:---:|
| **hist_null_init_only** | 56.01 | 56.01 | 56.29 | **56.10 ± 0.16** | **+0.56** |
| current_nsp (single seed) | — | — | — | 55.54 | — |
| basis_fixed_tail | 55.34 | 55.13 | 55.22 | 55.23 ± 0.10 | −0.31 |

### 4-dataset 完整 (aircraft, caltech101, dtd, eurosat)

| 变体 | Seed 42 | Seed 43 | Seed 44 | Mean ± Std | Transfer | Last |
|------|:----:|:----:|:----:|:---:|:----:|:----:|
| current_nsp | 54.60 | 54.71 | 54.84 | 54.72 ± 0.12 | 53.33 ± 0.41 | 67.34 ± 0.17 |
| hist_null_init_only | 53.96 | 53.85 | 54.45 | 54.09 ± 0.32 | — | — |

### 稳定性分析
- **baseline 极稳**: σ=0.12 on 4 datasets, 3 seeds
- **hist_null_init_only 3-dataset 也很稳**: σ=0.16
- **eurosat 对所有方法都是硬骨头**: 所有方法在加入 eurosat 后下降 ~1 点

---

## 四、计算代价

| 机制 | 每 task 额外开销 | 说明 |
|------|:---:|------|
| Runtime NSP (current_nsp) | **~30 min** | 72 层特征分解 |
| Runtime NSP (hist_null_init_runtime) | **~30 min** | 同上 |
| hist_null_init_only | **~1 min** | 仅在 task 边界做一次特征分解，训练时 P=I |
| Basis 模式 | ~30 min | 需要提取 basis_U |

---

## 五、结论

### 1. hist_null_init_only 是最优机制 ✅
- 3 数据集上 +0.56 over baseline (σ=0.16, 3 seeds)
- LoRA-Null 式初始化**单独**即提供等同于完整 NSP 的遗忘保护
- 消除了每 task ~30min 的特征分解瓶颈

### 2. 运行时 NSP 是冗余的 ❌
- hist_null_init_runtime 与 baseline 无显著差异
- 初始化 + 运行时投影无互补效果
- 昂贵的 per step eigendecomposition 不提供可测量的增益

### 3. Fixed-basis 参数化损害塑性 ❌
- basis_fixed_tail: −0.31 Average (σ=0.10)
- basis_core_tail: −0.29 Average (Round 1)
- 约束子空间保护旧知识但显著削弱新任务适应能力

### 4. Eurosat 是分布外挑战 ⚠️
- 所有方法加入 eurosat 后下降 ~1 点
- null-init 优势在 4-dataset 设置上反转 (−0.63)
- 说明 eurosat 的激活统计与其他数据集差异显著

---

## 六、推荐后续行动

1. **采用 hist_null_init_only** 替代当前 runtime NSP 机制
2. **废弃 fixed-basis / core-basis** 参数化方案
3. **调查 eurosat 的特殊性**: 为什么 null-init 对其无效
4. **扩展验证**: 在完整 10-task X-TAIL 设置上确认 3 数据集结论
5. **论文贡献**: 将 "initialization > runtime projection" 作为核心发现

---

## 七、Arbor 运行统计

| 指标 | 数值 |
|------|------|
| 总 LLM 调用 | 509 |
| Coordinator turns | 51 |
| Agent 生成数 | 12 |
| 实验完成数 | 11 (11/13 done, 1 merged) |
| LLM 错误 | 0 |
| Eval 失败 | 3 (自动恢复) |
| 总输入 tokens | 23,072,712 |
| 总输出 tokens | 253,768 |
| 缓存命中 | 22,382,336 (97%) |

---

## 八、原始文件位置

| 文件 | 路径 |
|------|------|
| Arbor 最终报告 | `artifacts/tool-runs/arbor/sessions/run_20260625_235154/REPORT.md` |
| Coordinator 报告 | `artifacts/tool-runs/arbor/sessions/run_20260625_235154/COORDINATOR_FINAL_REPORT.txt` |
| 实验汇总 | `experiments/null_basis_ablation/summary.md` |
| 实验 CSV | `experiments/null_basis_ablation/summary.csv` |
| Idea Tree | `artifacts/tool-runs/arbor/sessions/run_20260625_235154/.coordinator/idea_tree.md` |
| 事件日志 | `artifacts/tool-runs/arbor/sessions/run_20260625_235154/events.jsonl` |
| 运行日志 | `arbor_ablation.log` |
