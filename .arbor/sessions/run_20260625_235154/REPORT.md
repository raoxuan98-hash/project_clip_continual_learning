# Research Report: Implement and execute the experimental plan in docs/lora_null_basis_ablation_plan.md. This requir...

_Generated 2026-06-26 15:04_   _Exit: `ok`_

## Instruction

> Implement and execute the experimental plan in docs/lora_null_basis_ablation_plan.md. This requires: 1) Implement 5 LoRA parameterization variants in src/models/lora_sgp.py -- current_nsp baseline, basis_fixed_tail, basis_core_tail, hist_null_init_only, hist_null_init_runtime -- with new CLI args --projection_param_mode --basis_rank --null_init_mode. 2) Run Round 1 experiments on aircraft caltech101 dtd with 16-shot 200-iter. 3) Analyze and run Round 2 on top 2 + baseline across 4 datasets with seeds 42 43 44. 4) Produce summary tables in experiments/null_basis_ablation/. Data at /data1/open_datasets/X-TAIL.

**Model**: `deepseek-v4-pro`
**Duration**: 15h12m15s

## Event Summary

| Metric | Value |
|--------|-------|
| Cycles completed | 11 |
| Ideas proposed | 12 |
| Ideas completed | 4 |
| Ideas pruned | 0 |
| Ideas merged | 1 |
| Sub-agent runs | 11 |
| LLM errors | 0 |
| Eval failures | 3 |
| Wall clock | 15h 12m 15s |

## Run Stats

_Token scope_: `research_agent_process_llm_only; excludes LLM calls made by external Bash/RunTraining/eval subprocesses`

| Metric | Value |
|--------|-------|
| Total LLM calls | 509 |
| Total agents spawned | 12 |
| Total input tokens | 23,072,712 |
| Total uncached input tokens | 690,376 |
| Total cache-read input tokens | 22,382,336 |
| Total output tokens | 253,768 |
| Coordinator turns | 51 |

## Results

**Baseline → Final**: `55.5400` → `55.5400` (+0.00%)

**Test set**: baseline=`55.54` final=`54.72`

## Exploration

_13 nodes total, 9 scored, 1 merged_

### Merged Ideas

- **1.1** (`55.6`): Mechanism: Modify SGPBaseLoRA.forward() and SGPBaseDoRA.forward() to support 3 projection_param_modes via branching on self.projection_param_mode: "full" (default BAP), "fixed_basis" (B @ U^T using first k columns of B), "core_basis" (B @ C @ U^T with new nn.Parameter C). Add basis_U buffer, basis_ready flag, basis_rank attr. Modify merge_lora_weights() to match forward delta. Add set_basis_U() method. Extend LoRACLIPVisionTransformer.__init__ to accept new args and propagate to modules. Add initialize_history_null() for LoRA-Null-style init. Add CLI args --projection_param_mode, --basis_rank, --basis_window, --null_init_mode to main_incremental.py and wire through clip.py. Handle DoRA weight_directions/magnitude in null_init.

### Top Ideas by Score

- **3.2** `56.1000` _done_: Mechanism: Round 2 validation of hist_null_init_only (best Round 1 variant) on 4 datasets with 3 seeds. Test whether the initialization-only approach generalizes to an additional dataset (eurosat) and across random seeds.
- **2.4** `55.7500` _done_: Mechanism: Run hist_null_init_only — projection_param_mode=full, null_init_mode=history_init_only. Initializes BA via LoRA-Null-style history-null projection but trains WITHOUT runtime P (P=I throughout).
- **1.1** `55.6000` _merged_: Mechanism: Modify SGPBaseLoRA.forward() and SGPBaseDoRA.forward() to support 3 projection_param_modes via branching on self.projection_param_mode: "full" (default BAP), "fixed_basis" (B @ U^T using first k columns of B), "core_basis" (B @ C @ U^T with new nn.Parameter C). Add basis_U buffer, basis_ready flag, basis_rank attr. Modify merge_lora_weights() to match forward delta. Add set_basis_U() method. Extend LoRACLIPVisionTransformer.__init__ to accept new args and propagate to modules. Add initialize_history_null() for LoRA-Null-style init. Add CLI args --projection_param_mode, --basis_rank, --basis_window, --null_init_mode to main_incremental.py and wire through clip.py. Handle DoRA weight_directions/magnitude in null_init.
- **2.5** `55.5800` _done_: Mechanism: Run hist_null_init_runtime — projection_param_mode=full, null_init_mode=history_init_runtime. Initializes BA via LoRA-Null-style history-null projection AND continues using runtime NSP (P_h active during training).
- **2.1** `55.5400` _done_: Mechanism: Run baseline current_nsp (projection_param_mode=full, null_init_mode=none) on B_dev.
- **2.2** `55.3400` _done_: Mechanism: Run basis_fixed_tail — projection_param_mode=fixed_basis, basis_rank=4, basis_window=tail, null_init_mode=none. Uses ΔW = B U_h^T with fixed tail basis from history covariance.
- **2.3** `55.2500` _done_: Mechanism: Run basis_core_tail — projection_param_mode=core_basis, basis_rank=16, basis_window=tail, null_init_mode=none. Uses ΔW = B C U_h^T where C ∈ R^{r×k} learns a low-rank combination within the k=16 tail directions.
- **3.3** `55.2300` _done_: Mechanism: Round 2 validation of basis_fixed_tail (best parameterization variant) on 4 datasets with 3 seeds. Test whether the fixed basis approach generalizes and whether the Transfer/Last tradeoff persists with eurosat.
- **3.1** `54.7200` _done_: Mechanism: Round 2 validation of current_nsp baseline on 4 datasets (aircraft caltech101 dtd eurosat) with 3 seeds (42, 43, 44), 16-shot, 200 iter. Run all 3 seeds sequentially in one experiment and report mean±std.

## Artifacts

- Coordinator final report: `/home/raoxuan/projects/project_clip_continual_learning/.arbor/sessions/run_20260625_235154/COORDINATOR_FINAL_REPORT.txt`
- Idea tree (JSON): `/home/raoxuan/projects/project_clip_continual_learning/.arbor/sessions/run_20260625_235154/.coordinator/idea_tree.json`
- Idea tree (Markdown): `/home/raoxuan/projects/project_clip_continual_learning/.arbor/sessions/run_20260625_235154/.coordinator/idea_tree.md`
- Run stats: `/home/raoxuan/projects/project_clip_continual_learning/.arbor/sessions/run_20260625_235154/run_stats.json`
- Event log: `/home/raoxuan/projects/project_clip_continual_learning/.arbor/sessions/run_20260625_235154/events.jsonl`
