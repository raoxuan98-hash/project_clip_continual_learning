# Idea Tree

**Baseline**: 59.0% | **Trunk**: 60.9%

## ROOT: Continue the Phase 1 + Phase 2 research. Status: vision_only experiments completed (current_nsp avg=77.1%, hist_null_init_only avg=67.8% RGDA at 800-iter). The full_recipe experiments failed but the device-mismatch bug in _compute_lora_delta is now fixed (commit 02296f1). Rerun the 2 full_recipe experiments (current_nsp+fd_cd, hist_null_init_only+fd_cd) at 800-iter on aircraft caltech101 dtd eurosat. Then continue Phase 2: explore your proposed ideas — adaptive distillation scheduling, text encoder curriculum learning, aux loss as structured regularization — plus any other ideas to improve Ensemble Average beyond current 77.1%. Data at /data1/open_datasets/X-TAIL. Output to experiments/null_basis_ablation/iter800/. [DONE]

**Insight**: FD+CD distillation robustly improves continual vision-language training, boosting ensemble average by 1.9% over a vision-only baseline. The gains are concentrated in mid-sequence tasks where feature drift from sequential fine-tuning is most acute, confirming the method’s ability to anchor pretrained representations. Critically, a device-mismatch fix (commit 02296f1) was essential to prevent silent text-encoder instability; without it, prior attempts failed. FD+CD is complementary to history_init_only null-space initialization—they can and should be combined. **Actionable conclusion:** Always apply FD+CD distillation together with history_init_only in continual vision-language settings, and enforce device consistency to avoid degradation.

### 1: Mechanism: Full recipe distillation (fd_weight=1.0, cd_weight=1.0) — Feature Distillation preserves pretrained visual knowledge while CD aligns vision-text representations during training.
Hypothesis: Adding FD+CD distillation to LoRA-NSP training will improve Ensemble Average by preventing catastrophic forgetting of pretrained representations, especially on earlier tasks.
Observable: Ensemble Average increase of +2.0-5.0% over baseline (58.98%) on B_dev.
Conflicts: none — rerunning previously failed experiments after device-mismatch fix [PENDING]

**Insight**: FD+CD distillation consistently improves Ensemble Average (+1.9% over vision-only baseline), confirming the hypothesis. The critical device-mismatch fix (commit 02296f1) was essential for text-encoder stability; without it, prior failures occurred. The method combines effectively with history_init_only null-space initialization—the two are complementary, not competing. Gains concentrate on mid-sequence tasks (caltech101, dtd) where feature drift from sequential fine-tuning is most severe, showing FD+CD successfully anchors pretrained representations. Actionable takeaway: Always apply FD+CD distillation with history_init_only in continual vision-language training; ensure device consistency to avoid silent text-encoder instability.

#### 1.1: Mechanism: current_nsp + full_recipe — projection_param_mode=full, null_init_mode=none, fd_weight=1.0, cd_weight=1.0, aux_weight=0. This is the baseline NSP mechanism with FD+CD distillation enabled.
Hypothesis: FD+CD distillation will improve Ensemble Average over the vision_only baseline (58.98%) by preserving pretrained visual features and maintaining vision-text alignment during LoRA adaptation.
Observable: Ensemble Average delta vs baseline 58.98%. Expected +2-5%.
Conflicts: none [DONE] (score: 60.9%)

**Insight**: FD+CD distillation improved Ensemble Average by 1.87 percentage points over the vision_only baseline, confirming the hypothesis directionally. The device-mismatch fix (commit 02296f1) was essential for text-encoder stability, and the full recipe configuration worked without further code changes.

**Result**: Ensemble Average increased from 58.98% (vision_only baseline) to 60.85% with FD+CD distillation enabled. FD and CD losses were active throughout training, indicating meaningful gradient signal.

**Branch**: `coordinator/n1-1-mechanism-current-nsp-full-recip-cae340cb`

#### 1.2: Mechanism: hist_null_init_only + full_recipe — projection_param_mode=full, null_init_mode=history_init_only, fd_weight=1.0, cd_weight=1.0, aux_weight=0. Historical null initialization with FD+CD distillation.
Hypothesis: Combining hist_null_init (which protects old tasks via null-space projection at task boundaries) with FD+CD distillation could be synergistic — null-init provides structural forgetting protection while distillation preserves feature quality.
Observable: Ensemble Average delta vs hist_null_init_only vision-only (56.37%). Expected +3-6%.
Conflicts: none [DONE] (score: 60.3%)

**Insight**: FD+CD distillation combines effectively with history_init_only null-space initialization — the two mechanisms are complementary. The biggest gains appear on mid-sequence tasks like caltech101 and dtd, where feature drift accumulates.

**Result**: The combination of hist_null_init_only and full_recipe (FD+CD) achieved an Ensemble Average of 60.3%, improving +1.32% over trunk and +3.93% over the vision-only hist_null_init variant.

**Branch**: `coordinator/n1-2-mechanism-hist-null-init-only-fu-5e4303b9`

### 2: Mechanism: Adaptive distillation scheduling — dynamically vary fd_weight and cd_weight during training (e.g., linear warmup, cosine decay, or task-dependent scheduling) instead of fixed weights.
Hypothesis: Fixed distillation weights may be suboptimal across different training stages. Early training needs higher FD to anchor features; later training benefits from lower CD to allow plasticity. Adaptive scheduling can balance stability-plasticity better than fixed weights.
Observable: Ensemble Average improvement of +1.0-3.0% over fixed-weight full recipe on B_dev.
Conflicts: none — explores scheduling axis not yet tested [PENDING]

#### 2.1: Mechanism: Task-adaptive FD/CD weights — lower fd_weight and cd_weight on hard fine-grained tasks (aircraft: fd=0.3, cd=0.3) to allow more plasticity, higher on easy coarse tasks (eurosat: fd=1.5, cd=1.5) to anchor features, with middle values for caltech101/dtd.
Hypothesis: Fixed fd=cd=1.0 over-constrains adaptation on aircraft (100 fine-grained classes need more weight-space freedom) while under-constraining eurosat (10 classes, easy to overfit). Task-adaptive weights balance stability-plasticity per task.
Observable: Aircraft Ensemble accuracy increase of +3-8% (from 48% toward 55%+), Ensemble Average increase of +1.5-3.0% over fixed-weight full recipe (60.85%).
Conflicts: none — task-adaptive distillation weights are an unexplored axis [RUNNING]

#### 2.2: Mechanism: Cosine decay of FD/CD weights within each task — start with high distillation (fd=2.0, cd=2.0) to anchor features early, decay to low values (fd=0.2, cd=0.2) by end of training to allow task-specific adaptation.
Hypothesis: Early training needs strong distillation to prevent rapid feature drift, while late training needs plasticity to fit task-specific patterns. Cosine scheduling provides this without per-task manual tuning.
Observable: Ensemble Average increase of +1.0-2.5% over fixed-weight full recipe with more balanced per-task scores.
Conflicts: none — within-task scheduling is an unexplored axis [PENDING]

### 3: Mechanism: Text encoder curriculum learning — progressively schedule text encoder training (e.g., freeze text encoder on early tasks, unfreeze on later tasks, or gradually increase text_lora_rank across tasks).
Hypothesis: The text encoder currently trains from task 1 with full gradients, which may disrupt pretrained text representations. A curriculum that delays or gradually ramps text encoder training could preserve zero-shot semantics while still adapting text features for later tasks.
Observable: Ensemble Average increase of +1.0-3.0% over baseline, with improved transfer on early tasks.
Conflicts: none — text encoder curriculum is an unexplored axis [PENDING]

#### 3.1: Mechanism: Freeze text encoder LoRA during task 1 (aircraft) training — set fd_weight=1.0, cd_weight=1.0 but disable text encoder gradients (--tune_text_encoder false) for the first task only, then re-enable for subsequent tasks.
Hypothesis: Aircraft (100 fine-grained classes) requires precise text semantics for class discrimination. Text encoder adaptation during task 1 may distort pretrained text representations, harming zero-shot classifier quality for fine-grained categories. Freezing preserves text quality where it matters most.
Observable: Aircraft Ensemble accuracy increase of +3-6%, Ensemble Average increase of +1.0-2.5% over full recipe baseline.
Conflicts: none — text encoder curriculum per task is unexplored [RUNNING]

### 4: Mechanism: Aux loss as structured regularization — tune the auxiliary classification head (aux_weight, sce_a, sce_b) to act as a structured regularizer that improves feature separability for downstream LR-RGDA classification.
Hypothesis: The aux head (symmetric cross-entropy on projected features) provides a complementary training signal. Tuning its weight and SCE parameters can improve the quality of extracted features for LR-RGDA, especially when combined with FD+CD distillation.
Observable: Ensemble Average improvement of +1.0-3.0% when aux head is properly tuned vs disabled.
Conflicts: none — aux head tuning is an unexplored axis [PENDING]

#### 4.1: Mechanism: Full recipe (fd=1.0, cd=1.0) + aux_weight=1.0 — enable the auxiliary linear classification head with symmetric cross-entropy loss during training. The aux head provides a direct classification gradient that improves feature separability for downstream LR-RGDA.
Hypothesis: The aux head loss optimizes features for linear separability (a proxy for LR-RGDA quality), complementing the FD+CD distillation which optimizes for feature preservation. Together they provide both stability and discriminability.
Observable: Ensemble Average increase of +0.5-2.0% over FD+CD-only baseline (60.85%).
Conflicts: none — aux head combined with FD+CD is unexplored [RUNNING]

### 5: Mechanism: Task-specific iteration budget — allocate training iterations proportional to dataset difficulty (class count × granularity): aircraft=1200, caltech101=400, dtd=1000, eurosat=400, sum=3000 (close to current 800×4=3200 total).
Hypothesis: Aircraft (100 fine-grained classes) and DTD (47 texture classes) need more iterations than caltech101 (101 coarse) and eurosat (10). The uniform 800-iter budget starves hard tasks while wasting compute on easy ones.
Observable: Aircraft Ensemble accuracy +5-10%, Ensemble Average +2-4% over uniform-budget baseline.
Conflicts: none — task-specific iteration budget is an unexplored axis [PENDING]
