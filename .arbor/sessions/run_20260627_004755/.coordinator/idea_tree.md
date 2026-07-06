# Idea Tree

**Baseline**: N/A | **Trunk**: N/A

## ROOT: Phase 1: Verify 4 key experiments at 800-iter on aircraft caltech101 dtd eurosat, 2x2 grid: (current_nsp, hist_null_init_only) x (FdCd=0, FdCd=1). Use --lora_type lora_nsp --classifier_feature_transform test --batch_size 64 --seed 42. Run parallel on separate GPUs. Output experiments/null_basis_ablation/iter800/. Phase 2: Autonomously research to improve continual learning. Explore text encoder schedules, distillation weights, LoRA ranks, aux loss, projection/basis setups. Maximize Ensemble Average. Data at /data1/open_datasets/X-TAIL. [DONE]

**Insight**: Progress on the plasticity/stability grid experiment is stalled at the execution stage. The `use_safetensors` fix resolved model loading, but a new device-mismatch error in `_compute_lora_delta` blocks the planned ablation (null-basis: current_nsp vs. hist_null_init_only, FdCd=0 vs. 1). No empirical insights have been generated yet. The immediate priority is to debug and resolve the device-mismatch bug to unblock the experiment and test the hypotheses that hist_null_init_only + vision-only maximizes plasticity, while current_nsp + full recipe maximizes stability.

### 1: Mechanism: Null-basis ablation — 2x2 grid: (current_nsp, hist_null_init_only) × (FdCd=0, FdCd=1) at 800 iters on 4 datasets.
Hypothesis: This grid establishes the baseline landscape for Phase 2. Based on the 200-iter screening, hist_null_init_only should outperform current_nsp in vision-only mode (+0.56 avg on 3ds), but the FD/CD distillation may interact differently with each null-init variant.
Observable: Ensemble Average scores across 4 configurations. Expect hist_null_init_only + vision_only to be best for plasticity, current_nsp + full_recipe to be best for stability.
Conflicts: none — confirmatory grid, no prior conflicts. [RUNNING]

**Insight**: 4 experiments launched on GPUs 0-3. All running successfully: (1) current_nsp + full_recipe on GPU0, (2) current_nsp + vision_only on GPU1, (3) hist_null_init_only + vision_only on GPU2, (4) hist_null_init_only + full_recipe on GPU3. ETA ~6-8 hours for completion.

#### 1.1: Mechanism: Remove `use_safetensors` kwarg from CLIPModel.from_pretrained() in src/models/clip.py — transformers 4.15.0 passes it through to CLIPModel.__init__() which rejects it.
Hypothesis: This is an environment compatibility fix, not a research change. All 4 Phase 1 experiments fail without this.
Observable: Experiments start successfully.
Conflicts: none — pure compatibility fix. [MERGED]

**Insight**: Fixed transformers 4.15.0 compatibility: removed use_safetensors kwarg from CLIPModel.from_pretrained(). Fix committed to trunk (e5a1c2e).

**Result**: Removing `use_safetensors` resolved the TypeError crash. Model loading succeeded, but the experiment later crashed in `lora_sgp.py` due to a pre-existing device mismatch bug.

**Branch**: `coordinator/n1-1-mechanism-remove-use-safetensors-ee34cfb2`

### 2: Mechanism: Adaptive distillation scheduling — dynamically vary fd_weight and cd_weight across tasks (e.g., high distillation for early tasks to preserve them, tapering to lower distillation for later tasks to allow plasticity).
Hypothesis: Fixed distillation weights treat all tasks equally, but early tasks need more protection from forgetting while later tasks need more adaptation freedom. A monotonically decreasing schedule (e.g., fd_weight=1.5→0.5, cd_weight=1.5→0.5 over the 4-task sequence) should improve the stability-plasticity balance, yielding higher Average than fixed fd=cd=1.
Observable: Ensemble Average on B_dev should increase by ≥2 points vs baseline (fixed fd=cd=1) on the 4-dataset sequence.
Conflicts: none — attacks an axis (distillation schedule) no prior node touched. [PENDING]

### 3: Mechanism: Text encoder curriculum learning — systematically explore text_tuning_schedule variants beyond the simple freeze_after/low_lr_after. Key variants: (a) tune text for first K tasks then freeze, (b) progressively reduce text LR per task, (c) tune text only on tasks with large domain shifts.
Hypothesis: The text encoder shapes the semantic space used for zero-shot classification. Aggressive text tuning early builds a rich representation, while freezing later prevents interference. A well-chosen schedule could improve both Transfer (better semantic space) and Last (less interference), yielding +1–3 points Average.
Observable: Transfer and Last metrics on B_dev. Expect Transfer improvement from early text tuning, Last improvement from later freezing.
Conflicts: none — explores text-tuning axis not covered by prior ablation (which only varied null_init_mode and projection_param_mode). [PENDING]

### 4: Mechanism: Auxiliary loss as structured regularization — replace the simple linear aux_head with a deeper MLP aux_head or add multiple aux_heads at different feature levels (e.g., intermediate transformer layers), and tune aux_weight to provide complementary gradient signals that improve feature quality.
Hypothesis: The current aux_head (single linear layer, aux_weight=1.0) provides a weak regularization signal. A deeper aux_head or multi-level aux supervision could produce better-structured image features that are more robust to forgetting, without the computational cost of FD/CD distillation. This should improve Average by 1–3 points, especially when combined with reduced distillation.
Observable: Compare Average with and without FD/CD. A deeper aux_head should reduce dependence on distillation.
Conflicts: none — attacks the aux_loss axis, orthogonal to prior null-basis ablation. [PENDING]
