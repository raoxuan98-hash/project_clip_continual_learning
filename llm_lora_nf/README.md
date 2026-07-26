# LLM LoRA-NF

Independent LLM implementation of the project’s full runtime LoRA-NF filter.

The implementation is intentionally isolated from the existing CLIP `src/`, `configs/`,
`scripts/`, and `tests/` trees.

## Method boundary

- Instruct checkpoints only.
- Attention projections only: `q_proj`, `k_proj`, `v_proj`, `o_proj`.
- Runtime input-side filter on every training forward.
- Standard LoRA initialization; DoRA remains a separate baseline.
- Dense second-moment statistics may be used for calibration, but the runtime filter
  stores and applies only the protected basis.

The publishable comparison has two explicitly separated tracks:

- Track A is a protocol-aligned Instruct reimplementation of LoRA-Null commit
  `1e6808abb81fe10e50b8172c40ac9a8ab4f11e83`. It uses the official seven
  decoder projections, raw NQ character spans, signed-maximum normalization,
  and Alpaca-style MetaMathQA prompt. Its code lives in `track_a.py` and
  `run_track_a_sft.py`.
- Track B is the controlled attention-only study. LoRA-NF and every baseline
  target only `q_proj/k_proj/v_proj/o_proj`.

Track A cannot relax the Track B `AdapterConfig`; its configuration and
checkpoint loader are separate.

The authoritative specification is
`../docs/llm-lora-nf/METHOD_SPEC.md` in the local research workspace.

## Execution boundary

Do not run this project locally. Tests, model loading, training, and evaluation run on:

```text
raoxuan@10.20.34.30
/home/raoxuan/projects/project_clip_continual_learning_llm/llm_lora_nf
```

No more than two GPUs may be used by this development line, and one GPU must be
left unassigned at every admission decision. If CUDA is unavailable, only CPU smoke
tests are permitted and their outputs must be marked `cpu_smoke_only`.

Before each commit, stage only the intended code and run:

```bash
PYTHONPATH=src python scripts/check_git_payload.py
```

The guard uses an allowlist for the isolated LLM code/config/test/documentation
payload, and rejects local research records, unknown paths, model/data artifacts,
and files larger than 10 MiB.

## Reproducible pipeline

The main attention-only math protocol is split into explicit stages:

1. `scripts/download_model.py` downloads a pinned ModelScope snapshot and writes a
   per-file manifest. Publisher-native `original/*.pth` exports are excluded by
   default because the Transformers safetensors are the executable checkpoint.
2. `scripts/prepare_eval_data.py` resolves the dataset revisions recorded in
   `configs/evaluation/track_b_math_knowledge.yaml` through HF-Mirror and prepares
   the offline cache.
3. `scripts/run_sft.py` runs the resource-guarded unified SFT implementation for
   LoRA, DoRA, LoRA-Null, PiSSA, MiLoRA, CorDA, or LoRA-NF.
   `scripts/run_track_a_sft.py` separately runs the LoRA-Null public-protocol
   gate for LoRA and LoRA-Null. Formal configs are hard-validated rather than
   trusted by filename. LoRA-NF and LoRA-Null moments may use the external,
   content-addressed calibration cache configured by the matrix; its identity
   binds model/data hashes, code commit, dtype, target scope, and protocol.
   The controlled LoRA baseline uses the pinned PEFT implementation; the
   project-native wrapper is reserved for methods whose decomposition or
   runtime filter is not represented by ordinary PEFT LoRA.
4. `scripts/export_merged.py` restores and merges either a native or PEFT adapter
   into a standard Hugging Face checkpoint. The primary comparison exports in
   FP32: BF16 matrix merging is not numerically identical to evaluating the
   factorized low-rank branch and is treated as a separate precision ablation.
5. `scripts/run_lm_eval.py` verifies the evaluator commit, source cleanliness,
   model eligibility, HF-Mirror dataset manifest, actual offline cache-file
   checksums, and GPU admission before invoking the official harness. It seals
   the complete raw evaluation directory with a file-by-file SHA-256 manifest
   and embeds the merged export metadata so the dense merged checkpoint need
   not be retained. It also derives task definitions from the exact evaluator
   checkout, injects the locked dataset revisions, and supplies them through
   `--include_path`; the resolved task configs are checked again at aggregation.
6. `scripts/aggregate_lm_eval.py` rejects CPU smoke outputs unless explicitly
   invoked for chain testing. Summaries must be written outside the
   integrity-protected raw evaluation directory.
7. `scripts/aggregate_seed_results.py` requires the exact registered seed set,
   reports mean/sample standard deviation/seed-level bootstrap intervals, and
   computes paired LoRA-NF-minus-baseline differences only when the normalized
   cross-method training-protocol hashes and training software/GPU identities
   match. Sample-level bootstrap
   is performed separately by `scripts/bootstrap_paired_samples.py`, which
   pairs exact document hashes from lm-eval sample logs.

The matrix launcher exposes `track_a_gate`, `protocol_gate`, and `main` stages.
Each stage records which runner it uses. The evaluation launcher automatically
creates or reuses one content-addressed base evaluation per exact model snapshot,
evaluation protocol, dataset manifest, and source commit; `--base-run` remains
only as an audited override.

The formal artifact-retention policy is deliberately small:

- one shared base snapshot per exact model revision;
- one final adapter checkpoint per model/method/seed run;
- no intermediate, optimizer, or scheduler checkpoints;
- at most one merged checkpoint at a time;
- the merged checkpoint is deleted only after qualified evaluation output is
  integrity-sealed and aggregated.

Base snapshots, final adapters, raw sample logs, integrity manifests, and
summaries remain as the reproducibility evidence. The deleted merged model is
reconstructible from the shared base snapshot and final adapter.

Formal evaluations fix lm-eval's Python/NumPy/Torch/few-shot seeds to
`0,1234,1234,1234`, independently of the training seed. This makes prompt
construction identical across methods and training repetitions. Its request
cache is stored once in a dedicated content-addressed root and integrity-sealed;
CPU smoke runs disable request caching because lm-eval otherwise builds the
entire task cache even when `--limit` is set. The cache key also binds the
Python/package/distribution/CUDA/cuDNN software fingerprint so serialized task
instances are never reused across an environment change.

The locked evaluator is `lm-evaluation-harness` v0.4.12 at commit
`6d642546f4688648fced259eb3302efd36ece5af`. Its optional dependencies are
recorded separately in `requirements-evaluation.txt`.

All model, data, merged-weight, sample-log, and result paths must remain outside
Git under the server data root. Calibration cache entries are large runtime
artifacts and must never be staged. A CPU run may validate every stage, but its
manifest always has `formal_result_eligible=false`.
