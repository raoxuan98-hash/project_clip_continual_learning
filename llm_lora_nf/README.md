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

The guard rejects local research records and files larger than 10 MiB.
