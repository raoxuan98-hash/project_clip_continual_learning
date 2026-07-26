#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


def _mib(value: int) -> float:
    return value / (1024**2)


def _gib(value: int) -> float:
    return value / (1024**3)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate static attention-only adapter and LoRA-NF storage"
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument(
        "--protected-fraction",
        type=float,
        default=0.50,
        help="Assumed fraction of each activation dimension stored as protected basis.",
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.rank <= 0:
        raise ValueError("rank must be positive")
    if not 0.0 <= args.protected_fraction <= 1.0:
        raise ValueError("protected-fraction must be in [0, 1]")

    model_path = Path(args.model_path).resolve()
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    hidden = int(config["hidden_size"])
    layers = int(config["num_hidden_layers"])
    attention_heads = int(config["num_attention_heads"])
    key_value_heads = int(config.get("num_key_value_heads", attention_heads))
    head_dim = int(config.get("head_dim", hidden // attention_heads))
    q_out = attention_heads * head_dim
    kv_out = key_value_heads * head_dim
    intermediate = int(config["intermediate_size"])
    rank = args.rank

    q_params = rank * (hidden + q_out)
    k_params = rank * (hidden + kv_out)
    v_params = rank * (hidden + kv_out)
    o_params = rank * (q_out + hidden)
    lora_params = layers * (q_params + k_params + v_params + o_params)
    dora_magnitude_params = layers * (q_out + kv_out + kv_out + hidden)
    mlp_params = rank * (
        2 * (hidden + intermediate) + (intermediate + hidden)
    )
    track_a_lora_params = lora_params + layers * mlp_params

    qkv_protected = math.ceil(hidden * args.protected_fraction)
    o_protected = math.ceil(q_out * args.protected_fraction)
    shared_basis_elements = layers * (
        hidden * qkv_protected + q_out * o_protected
    )
    duplicated_basis_elements = layers * (
        3 * hidden * qkv_protected + q_out * o_protected
    )
    second_moment_elements = layers * (hidden * hidden + q_out * q_out)
    track_a_unique_moment_elements = layers * (
        2 * hidden * hidden
        + q_out * q_out
        + intermediate * intermediate
    )
    weight_bytes = sum(
        path.stat().st_size for path in model_path.glob("*.safetensors")
    )

    report = {
        "model_path": str(model_path),
        "architecture": {
            "model_type": config.get("model_type"),
            "layers": layers,
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "attention_heads": attention_heads,
            "key_value_heads": key_value_heads,
            "head_dim": head_dim,
            "q_output_size": q_out,
            "kv_output_size": kv_out,
        },
        "rank": rank,
        "attention_only_lora": {
            "trainable_parameters": lora_params,
            "fp32_parameter_mib": _mib(lora_params * 4),
            "fp32_train_optimizer_static_mib_range": [
                _mib(lora_params * 16),
                _mib(lora_params * 20),
            ],
            "dora_extra_magnitude_parameters": dora_magnitude_params,
            "decomposition_checkpoint_fp32_mib": _mib(lora_params * 8),
            "standard_checkpoint_fp32_mib": _mib(lora_params * 4),
        },
        "lora_nf": {
            "assumed_protected_fraction": args.protected_fraction,
            "qkv_protected_dimension": qkv_protected,
            "o_protected_dimension": o_protected,
            "shared_basis_bf16_mib": _mib(shared_basis_elements * 2),
            "unshared_qkv_basis_bf16_mib": _mib(
                duplicated_basis_elements * 2
            ),
            "sharing_saves_mib": _mib(
                (duplicated_basis_elements - shared_basis_elements) * 2
            ),
            "calibration_second_moments_fp32_gib": _gib(
                second_moment_elements * 4
            ),
            "dense_filter_fp32_gib_if_materialized": _gib(
                second_moment_elements * 4
            ),
        },
        "track_a_lora_null": {
            "target_scope": "q/k/v/o/gate/up/down",
            "trainable_parameters": track_a_lora_params,
            "fp32_parameter_mib": _mib(track_a_lora_params * 4),
            "decomposition_checkpoint_fp32_mib": _mib(
                track_a_lora_params * 8
            ),
            "calibration_unique_input_groups": (
                "qkv shared, o, gate/up shared, down"
            ),
            "calibration_moments_fp16_gib": _gib(
                track_a_unique_moment_elements * 2
            ),
            "calibration_moments_fp32_gib": _gib(
                track_a_unique_moment_elements * 4
            ),
            "note": (
                "Shared-input moments are mathematically identical to the "
                "official duplicated hooks and reduce host-memory storage."
            ),
        },
        "base_snapshot": {
            "transformers_weight_files_gib": _gib(weight_bytes),
            "estimated_fp32_merged_weights_gib": _gib(weight_bytes * 2),
        },
        "scope_note": (
            "Static estimates exclude activations, allocator fragmentation, "
            "eigensolver workspace, CUDA kernels, and framework overhead. "
            "Formal runs must also report measured peaks."
        ),
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
