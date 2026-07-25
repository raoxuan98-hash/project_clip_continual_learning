#!/usr/bin/env python3
import argparse
import json

from llm_lora_nf.resource_guard import inspect_admission, project_gpu_lock


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requested", type=int, default=2)
    parser.add_argument("--lock-path", default="/tmp/llm_lora_nf_gpu.lock")
    args = parser.parse_args()

    with project_gpu_lock(args.lock_path):
        decision = inspect_admission(requested=args.requested)
        print(
            json.dumps(
                {
                    "mode": decision.mode,
                    "selected_gpu_indices": decision.selected_gpu_indices,
                    "idle_gpu_indices": decision.idle_gpu_indices,
                    "reason": decision.reason,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()

