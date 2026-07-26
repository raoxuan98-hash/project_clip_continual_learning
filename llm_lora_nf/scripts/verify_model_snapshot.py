#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from llm_lora_nf.integrity import write_directory_integrity


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hash a fully local model snapshot without network access"
    )
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()
    model_path = Path(args.model_path).resolve()
    snapshot_manifest = model_path / "local_snapshot_manifest.json"
    if not snapshot_manifest.is_file():
        raise FileNotFoundError(
            f"Missing local snapshot provenance manifest: {snapshot_manifest}"
        )
    record = write_directory_integrity(
        str(model_path),
        kind="model_snapshot",
    )
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
