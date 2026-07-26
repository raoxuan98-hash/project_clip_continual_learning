#!/usr/bin/env python3
"""Acquire and seal the pinned Python CodeFeedback split via HF-Mirror."""

import argparse
import json
import os
from pathlib import Path

from llm_lora_nf.dataset_evidence import LOCKED_HF_MIRROR_ENDPOINT
from llm_lora_nf.dataset_io import (
    CODEFEEDBACK_SOURCE_ID,
    CODEFEEDBACK_SOURCE_REVISION,
    PISSA_CODEFEEDBACK_PYTHON_FILE,
    PISSA_CODEFEEDBACK_PYTHON_EMPTY_OUTPUT_INDICES,
    PISSA_CODEFEEDBACK_PYTHON_ROWS,
    PISSA_CODEFEEDBACK_PYTHON_SHA256,
    PISSA_CODEFEEDBACK_PYTHON_SIZE_BYTES,
    PISSA_DATASET_ID,
    PISSA_DATASET_REVISION,
    load_pissa_codefeedback_python_examples,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download the pinned PiSSA Python CodeFeedback split through "
            "HF-Mirror and write an external integrity manifest"
        )
    )
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--endpoint", default=LOCKED_HF_MIRROR_ENDPOINT)
    args = parser.parse_args()
    if args.endpoint != LOCKED_HF_MIRROR_ENDPOINT:
        raise ValueError(
            f"Locked endpoint is {LOCKED_HF_MIRROR_ENDPOINT}"
        )

    cache_dir = Path(args.cache_dir).resolve()
    output = Path(args.output_manifest).resolve()
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite code-data manifest: {output}"
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp"
    if temporary.exists():
        raise FileExistsError(
            f"Interrupted code-data manifest requires audit: {temporary}"
        )

    os.environ["HF_ENDPOINT"] = args.endpoint
    os.environ.pop("HF_HUB_OFFLINE", None)
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(endpoint=args.endpoint)
    actual_revision = api.dataset_info(
        PISSA_DATASET_ID,
        revision=PISSA_DATASET_REVISION,
    ).sha
    if actual_revision != PISSA_DATASET_REVISION:
        raise ValueError(
            f"{PISSA_DATASET_ID} resolved to {actual_revision}, "
            f"expected {PISSA_DATASET_REVISION}"
        )
    downloaded = Path(
        hf_hub_download(
            repo_id=PISSA_DATASET_ID,
            filename=PISSA_CODEFEEDBACK_PYTHON_FILE,
            repo_type="dataset",
            revision=PISSA_DATASET_REVISION,
            cache_dir=str(cache_dir),
            endpoint=args.endpoint,
        )
    ).resolve()
    try:
        downloaded.relative_to(cache_dir)
    except ValueError as error:
        raise ValueError(
            f"Downloaded code dataset escaped the external cache: {downloaded}"
        ) from error

    examples = load_pissa_codefeedback_python_examples(
        str(downloaded),
        first_n=PISSA_CODEFEEDBACK_PYTHON_ROWS,
        expected_sha256=PISSA_CODEFEEDBACK_PYTHON_SHA256,
        expected_size_bytes=PISSA_CODEFEEDBACK_PYTHON_SIZE_BYTES,
        expected_total_rows=PISSA_CODEFEEDBACK_PYTHON_ROWS,
        allowed_empty_output_indices=(
            PISSA_CODEFEEDBACK_PYTHON_EMPTY_OUTPUT_INDICES
        ),
    )
    payload = {
        "format_version": 1,
        "endpoint": args.endpoint,
        "dataset": {
            "repository": PISSA_DATASET_ID,
            "revision": actual_revision,
            "file": PISSA_CODEFEEDBACK_PYTHON_FILE,
            "resolved_path": str(downloaded),
            "size_bytes": downloaded.stat().st_size,
            "sha256": sha256_file(str(downloaded)),
            "rows": len(examples),
            "empty_output_indices": list(
                PISSA_CODEFEEDBACK_PYTHON_EMPTY_OUTPUT_INDICES
            ),
            "schema": {"instruction": "str", "output": "str"},
            "source_dataset": CODEFEEDBACK_SOURCE_ID,
            "source_revision": CODEFEEDBACK_SOURCE_REVISION,
        },
    }
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(
        json.dumps(
            {
                "manifest_path": str(output),
                "manifest_sha256": sha256_file(str(output)),
                "dataset": payload["dataset"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
