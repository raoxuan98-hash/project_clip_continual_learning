#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import yaml

from llm_lora_nf.dataset_evidence import (
    EVAL_DATA_MANIFEST_FORMAT_VERSION,
    LOCKED_HF_MIRROR_ENDPOINT,
    validate_eval_data_manifest,
)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_file_record(path: Path, *, hf_home: Path) -> Dict[str, Any]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(hf_home)
    except ValueError as error:
        raise ValueError(
            f"Prepared dataset cache file escapes HF_HOME: {resolved}"
        ) from error
    return {
        "path": relative.as_posix(),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": _sha256(resolved),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve and cache pinned lm-eval datasets through HF-Mirror"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--hf-home", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--endpoint", default=LOCKED_HF_MIRROR_ENDPOINT)
    parser.add_argument(
        "--only",
        action="append",
        help="Prepare only the named dataset ID; may be repeated.",
    )
    args = parser.parse_args()
    if args.endpoint != LOCKED_HF_MIRROR_ENDPOINT:
        raise ValueError(
            f"Locked endpoint is {LOCKED_HF_MIRROR_ENDPOINT}"
        )

    hf_home = Path(args.hf_home).resolve()
    os.environ["HF_ENDPOINT"] = args.endpoint
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_DATASETS_CACHE"] = str(
        (hf_home / "datasets").resolve()
    )
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("HF_DATASETS_OFFLINE", None)

    from datasets import load_dataset
    from huggingface_hub import HfApi

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    datasets_config = config["datasets"]
    selected = set(args.only or datasets_config)
    unknown = selected - set(datasets_config)
    if unknown:
        raise ValueError("Unknown dataset IDs: " + ", ".join(sorted(unknown)))

    api = HfApi(endpoint=args.endpoint)
    records: Dict[str, Dict[str, Any]] = {}
    for dataset_id, specification in datasets_config.items():
        if dataset_id not in selected:
            continue
        expected_revision = str(specification["revision"])
        actual_revision = api.dataset_info(
            dataset_id,
            revision=expected_revision,
        ).sha
        if actual_revision != expected_revision:
            raise ValueError(
                f"{dataset_id} resolved to {actual_revision}, "
                f"expected {expected_revision}"
            )
        dataset = load_dataset(
            dataset_id,
            name=specification.get("config"),
            revision=expected_revision,
        )
        records[dataset_id] = {
            "config": specification.get("config"),
            "revision": actual_revision,
            "splits": {
                split: {
                    "rows": len(value),
                    "fingerprint": value._fingerprint,
                    "cache_files": [
                        _cache_file_record(
                            Path(cache_file["filename"]),
                            hf_home=hf_home,
                        )
                        for cache_file in value.cache_files
                    ],
                }
                for split, value in dataset.items()
            },
        }

    output = Path(args.output_manifest).resolve()
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite evaluation-data manifest: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp"
    if temporary.exists():
        raise FileExistsError(
            f"Interrupted evaluation-data manifest requires audit: {temporary}"
        )
    payload = {
        "format_version": EVAL_DATA_MANIFEST_FORMAT_VERSION,
        "endpoint": args.endpoint,
        "hf_home": str(hf_home),
        "datasets": records,
    }
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validation = None
    if not args.only:
        validation = validate_eval_data_manifest(
            str(temporary),
            config=config,
            hf_home=str(hf_home),
        )
    temporary.replace(output)
    if validation is not None:
        validation["manifest_path"] = str(output)
    print(
        json.dumps(
            {
                "manifest": payload,
                "validation": validation,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
