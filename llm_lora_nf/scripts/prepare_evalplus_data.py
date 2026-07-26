#!/usr/bin/env python3
"""Prepare exact EvalPlus release assets outside Git."""

import argparse
import gzip
import json
import urllib.request
from pathlib import Path

from llm_lora_nf.config_io import load_yaml_config
from llm_lora_nf.evalplus_evidence import (
    EVALPLUS_DATASETS,
    EVALPLUS_DATA_MANIFEST_FORMAT_VERSION,
    sha256_file,
    validate_evalplus_data_manifest,
    validate_evalplus_dataset_file,
)


def _download(url: str, destination: Path) -> None:
    partial = destination.parent / f".{destination.name}.partial"
    if partial.exists():
        raise FileExistsError(
            f"Interrupted EvalPlus download requires audit: {partial}"
        )
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "llm-lora-nf-evalplus-preparer/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"EvalPlus download returned HTTP {response.status}: {url}"
                )
            with partial.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
        partial.replace(destination)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise


def _decompress(source: Path, destination: Path) -> None:
    partial = destination.parent / f".{destination.name}.partial"
    if partial.exists():
        raise FileExistsError(
            f"Interrupted EvalPlus decompression requires audit: {partial}"
        )
    try:
        with gzip.open(source, "rb") as compressed, partial.open("wb") as output:
            while True:
                chunk = compressed.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
        partial.replace(destination)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download exact EvalPlus official release assets, validate both "
            "compressed and decompressed bytes, and write an external manifest"
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument(
        "--reuse-verified",
        action="store_true",
        help="Reuse files only after complete hash/schema validation.",
    )
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    if config.get("datasets") != EVALPLUS_DATASETS:
        raise ValueError("Evaluation config does not contain the locked datasets")

    output_root = Path(args.output_root).resolve()
    manifest_path = Path(args.output_manifest).resolve()
    if manifest_path.exists():
        if not args.reuse_verified:
            raise FileExistsError(
                f"Refusing to overwrite EvalPlus manifest: {manifest_path}"
            )
        print(
            json.dumps(
                validate_evalplus_data_manifest(
                    str(manifest_path),
                    config=config,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = (
        manifest_path.parent / f".{manifest_path.name}.partial"
    )
    if temporary_manifest.exists():
        raise FileExistsError(
            "Interrupted EvalPlus manifest requires audit: "
            f"{temporary_manifest}"
        )

    records = {}
    for dataset, specification in EVALPLUS_DATASETS.items():
        compressed = output_root / str(specification["compressed_file"])
        jsonl = output_root / str(specification["jsonl_file"])
        if not compressed.exists():
            _download(str(specification["url"]), compressed)
        elif not args.reuse_verified:
            raise FileExistsError(
                f"EvalPlus file exists; pass --reuse-verified: {compressed}"
            )
        compressed_record = validate_evalplus_dataset_file(
            compressed,
            dataset=dataset,
            compressed=True,
        )
        if not jsonl.exists():
            _decompress(compressed, jsonl)
        elif not args.reuse_verified:
            raise FileExistsError(
                f"EvalPlus file exists; pass --reuse-verified: {jsonl}"
            )
        jsonl_record = validate_evalplus_dataset_file(
            jsonl,
            dataset=dataset,
            compressed=False,
        )
        records[dataset] = {
            "release": specification["release"],
            "version": specification["version"],
            "url": specification["url"],
            "compressed": compressed_record,
            "jsonl": jsonl_record,
        }

    payload = {
        "format_version": EVALPLUS_DATA_MANIFEST_FORMAT_VERSION,
        "source": "official_github_release",
        "datasets": records,
    }
    temporary_manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)
    validated = validate_evalplus_data_manifest(
        str(manifest_path),
        config=config,
    )
    validated["manifest_sha256"] = sha256_file(manifest_path)
    print(json.dumps(validated, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
