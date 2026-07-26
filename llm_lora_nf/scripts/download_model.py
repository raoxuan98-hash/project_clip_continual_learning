#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

from llm_lora_nf.integrity import write_directory_integrity


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a model through ModelScope and record its local snapshot."
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", default="master")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument(
        "--include-original",
        action="store_true",
        help=(
            "Also download publisher-native weight exports such as original/*.pth. "
            "By default only the Transformers-compatible snapshot is downloaded."
        ),
    )
    args = parser.parse_args()

    from modelscope import snapshot_download
    from modelscope.hub.api import HubApi

    metadata = HubApi().get_model(args.model_id)
    remote_files = HubApi().get_model_files(
        args.model_id,
        revision=args.revision,
        recursive=True,
    )
    remote_by_path = {
        str(item.get("Path")): item
        for item in remote_files
        if item.get("Type") == "blob"
    }
    ignore_patterns = None
    if not args.include_original:
        ignore_patterns = ["original/**", "*.pth"]
    resolved = Path(
        snapshot_download(
            args.model_id,
            revision=args.revision,
            cache_dir=args.cache_dir,
            ignore_file_pattern=ignore_patterns,
        )
    ).resolve()
    files = []
    for path in sorted(resolved.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(resolved))
        if relative in {
            "local_snapshot_manifest.json",
            "artifact_integrity.json",
        }:
            continue
        record = {"path": relative, "size_bytes": path.stat().st_size}
        remote = remote_by_path.get(relative)
        if remote is not None:
            record["remote_revision"] = remote.get("Revision")
            record["remote_sha256"] = remote.get("Sha256")
        if relative in {
            "config.json",
            "generation_config.json",
            "tokenizer_config.json",
        }:
            record["sha256"] = _sha256(path)
        files.append(record)
    record = {
        "model_id": args.model_id,
        "requested_revision": args.revision,
        "resolved_path": str(resolved),
        "modelscope_revision": metadata.get("Revision"),
        "modelscope_last_updated_time": metadata.get("LastUpdatedTime"),
        "license": metadata.get("License"),
        "included_publisher_original_weights": args.include_original,
        "files": files,
    }
    manifest = resolved / "local_snapshot_manifest.json"
    manifest.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    integrity = write_directory_integrity(
        str(resolved),
        kind="model_snapshot",
    )
    print(
        json.dumps(
            {
                "snapshot": record,
                "integrity": integrity,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
