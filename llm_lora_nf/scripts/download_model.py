#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path


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
    args = parser.parse_args()

    from modelscope import snapshot_download
    from modelscope.hub.api import HubApi

    metadata = HubApi().get_model(args.model_id)
    resolved = Path(
        snapshot_download(
            args.model_id,
            revision=args.revision,
            cache_dir=args.cache_dir,
        )
    ).resolve()
    files = []
    for path in sorted(resolved.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(resolved))
        record = {"path": relative, "size_bytes": path.stat().st_size}
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
        "files": files,
    }
    manifest = resolved / "local_snapshot_manifest.json"
    manifest.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
