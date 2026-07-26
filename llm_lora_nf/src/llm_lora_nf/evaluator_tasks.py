import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping


PINNED_TASK_SOURCES = {
    "gsm8k_cot": {
        "dataset_id": "openai/gsm8k",
        "primary": "gsm8k/gsm8k-cot.yaml",
        "files": ["gsm8k/gsm8k-cot.yaml"],
    },
    "minerva_math500": {
        "dataset_id": "HuggingFaceH4/MATH-500",
        "primary": "minerva_math/minerva_math500.yaml",
        "files": [
            "minerva_math/minerva_math500.yaml",
            "minerva_math/minerva_math_algebra.yaml",
            "minerva_math/utils.py",
        ],
    },
    "triviaqa": {
        "dataset_id": "mandarjoshi/trivia_qa",
        "primary": "triviaqa/default.yaml",
        "files": ["triviaqa/default.yaml"],
    },
    "nq_open": {
        "dataset_id": "google-research-datasets/nq_open",
        "primary": "nq_open/nq_open.yaml",
        "files": ["nq_open/nq_open.yaml"],
    },
    "webqs": {
        "dataset_id": "web_questions",
        "primary": "webqs/webqs.yaml",
        "files": ["webqs/webqs.yaml", "webqs/utils.py"],
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def materialize_pinned_tasks(
    evaluator_root: str,
    output_root: str,
    *,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    evaluator_tasks = Path(evaluator_root).resolve() / "lm_eval" / "tasks"
    destination = Path(output_root).resolve()
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite materialized evaluator tasks: {destination}"
        )
    configured_tasks = [
        *config["tasks"]["downstream"],
        *config["tasks"]["retention"],
    ]
    if configured_tasks != list(PINNED_TASK_SOURCES):
        raise ValueError("Evaluator task materialization requires the locked task order")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            dir=str(destination.parent),
        )
    )
    try:
        primary_records = {}
        copied_sources = set()
        for task, specification in PINNED_TASK_SOURCES.items():
            dataset_id = specification["dataset_id"]
            revision = str(config["datasets"][dataset_id]["revision"])
            for relative in specification["files"]:
                source = evaluator_tasks / relative
                target = temporary / relative
                if not source.is_file():
                    raise FileNotFoundError(
                        f"Locked evaluator task source is missing: {source}"
                    )
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                copied_sources.add(relative)

            primary = temporary / specification["primary"]
            text = primary.read_text(encoding="utf-8")
            if "dataset_kwargs:" in text:
                raise ValueError(
                    f"Upstream task already defines dataset_kwargs: {task}"
                )
            primary.write_text(
                text.rstrip()
                + "\n"
                + "dataset_kwargs:\n"
                + f"  revision: {revision}\n",
                encoding="utf-8",
            )
            primary_records[task] = {
                "dataset_id": dataset_id,
                "revision": revision,
                "primary": specification["primary"],
            }

        files = [
            {
                "path": relative,
                "size_bytes": int((temporary / relative).stat().st_size),
                "sha256": _sha256(temporary / relative),
            }
            for relative in sorted(copied_sources)
        ]
        temporary.replace(destination)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return {
        "format_version": 1,
        "root": str(destination),
        "files": files,
        "tree_sha256": _canonical_hash(files),
        "tasks": primary_records,
    }
