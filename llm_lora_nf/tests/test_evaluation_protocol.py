import copy
import hashlib
import json
from pathlib import Path

import pytest

from llm_lora_nf.config_io import load_yaml_config
from llm_lora_nf.dataset_evidence import validate_eval_data_manifest
from llm_lora_nf.evaluator_tasks import (
    PINNED_TASK_SOURCES,
    materialize_pinned_tasks,
)
from llm_lora_nf.protocol_validation import validate_formal_evaluation_config
from scripts.aggregate_lm_eval import _validate_result_task_configs


def _config():
    return load_yaml_config(
        str(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "evaluation"
            / "track_b_math_knowledge.yaml"
        )
    )


def test_locked_formal_evaluation_config_accepts_paper_protocol():
    validate_formal_evaluation_config(_config())


def test_locked_formal_evaluation_config_rejects_task_drift():
    config = copy.deepcopy(_config())
    config["tasks"]["retention"].pop()
    with pytest.raises(ValueError, match="tasks.retention"):
        validate_formal_evaluation_config(config)


def test_locked_formal_evaluation_config_rejects_dataset_revision_drift():
    config = copy.deepcopy(_config())
    config["datasets"]["openai/gsm8k"]["revision"] = "main"
    with pytest.raises(ValueError, match="datasets"):
        validate_formal_evaluation_config(config)


def test_locked_formal_evaluation_config_rejects_evaluator_seed_drift():
    config = copy.deepcopy(_config())
    config["evaluator"]["seeds"]["fewshot"] = 42
    with pytest.raises(ValueError, match="evaluator.seeds"):
        validate_formal_evaluation_config(config)


def test_evaluation_data_manifest_binds_actual_cache_files(tmp_path):
    config = _config()
    hf_home = tmp_path / "hf_home"
    records = {}
    for index, (dataset_id, specification) in enumerate(
        config["datasets"].items()
    ):
        cache_file = hf_home / "datasets" / f"dataset_{index}.arrow"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_bytes(f"dataset-{index}".encode())
        records[dataset_id] = {
            "config": specification["config"],
            "revision": specification["revision"],
            "splits": {
                "test": {
                    "rows": 1,
                    "fingerprint": f"fingerprint-{index}",
                    "cache_files": [
                        {
                            "path": cache_file.relative_to(hf_home).as_posix(),
                            "size_bytes": cache_file.stat().st_size,
                            "sha256": hashlib.sha256(
                                cache_file.read_bytes()
                            ).hexdigest(),
                        }
                    ],
                }
            },
        }
    manifest_path = tmp_path / "evaluation_data.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "endpoint": "https://hf-mirror.com",
                "hf_home": str(hf_home.resolve()),
                "datasets": records,
            }
        ),
        encoding="utf-8",
    )
    evidence = validate_eval_data_manifest(
        str(manifest_path),
        config=config,
        hf_home=str(hf_home),
    )
    assert evidence["verified_cache_files"] == len(config["datasets"])

    (hf_home / "datasets" / "dataset_0.arrow").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum|size"):
        validate_eval_data_manifest(
            str(manifest_path),
            config=config,
            hf_home=str(hf_home),
        )


def test_materialized_evaluator_tasks_pin_every_dataset_revision(tmp_path):
    evaluator_tasks = tmp_path / "evaluator" / "lm_eval" / "tasks"
    for specification in PINNED_TASK_SOURCES.values():
        for relative in specification["files"]:
            source = evaluator_tasks / relative
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text(
                "task: upstream\n",
                encoding="utf-8",
            )
    config = _config()
    destination = tmp_path / "materialized"
    evidence = materialize_pinned_tasks(
        str(tmp_path / "evaluator"),
        str(destination),
        config=config,
    )
    assert set(evidence["tasks"]) == set(PINNED_TASK_SOURCES)
    for task, specification in PINNED_TASK_SOURCES.items():
        dataset_id = specification["dataset_id"]
        primary = destination / specification["primary"]
        assert (
            "revision: " + config["datasets"][dataset_id]["revision"]
            in primary.read_text(encoding="utf-8")
        )


def test_materialized_evaluator_tasks_leave_no_partial_destination(tmp_path):
    destination = tmp_path / "materialized"
    with pytest.raises(FileNotFoundError, match="task source is missing"):
        materialize_pinned_tasks(
            str(tmp_path / "empty_evaluator"),
            str(destination),
            config=_config(),
        )
    assert not destination.exists()


def test_lm_eval_result_must_report_the_pinned_dataset_revisions():
    config = _config()
    tasks = [
        *config["tasks"]["downstream"],
        *config["tasks"]["retention"],
    ]
    result = {"configs": {}}
    for task in tasks:
        dataset_id = PINNED_TASK_SOURCES[task]["dataset_id"]
        specification = config["datasets"][dataset_id]
        result["configs"][task] = {
            "dataset_path": dataset_id,
            "dataset_name": specification["config"],
            "dataset_kwargs": {
                "revision": specification["revision"],
            },
        }
    _validate_result_task_configs(result, config, tasks)
    result["configs"]["gsm8k_cot"]["dataset_kwargs"]["revision"] = "main"
    with pytest.raises(ValueError, match="locked dataset revision"):
        _validate_result_task_configs(result, config, tasks)
