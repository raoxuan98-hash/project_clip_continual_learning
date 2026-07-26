from typing import Any, Mapping


LOCKED_ARTIFACT_RETENTION = {
    "base_snapshots_per_revision": 1,
    "final_adapter_checkpoints_per_run": 1,
    "save_intermediate_checkpoints": False,
    "save_optimizer_state": False,
    "save_scheduler_state": False,
    "merged_checkpoint_policy": (
        "ephemeral_delete_after_qualified_evaluation"
    ),
    "max_concurrent_merged_checkpoints": 1,
    "request_cache_scope": (
        "shared_content_addressed_per_base_model_protocol"
    ),
}


def validate_artifact_retention(policy: Mapping[str, Any]) -> None:
    if dict(policy) != LOCKED_ARTIFACT_RETENTION:
        raise ValueError(
            "Formal runs must use the locked low-storage artifact-retention "
            "policy"
        )
