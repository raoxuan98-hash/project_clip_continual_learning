from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class RunMetadata:
    run_id: str
    execution_mode: str
    method: str
    model_id: str
    seed: int
    commit_sha: str

    def __post_init__(self) -> None:
        if self.execution_mode not in {"gpu_formal", "cpu_smoke_only"}:
            raise ValueError(f"Unknown execution_mode: {self.execution_mode}")

    @property
    def formal_result_eligible(self) -> bool:
        return self.execution_mode == "gpu_formal"

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["formal_result_eligible"] = self.formal_result_eligible
        return payload


def require_formal_result(metadata: RunMetadata) -> None:
    if not metadata.formal_result_eligible:
        raise ValueError(
            f"Run {metadata.run_id} is {metadata.execution_mode} and cannot enter formal results"
        )

