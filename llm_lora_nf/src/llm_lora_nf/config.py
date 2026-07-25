from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


ATTENTION_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


@dataclass(frozen=True)
class FilterConfig:
    energy_fraction: float = 0.20
    leakage: float = 0.02
    ridge: float = 1e-4
    statistics_dtype: str = "float32"
    eigensolver_dtype: str = "float64"

    def __post_init__(self) -> None:
        if not 0.0 <= self.energy_fraction <= 1.0:
            raise ValueError("energy_fraction must lie in [0, 1]")
        if not 0.0 <= self.leakage <= 1.0:
            raise ValueError("leakage must lie in [0, 1]")
        if self.ridge < 0:
            raise ValueError("ridge must be non-negative")


@dataclass(frozen=True)
class AdapterConfig:
    method: str = "lora_nf"
    rank: int = 128
    alpha: float = 128.0
    dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: list(ATTENTION_TARGETS))
    filter: FilterConfig = field(default_factory=FilterConfig)

    def __post_init__(self) -> None:
        if self.method not in {"lora", "lora_nf", "lora_null", "milora"}:
            raise ValueError(f"Unsupported native adapter method: {self.method}")
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        invalid = set(self.target_modules) - set(ATTENTION_TARGETS)
        if invalid:
            raise ValueError(
                "LLM LoRA-NF is attention-only; unsupported targets: "
                + ", ".join(sorted(invalid))
            )

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
