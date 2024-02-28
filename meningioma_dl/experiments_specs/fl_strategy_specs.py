from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from typing_extensions import Self

FL_STRATEGY_SPECS: Dict[str, Dict[str, Any]] = {
    "fed_avg_default": {"name": "fed_avg", "config": {}},
}


@dataclass
class FLStrategySpecs:
    name: str = "fed_avg_default"
    config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def get_from_name(cls, name: str) -> Self:
        return cls(**FL_STRATEGY_SPECS[name])
