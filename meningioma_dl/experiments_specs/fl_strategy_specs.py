from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from typing_extensions import Self

FL_STRATEGY_SPECS: Dict[str, Dict[str, Any]] = {
    "fed_avg_all_clients": {
        "name": "fed_avg",
        "config": {"fraction_fit": 1.0, "fraction_eval": 1.0},
    },
    "fed_prox_01": {
        "name": "fed_prox",
        "config": {"fraction_fit": 0.5, "fraction_eval": 1.0, "proximal_mu": 0.01},
    },
    "fed_avg_05_fraction": {
        "name": "fed_avg",
        "config": {"fraction_fit": 0.5, "fraction_eval": 1.0},
    },
}


@dataclass
class FLStrategySpecs:
    name: str = "fed_avg_all_clients"
    config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def get_from_name(cls, name: str) -> Self:
        return cls(**FL_STRATEGY_SPECS[name])
