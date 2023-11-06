import copy
from dataclasses import dataclass, field
from typing import Dict, Any

from torch import optim
from typing_extensions import Self


SCHEDULER_SPECS: Dict[str, Dict[str, Any]] = {
    "001_lr_099_gamma": {
        "scheduler": "exponential",
        "learning_rate": 0.01,
        "gamma": 0.99,
    },
    "002_lr_099_gamma": {
        "scheduler": "exponential",
        "learning_rate": 0.02,
        "gamma": 0.99,
    },
    "002_lr_0995_gamma": {
        "scheduler": "exponential",
        "learning_rate": 0.02,
        "gamma": 0.995,
    },
    "005_lr_099_gamma": {
        "scheduler": "exponential",
        "learning_rate": 0.05,
        "gamma": 0.99,
    },
    "01_lr_099_gamma": {
        "scheduler": "exponential",
        "learning_rate": 0.1,
        "gamma": 0.99,
    },
    "02_lr_099_gamma": {
        "scheduler": "exponential",
        "learning_rate": 0.2,
        "gamma": 0.99,
    },
    "05_lr_09_gamma": {
        "scheduler": "exponential",
        "learning_rate": 0.5,
        "gamma": 0.9,
    },
    "05_lr_099_gamma": {
        "scheduler": "exponential",
        "learning_rate": 0.5,
        "gamma": 0.99,
    },
    "05_lr_0995_gamma": {
        "scheduler": "exponential",
        "learning_rate": 0.5,
        "gamma": 0.995,
    },
    "1_lr_099_gamma": {
        "scheduler": "exponential",
        "learning_rate": 1.0,
        "gamma": 0.99,
    },
    "cosine_lr_0004_t0_60": {
        "scheduler": "cosine",
        "learning_rate": 0.004,
        "T_0": 0,
        "eta_min": 60,
    },
    "cosine_lr_1_t0_60": {
        "scheduler": "cosine",
        "learning_rate": 1.0,
        "T_0": 0,
        "eta_min": 60,
    },
    "cosine_lr_05_t0_20": {
        "scheduler": "cosine",
        "learning_rate": 0.5,
        "T_0": 0,
        "eta_min": 20,
    },
}


SCHEDULERS = {
    "exponential": optim.lr_scheduler.ExponentialLR,
    "cosine": optim.lr_scheduler.CosineAnnealingWarmRestarts,
}


def default_field(obj):
    return field(default_factory=lambda: copy.copy(obj))


@dataclass
class SchedulerSpecs:
    scheduler_name: str = "exponential"
    learning_rate: float = 0.5
    parameters: Dict[str, Any] = default_field({"gamma": 0.99})

    @classmethod
    def get_from_name(cls, name: str) -> Self:
        chosen_specs = SCHEDULER_SPECS[name].copy()
        scheduler_name = chosen_specs.pop("scheduler")
        learning_rate = chosen_specs.pop("learning_rate")
        return cls(scheduler_name, learning_rate, chosen_specs)

    def get_scheduler(self, optimizer):
        return SCHEDULERS[self.scheduler_name](optimizer, **self.parameters)
