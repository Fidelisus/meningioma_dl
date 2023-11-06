from dataclasses import dataclass

from meningioma_dl.experiments_specs.experiments import TRAINING_SPECS


@dataclass
class TrainingSpecs:
    training_mode: str


@dataclass
class CentralizedTrainingSpecs(TrainingSpecs):
    epochs: int = 100
    batch_size: int = 4
    use_training_data_for_validation: bool = False
    training_mode: str = "centralized"


def get_training_specs(name: str) -> CentralizedTrainingSpecs:
    spec_dict = TRAINING_SPECS[name].copy()
    training_mode = spec_dict.pop("training_mode")

    if training_mode == "centralized":
        return CentralizedTrainingSpecs(**spec_dict)
    else:
        raise ValueError("Only centralized training_mode supported for now")
