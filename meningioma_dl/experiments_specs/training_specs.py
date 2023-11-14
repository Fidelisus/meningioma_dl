from dataclasses import dataclass


@dataclass
class TrainingSpecs:
    training_mode: str


@dataclass
class CentralizedTrainingSpecs(TrainingSpecs):
    epochs: int = 100
    batch_size: int = 4
    use_training_data_for_validation: bool = False
    training_mode: str = "centralized"


TRAINING_SPECS = {
    "central_2_epochs": {"training_mode": "centralized", "epochs": 2},
    "central_100_epochs": {"training_mode": "centralized", "epochs": 100},
    "central_150_epochs": {"training_mode": "centralized", "epochs": 150},
    "central_200_epochs": {"training_mode": "centralized", "epochs": 200},
    "central_300_epochs": {"training_mode": "centralized", "epochs": 300},
}


def get_training_specs(name: str) -> CentralizedTrainingSpecs:
    spec_dict = TRAINING_SPECS[name].copy()
    training_mode = spec_dict.pop("training_mode")

    if training_mode == "centralized":
        return CentralizedTrainingSpecs(**spec_dict)
    else:
        raise ValueError("Only centralized training_mode supported for now")
