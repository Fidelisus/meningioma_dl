from dataclasses import dataclass


@dataclass
class CentralizedTrainingSpecs:
    epochs: int = 100
    batch_size: int = 4
    use_training_data_for_validation: bool = False
    training_mode: str = "centralized"


@dataclass
class FederatedTrainingSpecs:
    global_epochs: int = 3
    epochs_per_round: int = 4
    batch_size: int = 2
    use_training_data_for_validation: bool = False
    training_mode: str = "federated"
    number_of_clients: int = 2
    partitioning_mode: str = "uniform"


TRAINING_SPECS = {
    "central_1_epochs": {"training_mode": "centralized", "epochs": 1},
    "central_2_epochs": {"training_mode": "centralized", "epochs": 2},
    "central_100_epochs": {"training_mode": "centralized", "epochs": 100},
    "central_150_epochs": {"training_mode": "centralized", "epochs": 150},
    "central_200_epochs": {"training_mode": "centralized", "epochs": 200},
    "central_300_epochs": {"training_mode": "centralized", "epochs": 300},
    "federated_local_run": {
        "training_mode": "federated",
        "global_epochs": 2,
        "epochs_per_round": 2,
        "number_of_clients": 2,
    },
    "federated_ci_run": {
        "training_mode": "federated",
        "global_epochs": 4,
        "epochs_per_round": 10,
        "number_of_clients": 4,
    },
}


def get_training_specs(name: str):
    spec_dict = TRAINING_SPECS[name].copy()
    training_mode = spec_dict.pop("training_mode")

    if training_mode == "centralized":
        return CentralizedTrainingSpecs(**spec_dict)
    else:
        return FederatedTrainingSpecs(**spec_dict)
