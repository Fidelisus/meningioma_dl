from dataclasses import dataclass
from typing import Optional, Dict, Any


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
    batch_size: int = 1
    use_training_data_for_validation: bool = False
    training_mode: str = "federated"
    number_of_clients: int = 2
    partitioning_mode: str = "uniform"
    reset_learning_rate_every_round: bool = False
    partitioning_settings: Optional[Dict[str, Any]] = None


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
    "federated_local_run_longer": {
        "training_mode": "federated",
        "global_epochs": 3,
        "epochs_per_round": 4,
        "number_of_clients": 2,
    },
    "federated_ci_run": {
        "training_mode": "federated",
        "global_epochs": 4,
        "epochs_per_round": 10,
        "number_of_clients": 4,
    },
    "federated_20r_20e_5c": {
        "training_mode": "federated",
        "global_epochs": 20,
        "epochs_per_round": 20,
        "number_of_clients": 5,
    },
    "federated_10r_40e_5c": {
        "training_mode": "federated",
        "global_epochs": 10,
        "epochs_per_round": 40,
        "number_of_clients": 5,
    },
    "federated_10r_40e_3c": {
        "training_mode": "federated",
        "global_epochs": 10,
        "epochs_per_round": 40,
        "number_of_clients": 3,
    },
    "federated_40r_10e_3c": {
        "training_mode": "federated",
        "global_epochs": 40,
        "epochs_per_round": 10,
        "number_of_clients": 3,
    },
    "federated_40r_10e_5c": {
        "training_mode": "federated",
        "global_epochs": 40,
        "epochs_per_round": 10,
        "number_of_clients": 5,
    },
    "federated_20r_20e_3c": {
        "training_mode": "federated",
        "global_epochs": 20,
        "epochs_per_round": 20,
        "number_of_clients": 3,
    },
    "federated_80r_10e_3c": {
        "training_mode": "federated",
        "global_epochs": 80,
        "epochs_per_round": 10,
        "number_of_clients": 3,
    },
    "federated_80r_2e_3c": {
        "training_mode": "federated",
        "global_epochs": 80,
        "epochs_per_round": 2,
        "number_of_clients": 3,
    },
    "ks05_local_run": {
        "training_mode": "federated",
        "global_epochs": 2,
        "epochs_per_round": 2,
        "number_of_clients": 2,
        "partitioning_mode": "ks_stat",
        "partitioning_settings": {"desired_ks_stat": 0.5},
    },
}


def get_training_specs(name: str):
    spec_dict = TRAINING_SPECS[name].copy()
    training_mode = spec_dict.pop("training_mode")

    if training_mode == "centralized":
        return CentralizedTrainingSpecs(**spec_dict)
    else:
        return FederatedTrainingSpecs(**spec_dict)
