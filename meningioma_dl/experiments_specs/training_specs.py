from dataclasses import dataclass
from typing import Optional, Dict, Any, Union


@dataclass
class CentralizedTrainingSpecs:
    epochs: int = 100
    batch_size: int = 4
    training_mode: str = "centralized"


@dataclass
class FederatedTrainingSpecs:
    global_epochs: int = 3
    epochs_per_round: int = 4
    batch_size: int = 1
    training_mode: str = "federated"
    number_of_clients: int = 2
    partitioning_mode: str = "uniform"
    reset_learning_rate_every_round: bool = False
    partitioning_settings: Optional[Dict[str, Any]] = None


def create_fl_parameters(
    global_epochs: int, epochs_per_round: int, number_of_clients: int = 3
) -> Dict[str, Union[str, int]]:
    return {
        "training_mode": "federated",
        "global_epochs": global_epochs,
        "epochs_per_round": epochs_per_round,
        "number_of_clients": number_of_clients,
    }


def create_fl_specs(run_name: str) -> Dict[str, Dict[str, Union[str, int]]]:
    run_name_tokens = run_name.split("_")
    global_epochs = int(run_name_tokens[1][:-1])
    epochs_per_round = int(run_name_tokens[2][:-1])
    number_of_clients = int(run_name_tokens[3][:-1])
    return {
        run_name: create_fl_parameters(
            global_epochs, epochs_per_round, number_of_clients
        )
    }


TRAINING_SPECS = {
    "evaluation": {"training_mode": "centralized", "epochs": 1},
    "central_1_epochs": {"training_mode": "centralized", "epochs": 1},
    "central_2_epochs": {"training_mode": "centralized", "epochs": 2},
    "central_100_epochs": {"training_mode": "centralized", "epochs": 100},
    "central_150_epochs": {"training_mode": "centralized", "epochs": 150},
    "central_200_epochs": {"training_mode": "centralized", "epochs": 200},
    "central_300_epochs": {"training_mode": "centralized", "epochs": 300},
    "central_400_epochs": {"training_mode": "centralized", "epochs": 400},
    "federated_local_run": create_fl_parameters(
        global_epochs=2, epochs_per_round=2, number_of_clients=2
    ),
    "federated_local_run_longer": create_fl_parameters(
        global_epochs=3, epochs_per_round=4, number_of_clients=2
    ),
    "federated_ci_run": create_fl_parameters(
        global_epochs=4, epochs_per_round=10, number_of_clients=4
    ),
    **create_fl_specs("federated_200r_1e_3c"),
    **create_fl_specs("federated_300r_1e_3c"),
    **create_fl_specs("federated_200r_1e_5c"),
    **create_fl_specs("federated_100r_2e_3c"),
    **create_fl_specs("federated_100r_2e_5c"),
    **create_fl_specs("federated_40r_5e_3c"),
    **create_fl_specs("federated_40r_5e_5c"),
    **create_fl_specs("federated_20r_20e_3c"),
    **create_fl_specs("federated_20r_20e_5c"),
    "ks05_local_run": {
        **create_fl_parameters(
            global_epochs=2, epochs_per_round=2, number_of_clients=3
        ),
        "partitioning_mode": "ks_stat",
        "partitioning_settings": {"file_name": "ks_stat_01_3c"},
    },
    "ks04_200r_1e_3c": {
        **create_fl_parameters(
            global_epochs=200, epochs_per_round=1, number_of_clients=3
        ),
        "partitioning_mode": "ks_stat",
        "partitioning_settings": {"file_name": "ks_stat_04_3c"},
    },
    "ks024_200r_1e_3c": {
        **create_fl_parameters(
            global_epochs=200, epochs_per_round=1, number_of_clients=3
        ),
        "partitioning_mode": "ks_stat",
        "partitioning_settings": {"file_name": "ks_stat_024_3c"},
    },
    "ks01_200r_1e_3c": {
        **create_fl_parameters(
            global_epochs=200, epochs_per_round=1, number_of_clients=3
        ),
        "partitioning_mode": "ks_stat",
        "partitioning_settings": {"file_name": "ks_stat_01_3c"},
    },
}


def get_training_specs(name: str):
    spec_dict = TRAINING_SPECS[name].copy()
    training_mode = spec_dict.pop("training_mode")

    if training_mode == "centralized":
        return CentralizedTrainingSpecs(**spec_dict)
    else:
        return FederatedTrainingSpecs(**spec_dict)
