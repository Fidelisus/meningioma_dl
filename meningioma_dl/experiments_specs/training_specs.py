from dataclasses import dataclass
from typing import Optional, Dict, Any, Union


@dataclass
class CentralizedTrainingSpecs:
    epochs: int = 100
    batch_size: int = 4
    training_mode: str = "centralized"
    client_specific_preprocessing: Optional[Dict[int, Optional[str]]] = None


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
    client_specific_preprocessing: Optional[Dict[int, Optional[str]]] = None


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


CLIENT_SPECIFIC_PREPROCESSINGS = {
    "histogram_shifts_high": {
        0: "histogram_shift_5",
        1: "histogram_shift_10",
        2: None,
    },
    "histogram_shifts_medium": {
        0: "histogram_shift_20",
        1: "histogram_shift_40",
        2: None,
    },
    "histogram_shifts_low": {
        0: "histogram_shift_50",
        1: "histogram_shift_100",
        2: None,
    },
    "bias_field_high": {
        0: "bias_field_05",
        1: "bias_field_08",
        2: None,
    },
    "bias_field_medium": {
        0: "bias_field_015",
        1: "bias_field_03",
        2: None,
    },
    "bias_field_low": {
        0: "bias_field_01",
        1: "bias_field_02",
        2: None,
    },
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
    "ks04_200r_1e_5c": {
        **create_fl_parameters(
            global_epochs=200, epochs_per_round=1, number_of_clients=5
        ),
        "partitioning_mode": "ks_stat",
        "partitioning_settings": {"file_name": "ks_stat_04_5c"},
    },
    "ks024_200r_1e_3c": {
        **create_fl_parameters(
            global_epochs=200, epochs_per_round=1, number_of_clients=3
        ),
        "partitioning_mode": "ks_stat",
        "partitioning_settings": {"file_name": "ks_stat_024_3c"},
    },
    "ks024_200r_1e_5c": {
        **create_fl_parameters(
            global_epochs=200, epochs_per_round=1, number_of_clients=5
        ),
        "partitioning_mode": "ks_stat",
        "partitioning_settings": {"file_name": "ks_stat_024_5c"},
    },
    "ks01_200r_1e_3c": {
        **create_fl_parameters(
            global_epochs=200, epochs_per_round=1, number_of_clients=3
        ),
        "partitioning_mode": "ks_stat",
        "partitioning_settings": {"file_name": "ks_stat_01_3c"},
    },
    "histogram_shifts_1r_1e_2c": {
        **create_fl_parameters(
            global_epochs=1, epochs_per_round=1, number_of_clients=2
        ),
        "partitioning_mode": "uniform",
        "client_specific_preprocessing": {
            0: "histogram_shift_5",
            1: "histogram_shift_10",
            2: None,
        },
    },
    "histogram_shifts_high_200r_1e_3c": {
        **create_fl_parameters(
            global_epochs=200, epochs_per_round=1, number_of_clients=3
        ),
        "partitioning_mode": "uniform",
        "client_specific_preprocessing": CLIENT_SPECIFIC_PREPROCESSINGS[
            "histogram_shifts_high"
        ],
    },
    "histogram_shifts_medium_200r_1e_3c": {
        **create_fl_parameters(
            global_epochs=200, epochs_per_round=1, number_of_clients=3
        ),
        "partitioning_mode": "uniform",
        "client_specific_preprocessing": CLIENT_SPECIFIC_PREPROCESSINGS[
            "histogram_shifts_medium"
        ],
    },
    "histogram_shifts_low_200r_1e_3c": {
        **create_fl_parameters(
            global_epochs=200, epochs_per_round=1, number_of_clients=3
        ),
        "partitioning_mode": "uniform",
        "client_specific_preprocessing": CLIENT_SPECIFIC_PREPROCESSINGS[
            "histogram_shifts_low"
        ],
    },
    "histogram_shifts_high_1r_200e_3c": {
        **create_fl_parameters(
            global_epochs=1, epochs_per_round=200, number_of_clients=3
        ),
        "partitioning_mode": "uniform",
        "client_specific_preprocessing": CLIENT_SPECIFIC_PREPROCESSINGS[
            "histogram_shifts_high"
        ],
    },
    "histogram_shifts_low_1r_200e_3c": {
        **create_fl_parameters(
            global_epochs=1, epochs_per_round=200, number_of_clients=3
        ),
        "partitioning_mode": "uniform",
        "client_specific_preprocessing": CLIENT_SPECIFIC_PREPROCESSINGS[
            "histogram_shifts_low"
        ],
    },
    "bias_field_high_1r_200e_3c": {
        **create_fl_parameters(
            global_epochs=1, epochs_per_round=200, number_of_clients=3
        ),
        "partitioning_mode": "uniform",
        "client_specific_preprocessing": CLIENT_SPECIFIC_PREPROCESSINGS[
            "bias_field_high"
        ],
    },
    "bias_field_medium_200r_1e_3c": {
        **create_fl_parameters(
            global_epochs=200, epochs_per_round=1, number_of_clients=3
        ),
        "partitioning_mode": "uniform",
        "client_specific_preprocessing": CLIENT_SPECIFIC_PREPROCESSINGS[
            "bias_field_medium"
        ],
    },
    "bias_field_low_200r_1e_3c": {
        **create_fl_parameters(
            global_epochs=200, epochs_per_round=1, number_of_clients=3
        ),
        "partitioning_mode": "uniform",
        "client_specific_preprocessing": CLIENT_SPECIFIC_PREPROCESSINGS[
            "bias_field_low"
        ],
    },
    "noniid_ci_run_central_400e": {
        "training_mode": "centralized",
        "epochs": 2,
        "client_specific_preprocessing": {
            0: "histogram_shift_5",
            1: "histogram_shift_10",
        },
    },
    "histogram_shifts_central_200e": {
        "training_mode": "centralized",
        "epochs": 200,
        "client_specific_preprocessing": CLIENT_SPECIFIC_PREPROCESSINGS[
            "histogram_shifts_high"
        ],
    },
    "bias_field_central_200e": {
        "training_mode": "centralized",
        "epochs": 200,
        "client_specific_preprocessing": CLIENT_SPECIFIC_PREPROCESSINGS[
            "bias_field_high"
        ],
    },
    "histogram_shifts_central1_200e": {
        "training_mode": "centralized",
        "epochs": 200,
        "client_specific_preprocessing": CLIENT_SPECIFIC_PREPROCESSINGS[
            "histogram_shifts_medium"
        ],
    },
    "histogram_shifts_central2_200e": {
        "training_mode": "centralized",
        "epochs": 200,
        "client_specific_preprocessing": CLIENT_SPECIFIC_PREPROCESSINGS[
            "histogram_shifts_low"
        ],
    },
    "bias_field_central1_200e": {
        "training_mode": "centralized",
        "epochs": 200,
        "client_specific_preprocessing": CLIENT_SPECIFIC_PREPROCESSINGS[
            "bias_field_medium"
        ],
    },
    "bias_field_central2_200e": {
        "training_mode": "centralized",
        "epochs": 200,
        "client_specific_preprocessing": CLIENT_SPECIFIC_PREPROCESSINGS[
            "bias_field_low"
        ],
    },
    **create_fl_specs("federated_1r_1e_2c"),
    **create_fl_specs("federated_1r_2e_3c"),
    **create_fl_specs("federated_1r_100e_3c"),
    **create_fl_specs("federated_1r_2e_3c"),
}


def get_training_specs(name: str):
    spec_dict = TRAINING_SPECS[name].copy()
    training_mode = spec_dict.pop("training_mode")

    if training_mode == "centralized":
        return CentralizedTrainingSpecs(**spec_dict)
    else:
        return FederatedTrainingSpecs(**spec_dict)
