from pathlib import Path
from typing import Dict, Optional, Tuple, List

import flwr as fl
import numpy as np
from flwr.common import Scalar, Parameters

from meningioma_dl.training_utils import _save_model


class SaveModelFedAvg(fl.server.strategy.FedAvg):
    saved_models_folder: Path
    trained_model_path: Optional[Path] = None

    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_weights(
                aggregated_parameters
            )
            self.trained_model_path = _save_model(
                aggregated_ndarrays,
                self.saved_models_folder,
                -1,  # -1 used to override previous best model
            )
        return aggregated_parameters, aggregated_metrics
