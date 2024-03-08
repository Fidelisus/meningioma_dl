import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import flwr as fl
import numpy as np
from flwr.common import Scalar, Parameters, EvaluateRes
from flwr.server.client_proxy import ClientProxy

from meningioma_dl.training_utils import _save_model


class SaveModelFedAvg(fl.server.server.FedAvg):
    saved_models_folder: Path
    trained_model_path: Optional[Path] = None
    best_model_f_score: float = 0.0
    last_model: Optional[Parameters] = None

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
            self.last_model = aggregated_parameters

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(
            rnd, results, failures
        )
        if metrics_aggregated["f_score"] > self.best_model_f_score:
            self.best_model_f_score = metrics_aggregated["f_score"]
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_weights(
                self.last_model
            )
            logging.info(
                f"Saving best model with the f_score of {self.best_model_f_score}. "
                f"Saving it at {self.saved_models_folder}"
            )
            self.trained_model_path = _save_model(
                aggregated_ndarrays,
                self.saved_models_folder,
                -1,  # -1 used to override previous best model
            )
        return loss_aggregated, metrics_aggregated
