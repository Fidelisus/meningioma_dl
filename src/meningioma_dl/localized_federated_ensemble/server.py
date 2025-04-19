import logging
from logging import WARNING
from pathlib import Path
from typing import List, Tuple, Union, Optional, Dict

import flwr as fl
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
    log,
)
from flwr.server.client_proxy import ClientProxy

from meningioma_dl.model_training.training_loop import save_model


class FedEnsemble(fl.server.server.FedAvg):
    saved_models_folder: Path

    def __repr__(self) -> str:
        return "FedEnsemble()"

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            raise RuntimeError(
                f"There are failures during fitting, aborting {failures}"
            )

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        clients_parameters: Dict[int, Parameters] = {}
        for client_id, weights in enumerate(weights_results):
            parameters = ndarrays_to_parameters(weights[0])
            clients_parameters[client_id] = parameters

        logging.info(
            f"Saving best models for {len(weights_results)} clients. "
            f"Saving it at {self.saved_models_folder}"
        )
        for client_id, parameters in clients_parameters.items():
            save_model(
                model=parameters_to_ndarrays(parameters),
                model_save_folder=self.saved_models_folder,
                file_name_suffix=str(client_id),
            )

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return None, metrics_aggregated
