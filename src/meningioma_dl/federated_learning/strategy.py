from pathlib import Path
from typing import Callable

from meningioma_dl.experiments_specs.fl_strategy_specs import FLStrategySpecs
from meningioma_dl.federated_learning.server import (
    SaveModelFedAvg,
    FedProx,
)
from meningioma_dl.localized_federated_ensemble.server import FedEnsemble


def create_strategy(
    fl_strategy_specs: FLStrategySpecs,
    saved_models_folder: Path,
    fit_metrics_aggregation_fn: Callable,
    evaluate_metrics_aggregation_fn: Callable,
    on_fit_config_fn: Callable,
) -> SaveModelFedAvg:
    if fl_strategy_specs.name == "fed_avg":
        strategy = SaveModelFedAvg(
            fraction_fit=fl_strategy_specs.config.get("fraction_fit", 1.0),
            fraction_evaluate=fl_strategy_specs.config.get("fraction_eval", 1.0),
            accept_failures=False,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            on_fit_config_fn=on_fit_config_fn,
        )
    elif fl_strategy_specs.name == "fed_prox":
        strategy = FedProx(
            fraction_fit=fl_strategy_specs.config.get("fraction_fit", 1.0),
            fraction_evaluate=fl_strategy_specs.config.get("fraction_eval", 1.0),
            accept_failures=False,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            on_fit_config_fn=on_fit_config_fn,
            proximal_mu=fl_strategy_specs.config["proximal_mu"],
        )
    elif fl_strategy_specs.name == "fed_ensemble":
        strategy = FedEnsemble(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            accept_failures=False,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            on_fit_config_fn=on_fit_config_fn,
        )
    else:
        raise KeyError(f"Strategy named {fl_strategy_specs.name} not supported")
    strategy.saved_models_folder = saved_models_folder
    return strategy
