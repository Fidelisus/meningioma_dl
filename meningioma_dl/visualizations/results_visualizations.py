from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn import metrics
import plotly.figure_factory as ff
import plotly.express as px


def plot_confusion_matrix(labels: np.array, predictions: np.array, save_path: Path):
    print(labels)
    print(predictions)
    confusion_matrix = metrics.confusion_matrix(labels, predictions)
    confusion_matrix = confusion_matrix.astype(int)
    print(confusion_matrix)
    fig = ff.create_annotated_heatmap(
        confusion_matrix,
        x=["predicted 1", "predicted 2", "predicted 3"],
        y=["real 1", "real 2", "real 3"],
        colorscale="Viridis",
    )

    fig.update_layout(title_text="<i><b>Confusion matrix</b></i>")

    fig["data"][0]["showscale"] = True
    save_path.mkdir(parents=True, exist_ok=True)
    fig.write_html(save_path.joinpath("confusion_matrix.html"))


def plot_training_curve(
    validation_losses: Sequence[float],
    training_losses: Sequence[float],
    f_scores: Sequence[float],
    cohen_kappa_scores: Sequence[float],
    save_path: Path,
):
    fig = px.scatter()
    fig.add_scatter(
        y=training_losses,
        x=tuple(range(len(training_losses))),
        name="training",
        mode="markers",
    )
    fig.add_scatter(
        y=validation_losses,
        x=tuple(range(len(training_losses))),
        name="validation",
        mode="markers",
    )
    fig.add_scatter(
        y=f_scores,
        x=tuple(range(len(training_losses))),
        name="f_scores",
        mode="markers",
    )
    fig.add_scatter(
        y=cohen_kappa_scores,
        x=tuple(range(len(training_losses))),
        name="cohen_kappa_scores",
        mode="markers",
    )
    fig.update_layout(title_text="<i><b>Learning curve</b></i>")

    save_path.mkdir(parents=True, exist_ok=True)
    fig.write_html(save_path.joinpath("learning_curve.html"))
