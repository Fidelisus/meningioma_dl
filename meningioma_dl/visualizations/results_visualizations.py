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

def plot_training_curve(validation_losses: Sequence[float], training_losses: Sequence[float], save_path: Path):
    fig = px.scatter(y=validation_losses, x=tuple(range(len(validation_losses))))
    fig.data[0].name="validation"
    fig.add_scatter(y=training_losses, x=tuple(range(len(training_losses))), name="training")
    fig.update_layout(title_text="<i><b>Training curve</b></i>")

    save_path.mkdir(parents=True, exist_ok=True)
    fig.write_html(save_path.joinpath("training_curve.html"))
