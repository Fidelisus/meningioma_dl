from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


def visualize_images(
    data: np.array,
    save_dir: Optional[Path] = None,
    images_names: Optional[Sequence[Union[Path, str]]] = None,
):
    if len(images_names) != data.shape[0]:
        raise ValueError(
            f"Number of images {data.shape[0]} should be the same as len of {images_names}"
        )
    for image_with_channel_dimension, image_name in zip(data, images_names):
        image = image_with_channel_dimension[0]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        sagittal_slice = image[:, image.shape[1] // 2, :]
        axes[0].imshow(sagittal_slice, cmap="gray")
        axes[0].set_title("Sagittal Slice")
        axes[0].axis("off")

        coronal_slice = image[image.shape[0] // 2, :, :]
        axes[1].imshow(coronal_slice, cmap="gray")
        axes[1].set_title("Coronal Slice")
        axes[1].axis("off")

        axial_slice = image[:, :, image.shape[2] // 2]
        axes[2].imshow(axial_slice, cmap="gray")
        axes[2].set_title("Axial Slice")
        axes[2].axis("off")

        if save_dir is not None:
            image_path = save_dir.joinpath(image_name)
            image_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(image_path))
        else:
            fig.show()
        fig.clf()


def create_images_errors_report(
    data_loader: DataLoader,
    images_paths: Sequence[Path],
    predictions: np.ndarray,
    directory: Path,
) -> None:
    images_with_predictions = {
        image: prediction.item()
        for image, prediction in zip(images_paths, predictions.data + 1)
    }
    for batch_data in data_loader:
        saved_images_paths = []
        for file, label in zip(
            batch_data["img_meta_dict"]["filename_or_obj"], batch_data["label"]
        ):
            saved_images_paths.append(
                Path(
                    f"label_{images_with_predictions[Path(file)]}",
                    f"{Path(file).stem.split('.')[0]}_label_{label.data + 1}",
                )
            )

        visualize_images(batch_data["img"], directory, saved_images_paths)
