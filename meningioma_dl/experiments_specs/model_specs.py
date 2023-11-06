from dataclasses import dataclass

from typing_extensions import Self

MODELS = {
    "resnet_10_0_unfreezed": {
        "model_depth": 10,
        "resnet_shortcut_type": "B",
        "number_of_layers_to_unfreeze": 0,
    },
    "resnet_10_1_unfreezed": {
        "model_depth": 10,
        "resnet_shortcut_type": "B",
        "number_of_layers_to_unfreeze": 1,
    },
    "resnet_10_2_unfreezed": {
        "model_depth": 10,
        "resnet_shortcut_type": "B",
        "number_of_layers_to_unfreeze": 2,
    },
}


@dataclass
class ModelSpecs:
    model_depth: int = 10
    resnet_shortcut_type: str = "B"
    number_of_layers_to_unfreeze: int = 0
    number_of_classes: int = 3

    @classmethod
    def get_from_name(cls, name: str) -> Self:
        return cls(**MODELS[name])
