from dataclasses import dataclass
from typing import Optional, Dict

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
    "resnet_10_3_unfreezed": {
        "model_depth": 10,
        "resnet_shortcut_type": "B",
        "number_of_layers_to_unfreeze": 3,
    },
    "resnet_10_4_unfreezed": {
        "model_depth": 10,
        "resnet_shortcut_type": "B",
        "number_of_layers_to_unfreeze": 4,
    },
    "resnet_34_4_unfreezed": {
        "model_depth": 34,
        "resnet_shortcut_type": "B",
        "number_of_layers_to_unfreeze": 4,
    },
    "class_1_and_2_together_2_unfreezed": {
        "model_depth": 10,
        "resnet_shortcut_type": "B",
        "number_of_layers_to_unfreeze": 2,
        "number_of_classes": 2,
        "class_mapping": {1: 1, 2: 1, 3: 2},
        "evaluation_metric_weighting": "macro",
    },
    "class_2_and_3_together_2_unfreezed": {
        "model_depth": 10,
        "resnet_shortcut_type": "B",
        "number_of_layers_to_unfreeze": 2,
        "number_of_classes": 2,
        "class_mapping": {1: 1, 2: 2, 3: 2},
        "evaluation_metric_weighting": "macro",
    },
    "class_1_and_2_together_3_unfreezed": {
        "model_depth": 10,
        "resnet_shortcut_type": "B",
        "number_of_layers_to_unfreeze": 3,
        "number_of_classes": 2,
        "class_mapping": {1: 1, 2: 1, 3: 2},
        "evaluation_metric_weighting": "macro",
    },
    "class_2_and_3_together_1_unfreezed": {
        "model_depth": 10,
        "resnet_shortcut_type": "B",
        "number_of_layers_to_unfreeze": 1,
        "number_of_classes": 2,
        "class_mapping": {1: 1, 2: 2, 3: 2},
        "evaluation_metric_weighting": "macro",
    },
    "class_2_and_3_together_3_unfreezed": {
        "model_depth": 10,
        "resnet_shortcut_type": "B",
        "number_of_layers_to_unfreeze": 3,
        "number_of_classes": 2,
        "class_mapping": {1: 1, 2: 2, 3: 2},
        "evaluation_metric_weighting": "macro",
    },
    "class_2_and_3_together_4_unfreezed": {
        "model_depth": 10,
        "resnet_shortcut_type": "B",
        "number_of_layers_to_unfreeze": 4,
        "number_of_classes": 2,
        "class_mapping": {1: 1, 2: 2, 3: 2},
        "evaluation_metric_weighting": "macro",
    },
    "class_2_and_3_together_4_unfreezed_resnet_34": {
        "model_depth": 34,
        "resnet_shortcut_type": "B",
        "number_of_layers_to_unfreeze": 4,
        "number_of_classes": 2,
        "class_mapping": {1: 1, 2: 2, 3: 2},
        "evaluation_metric_weighting": "macro",
    },
}


@dataclass
class ModelSpecs:
    model_depth: int = 10
    resnet_shortcut_type: str = "B"
    number_of_layers_to_unfreeze: int = 0
    number_of_classes: int = 3
    class_mapping: Optional[Dict[int, int]] = None
    evaluation_metric_weighting: str = "weighted"

    def __post_init__(self):
        if self.number_of_classes not in {2, 3}:
            raise ValueError("number_of_classes must be 2 or 3")
        if self.number_of_classes == 2 and self.class_mapping is None:
            raise ValueError(
                "class_grouping needs to be defined if number_of_classes==2"
            )

    @classmethod
    def get_from_name(cls, name: str) -> Self:
        return cls(**MODELS[name])
