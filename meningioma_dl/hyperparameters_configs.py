from typing import Tuple, Dict

SearchSpace = Dict[str, Dict[str, Tuple]]

FULL_SEARCH_SPACE: SearchSpace = {
    "rand_flip": {"spatial_axis": (0, 1), "prob": (0, 1)},
    "rand_rotate": {"prob": (0, 1)},
    "rand_zoom": {
        "min_zoom": (0.6, 1.0),
        "max_zoom": (1.0, 1.6),
        "prob": (0, 1),
    },
    "gaussian_noise": {
        "std": (0.01, 0.2),
        "prob": (0, 1),
    },
    "mask_after_gaussian": {},
}

AFFINE_TRANSFORMS_SEARCH_SPACE: SearchSpace = {
    "rand_flip": {"spatial_axis": (0, 1), "prob": (0, 1)},
    "rand_rotate": {"prob": (0, 1)},
    "rand_zoom": {
        "min_zoom": (0.6, 1.0),
        "max_zoom": (1.0, 1.6),
        "prob": (0, 1),
    },
}

SEARCH_SPACES: Dict[str, SearchSpace] = {
    "full": FULL_SEARCH_SPACE,
    "affine_transforms": AFFINE_TRANSFORMS_SEARCH_SPACE,
}
