from unittest import TestCase

import numpy as np
from sklearn.metrics import confusion_matrix

from meningioma_dl.visualizations.results_visualizations import (
    calculate_sensitivity_and_specificity,
)


def test_create_confusion_matrix():
    """
    Structure:
            pred1 pred2 pred3
    real1
    real2
    real3
    """
    true = np.array([1, 2, 2, 0, 0, 0, 2, 2, 2])
    predictions = np.array([1, 0, 2, 0, 0, 1, 2, 2, 0])
    expected = np.array(
        [
            [2, 1, 0],
            [0, 1, 0],
            [2, 0, 3],
        ]
    )

    returned = confusion_matrix(true, predictions)

    assert np.allclose(returned, expected)


def test_calculate_sensitivity_and_specificity():
    true = np.array([1, 2, 2, 0, 0, 0, 2, 2, 2])
    predictions = np.array([1, 0, 2, 0, 0, 1, 2, 2, 0])
    expected_sensitivity = {"0": 2 / (2 + 1), "1": 1 / 1, "2": 3 / (3 + 2)}
    expected_specificity = {"0": 4 / (4 + 2), "1": 7 / (7 + 1), "2": 4 / 4}

    returned_sensitivity, returned_specificity = calculate_sensitivity_and_specificity(
        true, predictions, 3
    )

    TestCase().assertDictEqual(returned_sensitivity, expected_sensitivity)
    TestCase().assertDictEqual(returned_specificity, expected_specificity)
