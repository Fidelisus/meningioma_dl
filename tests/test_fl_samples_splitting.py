import numpy as np
import pandas as pd
import pytest

from meningioma_dl.federated_learning.create_federated_data_splits import (
    calculate_ks_stat_between_all_clients,
    get_best_split_with_given_ks_stat,
)


samples_df = pd.DataFrame(
    data={"labels": [0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2]}
)


@pytest.mark.parametrize(
    "desired_ks_stat,n_partitions", [(0.5, 3), (0.5, 5), (0.0, 3), (0.25, 3)]
)
def test_get_partitions_ks_stat(desired_ks_stat: float, n_partitions: int):
    partitions = get_best_split_with_given_ks_stat(
        samples_df["labels"], desired_ks_stat, n_partitions
    )
    ks_stat = calculate_ks_stat_between_all_clients(partitions)
    np.testing.assert_allclose(ks_stat, desired_ks_stat, atol=0.05)
