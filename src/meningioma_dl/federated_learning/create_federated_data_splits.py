import copy
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.model_selection import StratifiedKFold

from meningioma_dl.data_loading.labels_loading import get_samples_df


def calculate_ks_stat_between_all_clients(
    clients_samples: Dict[int, np.ndarray]
) -> float:
    ks_stats = []
    client_ids = sorted(list(clients_samples.keys()))
    for first_client_idx in range(len(client_ids)):
        for second_client_idx in range(first_client_idx + 1, len(client_ids)):
            first_client_sample = clients_samples[client_ids[first_client_idx]]
            second_client_sample = clients_samples[client_ids[second_client_idx]]
            if first_client_sample.size == 0 or second_client_sample.size == 0:
                continue
            statistic, p_value = ks_2samp(first_client_sample, second_client_sample)
            ks_stats.append(statistic)
    return np.mean(ks_stats)


def _append_to_partition(
    partitions: Dict[int, np.ndarray], partition_id: int, item: int
) -> Dict[int, np.ndarray]:
    new_partitions = copy.deepcopy(partitions)
    new_partitions[partition_id] = np.append(new_partitions[partition_id], item)
    return new_partitions


def create_split_with_given_ks_stat(
    labels_series: pd.Series, desired_ks_stat: float, n_partitions: int, seed: int = 123
):
    np.random.seed(seed)
    partitions_labels = {
        partition_id: np.array([]) for partition_id in range(n_partitions)
    }
    partitions_indices = copy.deepcopy(partitions_labels)
    samples_ids = labels_series.index.to_list()
    np.random.shuffle(samples_ids)
    shortest_partition_len = 0

    partitions_labels = _append_to_partition(
        partitions_labels, 0, labels_series[samples_ids[0]]
    )
    partitions_indices = _append_to_partition(partitions_indices, 0, samples_ids[0])
    for sample_id in samples_ids[1:]:
        best_addition_ks_stat_diff = 1.0
        best_partition_where_to_add = None
        for partition_id in range(n_partitions):
            if len(partitions_labels[partition_id]) > shortest_partition_len:
                continue
            new_partitions = _append_to_partition(
                partitions_labels, partition_id, labels_series[sample_id]
            )
            new_ks_stat = calculate_ks_stat_between_all_clients(new_partitions)
            ks_stat_diff_of_new_partition = np.abs(new_ks_stat - desired_ks_stat)
            if ks_stat_diff_of_new_partition <= best_addition_ks_stat_diff:
                best_addition_ks_stat_diff = best_addition_ks_stat_diff
                best_partition_where_to_add = partition_id

        partitions_labels = _append_to_partition(
            partitions_labels, best_partition_where_to_add, labels_series[sample_id]
        )
        partitions_indices = _append_to_partition(
            partitions_indices, best_partition_where_to_add, sample_id
        )
        shortest_partition_len = min(
            [
                len(partitions_labels[partition_id])
                for partition_id in partitions_labels.keys()
            ]
        )
    return partitions_indices, partitions_labels


def get_best_split_with_given_ks_stat(
    labels_series: pd.Series,
    desired_ks_stat: float,
    n_partitions: int,
    bootstrap_rounds: int = 1000,
    manual_seed: int = 123,
) -> Dict[int, np.ndarray]:
    best_ks_stat_diff = 1.0
    best_partitions = None
    for i in range(bootstrap_rounds):
        partitions_indices, partitions_labels = create_split_with_given_ks_stat(
            labels_series, desired_ks_stat, n_partitions, seed=manual_seed + i
        )
        new_ks_stat = calculate_ks_stat_between_all_clients(partitions_labels)
        ks_stat_diff = np.abs(new_ks_stat - desired_ks_stat)
        if ks_stat_diff < best_ks_stat_diff:
            best_ks_stat_diff = ks_stat_diff
            best_partitions = partitions_indices
    logging.info(
        f"KS stat of the datasets is {best_ks_stat_diff+desired_ks_stat}. "
        f"Desired KS stat was {desired_ks_stat}"
    )
    return best_partitions


def get_uniform_client_partitions(
    samples_df: pd.Series, n_partitions: int, manual_seed: int
):
    partitions = {}
    splitter = StratifiedKFold(
        n_splits=n_partitions, shuffle=True, random_state=manual_seed
    )
    for i, (_, client_indexes) in enumerate(
        splitter.split(samples_df.index, samples_df)
    ):
        partitions[i] = client_indexes
    return partitions


def get_non_iid_partitions(file_path: Path, n_partitions: int):
    partitions = {}
    samples_df = get_samples_df(file_path)
    for partition in range(n_partitions):
        partitions[partition] = samples_df[samples_df["partition"] == partition].index
    return partitions
