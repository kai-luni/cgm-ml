from typing import Union
from cgmml.common.model_utils.model_utils_constants import BLACKLIST_QRCODES

import numpy as np


def preprocess_depthmap(depthmap):
    return depthmap.astype("float32")


def preprocess_targets(targets: Union[list, dict],
                       target_indices: list = None,
                       target_names: list = None) -> np.ndarray:
    assert (target_indices is not None) != (target_names is not None), (target_indices, target_names)  # xor
    if target_indices is not None:
        targets = targets[target_indices]
    elif target_names is not None:
        targets = [targets[target_name] for target_name in target_names]
    return np.array(targets).astype("float32")


def filter_blacklisted_persons(person_paths):
    person_paths_filtered = []
    assert len(person_paths) != 0, 'The provided person_path is empty'
    for person_path in person_paths:
        person_str = person_path.split('/')[-1]
        assert '-' in person_str and len(person_str) in [21, 36], person_str
        if person_str in BLACKLIST_QRCODES:
            continue
        person_paths_filtered.append(person_path)
    return person_paths_filtered
