from typing import Callable, List, NamedTuple, Tuple, Union

import h5py
import numpy as np

# Need own implementation of jaccard because scipy's
# implementation is different


def jaccard(a: List[int], b: List[int]) -> float:
    if len(a) == 0 or len(b) == 0:
        return 0
    intersect = len(set(a) & set(b))
    return intersect / (float)(len(a) + len(b) - intersect)


def norm(a):
    return np.sum(a**2) ** 0.5


def euclidean(a, b):
    return norm(a - b)

class Metric(NamedTuple):
    distance: Callable[[np.ndarray, np.ndarray], float]
    distance_valid: Callable[[float], bool]

metrics = {
    "hamming": Metric(
        distance=lambda a, b: np.mean(a.astype(np.bool_) ^ b.astype(np.bool_)),
        distance_valid=lambda a: True
    ),
    "jaccard": Metric(
        distance=lambda a, b: 1 - jaccard(a, b),
        distance_valid=lambda a: a < 1 - 1e-5
    ),
    "euclidean": Metric(
        distance=lambda a, b: euclidean(a, b),
        distance_valid=lambda a: True
    ),
    "angular": Metric(
        distance=lambda a, b: 1 - np.dot(a, b) / (norm(a) * norm(b)),
        distance_valid=lambda a: True
    ),
}

def compute_distance(metric: str, a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the distance between two points according to a specified metric.

    Args:
        metric (str): The name of the metric to use. Must be a key in the 'metrics' dictionary.
        a (np.ndarray): The first point.
        b (np.ndarray): The second point.

    Returns:
        float: The computed distance.

    Raises:
        KeyError: If the specified metric is not found in the 'metrics' dictionary.
    """
    if metric not in metrics:
        raise KeyError(f"Unknown metric '{metric}'. Known metrics are {list(metrics.keys())}")

    return metrics[metric].distance(a, b)


def is_distance_valid(metric: str, distance: float) -> bool:
    """
    Check if a computed distance is valid according to a specified metric.

    Args:
        metric (str): The name of the metric to use. Must be a key in the 'metrics' dictionary.
        distance (float): The computed distance to check.

    Returns:
        bool: True if the distance is valid, False otherwise.

    Raises:
        KeyError: If the specified metric is not found in the 'metrics' dictionary.
    """
    if metric not in metrics:
        raise KeyError(f"Unknown metric '{metric}'. Known metrics are {list(metrics.keys())}")

    return metrics[metric].distance_valid(distance)


def convert_sparse_to_list(data: np.ndarray, lengths: List[int]) -> List[np.ndarray]:
    """
    Converts sparse data into a list of arrays, where each array represents a separate data sample.

    Args:
        data (np.ndarray): The input sparse data represented as a numpy array.
        lengths (List[int]): List of lengths for each data sample in the sparse data.

    Returns:
        List[np.ndarray]: A list of arrays where each array is a data sample.
    """
    return [
        data[i - l : i] for i, l in zip(np.cumsum(lengths), lengths)
    ]


def dataset_transform(train_data, test_data, distance: str = None):
    """
    Transforms the dataset from the HDF5 format or numpy array to conventional numpy format.

    If the dataset is dense, it's returned as a numpy array.
    If it's sparse, it's transformed into a list of numpy arrays, each representing a data sample.

    Args:
        train_data: Training data (h5py.Dataset or np.ndarray)
        test_data: Testing data (h5py.Dataset or np.ndarray)
        distance: Distance metric string (optional)

    Returns:
        Tuple[Union[np.ndarray, List[np.ndarray]], Union[np.ndarray, List[np.ndarray]]]: Tuple of training and testing data in conventional format.
    """
    # 如果已经是 numpy 数组，直接返回
    if isinstance(train_data, np.ndarray):
        return train_data, test_data
    
    # 否则从 HDF5 数据集中读取
    if hasattr(train_data, 'attrs') and train_data.attrs.get("type", "dense") != "sparse":
        return np.array(train_data["train"]), np.array(train_data["test"])

    # we store the dataset as a list of integers, accompanied by a list of lengths in hdf5
    # so we transform it back to the format expected by the algorithms here (array of array of ints)
    return (
        convert_sparse_to_list(train_data["train"], train_data["size_train"]),
        convert_sparse_to_list(test_data["test"], test_data["size_test"])
    )
