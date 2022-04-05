from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray


def __calculate_prediction_interval_sizes(predictions: NDArray) -> NDArray:
    return predictions[:, 1] - predictions[:, 0]


def coverage(predictions: NDArray, y_true: ArrayLike) -> float:
    y_true = np.array(y_true).ravel()
    y_in_prediction = (predictions[:, 0] <= y_true) & (y_true <= predictions[:, 1])
    return y_in_prediction.sum() / y_true.size


def mean_and_std_interval_length(predictions: NDArray) -> Tuple[float, float]:
    prediction_set_sizes = __calculate_prediction_interval_sizes(predictions)
    return (prediction_set_sizes.mean(), prediction_set_sizes.std())
