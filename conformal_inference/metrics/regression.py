import numpy as np
from numpy.typing import ArrayLike, NDArray


def __calculate_prediction_interval_sizes(predictions: NDArray) -> NDArray:
    predictions = np.asarray(predictions)
    return predictions[:, 2] - predictions[:, 0]


def coverage(predictions: NDArray, y_true: ArrayLike) -> float:
    predictions = np.asarray(predictions)
    y_true = np.array(y_true).ravel()
    y_in_prediction = (predictions[:, 0] <= y_true) & (y_true <= predictions[:, 2])
    return y_in_prediction.sum() / y_true.size


def mean_and_std_interval_length(predictions: NDArray) -> tuple[float, float]:
    predictions = np.asarray(predictions)
    prediction_set_sizes = __calculate_prediction_interval_sizes(predictions)
    return (prediction_set_sizes.mean(), prediction_set_sizes.std())
