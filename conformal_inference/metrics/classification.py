from typing import Tuple

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray


def __calculate_prediction_set_sizes(predictions: NDArray) -> NDArray:
    return np.count_nonzero(~pd.isna(predictions), axis=1)


def coverage(predictions: NDArray, y_true: ArrayLike) -> float:
    y_true = np.array(y_true)
    y_in_prediction = [a in b for a, b in zip(y_true, predictions)]
    return sum(y_in_prediction) / y_true.size


def mean_and_std_prediction_set_size(predictions: NDArray) -> Tuple[float, float]:
    prediction_set_sizes = __calculate_prediction_set_sizes(predictions)
    return (prediction_set_sizes.mean(), prediction_set_sizes.std())
