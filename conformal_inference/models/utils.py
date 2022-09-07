from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


def calculate_q_hat(nonconformity_scores: NDArray, confidence_level: float) -> Optional[float]:

    n = len(nonconformity_scores)

    if n == 0:
        return None

    quantile = confidence_level * (1 + 1 / n)

    # clip `quantile` to make sure it is in 0 <= quantile <= 1
    quantile = 1 if quantile > 1 else 0 if quantile < 0 else quantile

    return np.quantile(nonconformity_scores, quantile, interpolation="higher")


def check_in_range(number: float, name: str, range: Tuple[int, int] = (0, 1)) -> None:
    if number < range[0] or number > range[1]:
        raise ValueError(f"Variable '{name}' is not valid! Need to be: 0 <= {name} <= 1")
