from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from quantile_forest import RandomForestQuantileRegressor
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from .base import InductiveConformalPredictor
from .utils import calculate_q_hat, check_in_range


class ConformalRandomForestRegressor(InductiveConformalPredictor):

    predictor: RandomForestQuantileRegressor
    calibration_nonconformity_scores_: Dict[float, NDArray]
    q_hat_for_confidence_level_: Dict[float, float]

    def __init__(self, forest_params: dict = {}) -> None:

        if type(forest_params) != dict:
            raise ValueError("'forest_params' need to be dictionary of arguments.")

        super().__init__(RandomForestQuantileRegressor(**forest_params), conditional=True, fit=True)

    @staticmethod
    def _calculate_nonconformity_scores(y_hat: NDArray, y: NDArray) -> NDArray:
        """
        Source:
            Romano, Y., Patterson, E., & Candès, E.J. (2019).
            Conformalized Quantile Regression. NeurIPS.
        """
        return np.stack((y_hat[:, 0] - y, y - y_hat[:, 1]), axis=1).max(axis=1)

    @staticmethod
    def _calculate_lower_upper_quantiles(confidence_level: float) -> Tuple[float, float]:
        miscoverage_level = 1 - confidence_level
        lower_quantile = miscoverage_level / 2
        upper_quantile = 1 - miscoverage_level / 2

        return lower_quantile, upper_quantile

    def _get_or_calculate_q_hat_for_confidence_level(self, confidence_level: float) -> float:
        if q_hat := self.q_hat_for_confidence_level_.get(confidence_level, None):
            return q_hat
        else:
            lower_quantile, upper_quantile = self._calculate_lower_upper_quantiles(confidence_level)

            self.calibration_nonconformity_scores_[
                confidence_level
            ] = self._calculate_nonconformity_scores(
                y_hat=self.predictor.predict(
                    self._X_calibration, quantiles=[lower_quantile, upper_quantile]
                ),
                y=self._y_calibration,
            )
            self.q_hat_for_confidence_level_[confidence_level] = calculate_q_hat(
                self.calibration_nonconformity_scores_[confidence_level], confidence_level
            )

            return self.q_hat_for_confidence_level_[confidence_level]

    def fit(
        self, X: ArrayLike, y: ArrayLike, calibration_size: float = 0.2
    ) -> ConformalRandomForestRegressor:
        X_training, X_calibration, y_training, y_calibration = self.check_and_split_X_y(
            X, y, calibration_size
        )

        self.predictor.fit(X_training, y_training)

        # bootstrap the necessary attributes
        self._X_calibration = X_calibration.copy()
        self._y_calibration = y_calibration.copy()
        self.calibration_nonconformity_scores_ = {}
        self.q_hat_for_confidence_level_ = {}
        self._get_or_calculate_q_hat_for_confidence_level(0.9)

        return self

    def predict(self, X: ArrayLike, confidence_level: float = 0.9) -> NDArray:
        check_in_range(confidence_level, "confidence_level")
        check_is_fitted(
            self, attributes=["calibration_nonconformity_scores_", "q_hat_for_confidence_level_"]
        )

        X = check_array(X, force_all_finite="allow-nan", estimator=self.predictor)

        lower_quantile, upper_quantile = self._calculate_lower_upper_quantiles(confidence_level)

        y_hat_quantiles = self.predictor.predict(X, quantiles=[lower_quantile, upper_quantile])
        q_hat = self._get_or_calculate_q_hat_for_confidence_level(0.9)

        y_hat_lower_bound = y_hat_quantiles[:, 0] - q_hat
        y_hat_upper_bound = y_hat_quantiles[:, 1] + q_hat

        return np.stack((y_hat_lower_bound, y_hat_upper_bound), axis=1)
