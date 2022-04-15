from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from numpy.typing import ArrayLike, NDArray
from quantile_forest import RandomForestQuantileRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from ..inductive import ConformalQuantileRegressor
from ..utils import calculate_q_hat
from .base import ConformalSKLearnClassifier, SKLearnConformalPredictor


class ConformalRandomForestRegressor(SKLearnConformalPredictor, ConformalQuantileRegressor):

    _predictor: RandomForestQuantileRegressor
    q_hat_for_confidence_level_: Dict[float, float]
    calibration_nonconformity_scores_: Dict[float, NDArray]

    def __init__(self, predictor_params: dict = {}) -> None:

        if type(predictor_params) != dict:
            raise ValueError("'predictor_params' need to be dictionary of arguments.")

        super().__init__(
            RandomForestQuantileRegressor(**predictor_params), conditional=True, fit=True
        )

    def _get_or_calculate_q_hat_for_confidence_level(self, confidence_level: float) -> float:
        if q_hat := self.q_hat_for_confidence_level_.get(confidence_level, None):
            return q_hat
        else:
            lower_quantile, upper_quantile = self._calculate_lower_upper_quantiles(confidence_level)

            self.calibration_nonconformity_scores_[
                confidence_level
            ] = self._calculate_nonconformity_scores(
                y_hat=self._predictor.predict(
                    self._X_calibration, quantiles=[lower_quantile, 0.5, upper_quantile]
                ),
                y=self._y_calibration,
            )
            self.q_hat_for_confidence_level_[confidence_level] = calculate_q_hat(
                self.calibration_nonconformity_scores_[confidence_level], confidence_level
            )

            return self.q_hat_for_confidence_level_[confidence_level]

    def _fit_method(
        self,
        X: ArrayLike,
        calibration_size: float = 0.2,
        y: Optional[ArrayLike] = None,
        fit_params: dict = {},
        **kwargs: Dict[str, Any],
    ) -> None:

        if y is None:
            raise ValueError(
                "This implementation of 'ConformalQuantileRegressor' requires to be called with 'y'."  # noqa
            )

        X_training, X_calibration, y_training, y_calibration = self.check_and_split_X_y(
            X, y, calibration_size
        )

        self._predictor.fit(X_training, y_training, **fit_params)

        # bootstrap the necessary attributes
        self._X_calibration = X_calibration.copy()
        self._y_calibration = y_calibration.copy()
        self.calibration_nonconformity_scores_ = {}  # type: ignore
        self.q_hat_for_confidence_level_ = {}

    def _predict_and_calculate_half_interval(
        self,
        X: ArrayLike,
        confidence_level: Optional[float] = None,
        predict_params: dict = {},
        **kwargs: Dict[str, Any],
    ) -> Tuple[NDArray, float]:

        check_is_fitted(
            self, attributes=["q_hat_for_confidence_level_", "calibration_nonconformity_scores_"]
        )
        X = check_array(X, force_all_finite="allow-nan", estimator=self._predictor)

        if confidence_level is None:
            confidence_level = 0.9

        lower_quantile, upper_quantile = self._calculate_lower_upper_quantiles(confidence_level)

        y_hat_quantiles = self._predictor.predict(
            X, quantiles=[lower_quantile, 0.5, upper_quantile], **predict_params
        )
        q_hat = self._get_or_calculate_q_hat_for_confidence_level(confidence_level)

        return y_hat_quantiles, q_hat


class ConformalRandomForestClassifier(ConformalSKLearnClassifier):

    _predictor: RandomForestClassifier

    def __init__(
        self,
        conditional: bool = True,
        fit: bool = True,
        predictor_params: Dict[str, Any] = {},
    ) -> None:
        super().__init__(
            RandomForestClassifier(**predictor_params), conditional=conditional, fit=fit
        )
