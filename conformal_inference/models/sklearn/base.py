from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import check_array

from conformal_inference.models.inductive import ConformalClassifier, ConformalPredictor, ConformalRegressor
from conformal_inference.models.utils import calculate_q_hat, check_in_range

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray
    from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


class SKLearnConformalPredictor(ConformalPredictor):
    _predictor: BaseEstimator

    def check_and_split_X_y(
        self,
        X: ArrayLike,
        y: ArrayLike,
        calibration_size: float,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        check_in_range(calibration_size, "calibration_size")

        X = check_array(np.array(X), force_all_finite="allow-nan", estimator=self._predictor)

        if self._fit:
            X_training, X_calibration, y_training, y_calibration = train_test_split(X, y, test_size=calibration_size)

        else:
            X_training = X_calibration = X
            y_training = y_calibration = y

        return (
            X_training,
            X_calibration,
            y_training,
            y_calibration,
        )


class ConformalSKLearnClassifier(SKLearnConformalPredictor, ConformalClassifier):
    _predictor: ClassifierMixin

    def _fit_and_calculate_calibration_nonconformity_scores(
        self,
        X: ArrayLike,
        calibration_size: float = 0.2,
        y: ArrayLike | None = None,
        fit_params: dict = {},
        **kwargs: dict[str, Any],
    ) -> tuple[NDArray, NDArray]:
        if y is None:
            msg = "This implementation of 'ConformalClassifier' requires to be called with 'y'."
            raise ValueError(msg)

        X_training, X_calibration, y_training, y_calibration = self.check_and_split_X_y(X, y, calibration_size)

        if self.fit:
            self._predictor.fit(X_training, y_training, **fit_params)

        nonconformity_scores = self._calculate_nonconformity_scores(y_hat=self._predictor.predict_proba(X_calibration))

        return y_calibration, nonconformity_scores

    def _get_label_to_index_mapping(self) -> dict[Any, int]:
        return {x: index for index, x in enumerate(self._predictor.classes_)}

    def _predict_and_calculate_nonconformity_scores(
        self,
        X: ArrayLike,
        predict_params: dict = {},
        **kwargs: dict[str, Any],
    ) -> tuple[NDArray, NDArray]:
        X = check_array(X, force_all_finite="allow-nan", estimator=self._predictor)

        y_hat = self._predictor.predict_proba(X, **predict_params)
        nonconformity_scores = self._calculate_nonconformity_scores(y_hat=y_hat)

        return y_hat, nonconformity_scores


class ConformalSKLearnRegressor(SKLearnConformalPredictor, ConformalRegressor):
    _predictor: RegressorMixin

    def _fit_and_calculate_calibration_nonconformity_scores(
        self,
        X: ArrayLike,
        calibration_size: float = 0.2,
        y: ArrayLike | None = None,
        fit_params: dict = {},
        **kwargs: dict[str, Any],
    ) -> tuple[NDArray, NDArray]:
        if y is None:
            msg = "This implementation of 'ConformalRegressor' requires to be called with 'y'."
            raise ValueError(msg)

        X_training, X_calibration, y_training, y_calibration = self.check_and_split_X_y(X, y, calibration_size)

        if self.fit:
            self._predictor.fit(X_training, y_training, **fit_params)

        if self._conditional:
            self.variance_predictor = deepcopy(self._predictor)
            self.variance_predictor.fit(
                X_training,
                self._calculate_nonconformity_scores(y_training, self._predictor.predict(X_training)),
                **fit_params,
            )
            nonconformity_scores = self._calculate_nonconformity_scores(
                y_calibration,
                self._predictor.predict(X_calibration),
            ) / self.variance_predictor.predict(X_calibration)
        else:
            nonconformity_scores = self._calculate_nonconformity_scores(
                y=y_calibration,
                y_hat=self._predictor.predict(X_calibration),
            )

        return y_calibration, nonconformity_scores

    def _predict_and_calculate_half_interval(
        self,
        X: ArrayLike,
        confidence_level: float = 0.9,
        predict_params: dict = {},
        **kwargs: dict[str, Any],
    ) -> tuple[NDArray, float]:
        X = check_array(X, force_all_finite="allow-nan", estimator=self._predictor)
        y_hat = self._predictor.predict(X, **predict_params)
        q_hat = calculate_q_hat(self.calibration_nonconformity_scores_, confidence_level)

        if self._conditional:
            variance_prediction = self.variance_predictor.predict(X, **predict_params)

            half_interval = variance_prediction * q_hat
        else:
            half_interval = q_hat

        return y_hat, half_interval
