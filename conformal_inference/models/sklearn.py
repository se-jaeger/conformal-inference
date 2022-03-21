from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from logging import getLogger
from math import ceil
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, is_classifier, is_regressor
from sklearn.model_selection import train_test_split
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

logger = getLogger(__name__)


class InductiveConformalPredictor(ABC):
    """
    Inductive Conformal Predictors are originally described in Section 4.1 of:
        Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic Learning in a Random World.

    Other literature call this approach Split Conformal Predictors.
    """

    calibration_nonconformity_scores_: Union[NDArray, Dict[Any, NDArray]]

    def __init__(
        self,
        predictor: BaseEstimator,
        fit: bool = True,
    ) -> None:
        self._fit = fit
        self.predictor = deepcopy(predictor)

    def check_in_range(self, number: float, name: str, range: Tuple[int, int] = (0, 1)) -> None:
        if number < range[0] or number > range[1]:
            raise ValueError(f"Variable '{name}' is not valid! Need to be: 0 <= {name} <= 1")

    def check_and_split_X_y(
        self, X: ArrayLike, y: ArrayLike, calibration_size: float
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:

        self.check_in_range(calibration_size, "calibration_size")

        X = check_array(np.array(X), force_all_finite="allow-nan", estimator=self.predictor)
        y = np.array(y).ravel()  # make sure target is 1d array

        if self.fit:
            X_training, X_calibration, y_training, y_calibration = train_test_split(
                X, y, test_size=calibration_size
            )

        else:
            X_training = X_calibration = X
            y_training = y_calibration = y

        return (
            X_training,
            X_calibration,
            y_training,
            y_calibration,
        )

    @staticmethod
    def get_max_nonconformity_score(
        nonconformity_scores: NDArray, confidence_level: float
    ) -> float:

        # Since we use `np.argpartition` in the following: subtract 1 from `k` because it works
        # with indices, which start at 0 not 1.
        n = len(nonconformity_scores)
        k = ceil((n + 1) * confidence_level) - 1

        # mathematically, it is possible that `k` not in: 0 <= k <= n - 1, therefore, we clip it
        k = 0 if k < 0 else n - 1 if k >= n else k

        max_nonconformity_score = nonconformity_scores[
            # `np.argpartition` promises that the k-th smallest number will be in its final
            # sorted position smaller on the left, larger on the right (not necessarily sorted)
            np.argpartition(nonconformity_scores, k)[k]
        ]

        return max_nonconformity_score

    @staticmethod
    @abstractmethod
    def _non_conformity_measure(y: Optional[NDArray], y_hat: NDArray) -> NDArray:
        pass

    @abstractmethod
    def fit(
        self, X: ArrayLike, y: ArrayLike, calibration_size: float = 0.2
    ) -> InductiveConformalPredictor:
        pass

    @abstractmethod
    def predict(self, X: ArrayLike, confidence_level: float = 0.9) -> NDArray:
        pass


class InductiveConformalClassifier(InductiveConformalPredictor):
    """
    TODO
    """

    calibration_nonconformity_scores_: Dict[Any, NDArray]

    @staticmethod
    def _non_conformity_measure(y: Optional[NDArray], y_hat: NDArray) -> NDArray:
        return 1 - y_hat

    def fit(
        self, X: ArrayLike, y: ArrayLike, calibration_size: float = 0.2, mondrian: bool = True
    ) -> InductiveConformalPredictor:

        self._mondrian = mondrian

        X_training, X_calibration, y_training, y_calibration = self.check_and_split_X_y(
            X, y, calibration_size
        )

        if self.fit:
            self.predictor.fit(X_training, y_training)

        self.label_2_index_ = {x: index for index, x in enumerate(self.predictor.classes_)}
        non_conformity_scores = self._non_conformity_measure(
            y=y_calibration, y_hat=self.predictor.predict_proba(X_calibration)
        )

        if self._mondrian:
            self.calibration_nonconformity_scores_ = {
                label: non_conformity_scores[y_calibration == label, index]
                for label, index in self.label_2_index_.items()
            }
        else:
            self.calibration_nonconformity_scores_ = {
                label: non_conformity_scores[range(len(y_calibration)), index]
                for label, index in self.label_2_index_.items()
            }

        return self

    def predict(self, X: ArrayLike, confidence_level: float = 0.9) -> NDArray:
        self.check_in_range(confidence_level, "confidence_level")
        check_is_fitted(
            self,
            attributes=["calibration_nonconformity_scores_", "label_2_index_"],
        )
        X = check_array(X, force_all_finite="allow-nan", estimator=self.predictor)

        y_hat = self.predictor.predict_proba(X)
        nonconformity_scores = self._non_conformity_measure(None, y_hat=y_hat)

        # prepare prediction sets. If class is not included in prediction, use `np.nan`
        prediction_sets = np.empty(nonconformity_scores.shape)
        prediction_sets[:] = np.nan

        for label, index in self.label_2_index_.items():
            max_nonconformity_score = self.get_max_nonconformity_score(
                self.calibration_nonconformity_scores_[label], confidence_level
            )

            prediction_sets[
                nonconformity_scores[:, index] < max_nonconformity_score, index
            ] = self.predictor.classes_[index]

        return prediction_sets


class InductiveConformalRegressor(InductiveConformalPredictor):
    """
    Algorithm described in:
        Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L. (2018).
        Distribution-free predictive inference for regression.
        Journal of the American Statistical Association, 113(523), 1094-1111.
    """

    calibration_nonconformity_scores_: NDArray

    @staticmethod
    def _non_conformity_measure(y: Optional[NDArray], y_hat: NDArray) -> NDArray:
        return np.abs(y - y_hat)

    @staticmethod
    def _non_conformity_measure_weighted(
        y: ArrayLike, y_hat: NDArray, mean_absolute_deviation_hat: ArrayLike
    ) -> NDArray:
        return np.array(np.abs(y - y_hat) / mean_absolute_deviation_hat)

    def fit(
        self, X: ArrayLike, y: ArrayLike, calibration_size: float = 0.2, weighted: bool = False
    ) -> InductiveConformalPredictor:

        self._weighted = weighted

        X_training, X_calibration, y_training, y_calibration = self.check_and_split_X_y(
            X, y, calibration_size
        )

        if self.fit:
            self.predictor.fit(X_training, y_training)

        if self._weighted:
            self.mean_absolute_deviation_predictor = deepcopy(self.predictor)
            self.mean_absolute_deviation_predictor.fit(
                X_training,
                self._non_conformity_measure(y_training, self.predictor.predict(X_training)),
            )
            self.calibration_nonconformity_scores_ = self._non_conformity_measure_weighted(
                y_calibration,
                self.predictor.predict(X_calibration),
                self.mean_absolute_deviation_predictor.predict(X_calibration),
            )
        else:
            self.calibration_nonconformity_scores_ = self._non_conformity_measure(
                y=y_calibration, y_hat=self.predictor.predict(X_calibration)
            )

        return self

    def predict(self, X: ArrayLike, confidence_level: float = 0.9) -> NDArray:
        self.check_in_range(confidence_level, "confidence_level")
        check_is_fitted(
            self,
            attributes=["calibration_nonconformity_scores_", "mean_absolute_deviation_predictor"]
            if self._weighted
            else ["calibration_nonconformity_scores_"],
        )
        X = check_array(X, force_all_finite="allow-nan", estimator=self.predictor)

        y_hat = self.predictor.predict(X)
        one_sided_interval = self.get_max_nonconformity_score(
            self.calibration_nonconformity_scores_, confidence_level
        )

        if self._weighted:
            mean_absolute_deviation_hat = self.mean_absolute_deviation_predictor.predict(X)

            y_hat_upper_bound = y_hat + mean_absolute_deviation_hat * one_sided_interval
            y_hat_lower_bound = y_hat - mean_absolute_deviation_hat * one_sided_interval

        else:
            y_hat_upper_bound = y_hat + one_sided_interval
            y_hat_lower_bound = y_hat - one_sided_interval

        return np.stack((y_hat_lower_bound, y_hat_upper_bound), axis=1)


def InductiveConformalPredictorFactory(
    predictor: BaseEstimator,
    fit: bool = True,
) -> InductiveConformalPredictor:
    if not fit:
        check_is_fitted(predictor)

    if is_classifier(predictor):
        return InductiveConformalClassifier(
            predictor=predictor,
            fit=fit,
        )

    elif is_regressor(predictor):
        return InductiveConformalRegressor(
            predictor=predictor,
            fit=fit,
        )
    else:
        raise Exception("Only Classifiers or Regressors from `scikit-learn` are supported!")
