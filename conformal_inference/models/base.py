from __future__ import annotations

from abc import ABC
from copy import deepcopy
from logging import getLogger
from typing import Any, Dict, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, is_classifier, is_regressor
from sklearn.model_selection import train_test_split
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from .utils import calculate_q_hat, check_in_range

logger = getLogger(__name__)


class InductiveConformalPredictor(ABC):
    """
    Inductive Conformal Predictors are originally described in Section 4.1 of:
        Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic Learning in a Random World.

    Other literature refer to this approach as Split Conformal Predictors.
    """

    calibration_nonconformity_scores_: Union[NDArray, Dict[Any, NDArray]]

    def __init__(
        self,
        predictor: BaseEstimator,
        conditional: bool = True,
        fit: bool = True,
    ) -> None:
        self._conditional = conditional
        self._fit = fit
        self.predictor = deepcopy(predictor)

    def check_and_split_X_y(
        self, X: ArrayLike, y: ArrayLike, calibration_size: float
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        check_in_range(calibration_size, "calibration_size")

        X = check_array(np.array(X), force_all_finite="allow-nan", estimator=self.predictor)
        y = np.array(y).ravel()  # make sure target is 1d array

        if self._fit:
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


class ConformalClassifier(InductiveConformalPredictor):
    """
    TODO
    """

    calibration_nonconformity_scores_: Dict[Any, NDArray]

    @staticmethod
    def _calculate_nonconformity_scores(y_hat: NDArray) -> NDArray:
        return 1 - y_hat

    def fit(
        self, X: ArrayLike, y: ArrayLike, calibration_size: float = 0.2
    ) -> InductiveConformalPredictor:

        X_training, X_calibration, y_training, y_calibration = self.check_and_split_X_y(
            X, y, calibration_size
        )

        if self.fit:
            self.predictor.fit(X_training, y_training)

        self.label_2_index_ = {x: index for index, x in enumerate(self.predictor.classes_)}
        nonconformity_scores = self._calculate_nonconformity_scores(
            y_hat=self.predictor.predict_proba(X_calibration)
        )

        if self._conditional:
            self.calibration_nonconformity_scores_ = {
                label: nonconformity_scores[y_calibration == label, index]
                for label, index in self.label_2_index_.items()
            }
        else:
            self.calibration_nonconformity_scores_ = {
                label: nonconformity_scores[range(len(y_calibration)), index]
                for label, index in self.label_2_index_.items()
            }

        return self

    def predict(self, X: ArrayLike, confidence_level: float = 0.9) -> NDArray:
        check_in_range(confidence_level, "confidence_level")
        check_is_fitted(
            self,
            attributes=["calibration_nonconformity_scores_", "label_2_index_"],
        )

        X = check_array(X, force_all_finite="allow-nan", estimator=self.predictor)

        y_hat = self.predictor.predict_proba(X)
        nonconformity_scores = self._calculate_nonconformity_scores(y_hat=y_hat)

        # prepare prediction sets. If class is not included in prediction, use `np.nan`
        prediction_sets = np.empty(nonconformity_scores.shape)
        prediction_sets[:] = np.nan

        for label, index in self.label_2_index_.items():
            q_hat = calculate_q_hat(self.calibration_nonconformity_scores_[label], confidence_level)

            prediction_sets[nonconformity_scores[:, index] < q_hat, index] = label

        return prediction_sets


class ConformalRegressor(InductiveConformalPredictor):
    """
    Algorithm described in:
        Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L. (2018).
        Distribution-free predictive inference for regression.
        Journal of the American Statistical Association, 113(523), 1094-1111.
    """

    calibration_nonconformity_scores_: NDArray

    @staticmethod
    def _calculate_nonconformity_scores(y: NDArray, y_hat: NDArray) -> NDArray:
        return np.abs(y - y_hat)

    def fit(
        self, X: ArrayLike, y: ArrayLike, calibration_size: float = 0.2
    ) -> InductiveConformalPredictor:

        X_training, X_calibration, y_training, y_calibration = self.check_and_split_X_y(
            X, y, calibration_size
        )

        if self.fit:
            self.predictor.fit(X_training, y_training)

        if self._conditional:
            self.uncertainty_predictor = deepcopy(self.predictor)
            self.uncertainty_predictor.fit(
                X_training,
                self._calculate_nonconformity_scores(
                    y_training, self.predictor.predict(X_training)
                ),
            )
            self.calibration_nonconformity_scores_ = self._calculate_nonconformity_scores(
                y_calibration, self.predictor.predict(X_calibration)
            ) / self.uncertainty_predictor.predict(X_calibration)
        else:
            self.calibration_nonconformity_scores_ = self._calculate_nonconformity_scores(
                y=y_calibration, y_hat=self.predictor.predict(X_calibration)
            )

        return self

    def predict(self, X: ArrayLike, confidence_level: float = 0.9) -> NDArray:
        check_in_range(confidence_level, "confidence_level")
        check_is_fitted(self, attributes=["calibration_nonconformity_scores_"])

        X = check_array(X, force_all_finite="allow-nan", estimator=self.predictor)

        y_hat = self.predictor.predict(X)
        q_hat = calculate_q_hat(self.calibration_nonconformity_scores_, confidence_level)

        if self._conditional:
            uncertainty_prediction = self.uncertainty_predictor.predict(X)

            y_hat_upper_bound = y_hat + uncertainty_prediction * q_hat
            y_hat_lower_bound = y_hat - uncertainty_prediction * q_hat

        else:
            y_hat_upper_bound = y_hat + q_hat
            y_hat_lower_bound = y_hat - q_hat

        return np.stack((y_hat_lower_bound, y_hat_upper_bound), axis=1)


def ConformalPredictor(
    predictor: BaseEstimator,
    conditional: bool = True,
    fit: bool = True,
) -> InductiveConformalPredictor:
    if not fit:
        check_is_fitted(predictor)

    if is_classifier(predictor):
        return ConformalClassifier(
            predictor=predictor,
            conditional=conditional,
            fit=fit,
        )

    elif is_regressor(predictor):
        return ConformalRegressor(
            predictor=predictor,
            conditional=conditional,
            fit=fit,
        )
    else:
        raise Exception("Only Classifiers or Regressors from `scikit-learn` are supported!")
