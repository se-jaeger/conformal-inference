from abc import ABC, abstractmethod
from copy import deepcopy
from logging import getLogger
from math import ceil
from typing import Tuple, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, is_classifier, is_regressor
from sklearn.model_selection import train_test_split
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

logger = getLogger(__name__)

ICP = TypeVar("ICP", bound="InductiveConformalPredictor")


# Also known as: Split Conformal Prediction Sets. Source:
# Lei, J., G’Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L. (2018).
# Distribution-free predictive inference for regression.
# Journal of the American Statistical Association, 113(523), 1094-1111.
class InductiveConformalPredictor(ABC):
    calibration_nonconformity_scores_: NDArray

    def __init__(
        self,
        predictor: BaseEstimator,
        fit: bool = True,
    ) -> None:
        self._fit = fit
        self.predictor = deepcopy(predictor)

    def check_and_split_X_y(
        self, X: ArrayLike, y: ArrayLike, calibration_size: float
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:

        if calibration_size >= 1 or calibration_size <= 0:
            raise ValueError("'calibration_size' not valid! Need to be: 0 < calibration_size < 1")

        X = check_array(X, force_all_finite="allow-nan", estimator=self.predictor)

        if self.fit:
            X_training, X_calibration, y_training, y_calibration = train_test_split(
                X, y, test_size=calibration_size
            )

        else:
            X_training = X_calibration = X
            y_training = y_calibration = y

        return (
            np.array(X_training),
            np.array(X_calibration),
            np.array(y_training),
            np.array(y_calibration),
        )

    @staticmethod
    def _calculate_p_values(alphas: NDArray) -> NDArray:
        n_alphas = len(alphas)
        # TODO: speedup possible?
        return np.array([(alphas >= alpha_i).sum() / n_alphas for alpha_i in alphas])

    @abstractmethod
    def fit(
        self, X: ArrayLike, y: ArrayLike, calibration_size: float = 0.8, weighted: bool = False
    ) -> ICP:
        pass

    @abstractmethod
    def predict(self, X: ArrayLike, confidence_level: float = 0.9) -> NDArray:
        pass


class _ClassifierICP(InductiveConformalPredictor):
    def fit(
        self, X: ArrayLike, y: ArrayLike, calibration_size: float = 0.8, weighted: bool = False
    ) -> ICP:

        return self

    def predict(self, X: ArrayLike, confidence_level: float = 0.9) -> ArrayLike:
        check_is_fitted(
            self,
            attributes=[
                "todo",
                "todo",
            ],
        )


class _RegressorICP(InductiveConformalPredictor):
    """
    Algorithm from:
    Lei, J., G’Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L. (2018).
    Distribution-free predictive inference for regression.
    Journal of the American Statistical Association, 113(523), 1094-1111.
    """

    @staticmethod
    def _non_conformity_measure(y: NDArray, y_hat: NDArray) -> NDArray:
        return np.abs(y - y_hat)

    @staticmethod
    def _non_conformity_measure_weighted(
        y: ArrayLike, y_hat: ArrayLike, mean_absolute_deviation_hat: ArrayLike
    ) -> NDArray:
        return np.array(np.abs(y - y_hat) / mean_absolute_deviation_hat)

    def fit(
        self, X: ArrayLike, y: ArrayLike, calibration_size: float = 0.8, weighted: bool = False
    ) -> ICP:

        self.weighted_ = weighted

        X_training, X_calibration, y_training, y_calibration = self.check_and_split_X_y(
            X, y, calibration_size
        )

        if self.fit:
            self.predictor.fit(X_training, y_training)

        if self.weighted_:
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
                y_calibration, self.predictor.predict(X_calibration)
            )

        return self

    def predict(self, X: ArrayLike, confidence_level: float = 0.9) -> NDArray:
        check_is_fitted(
            self,
            attributes=["calibration_nonconformity_scores_", "mean_absolute_deviation_predictor"]
            if self.weighted_
            else ["calibration_nonconformity_scores_"],
        )
        X = check_array(X, force_all_finite="allow-nan", estimator=self.predictor)

        y_hat = self.predictor.predict(X)
        n = len(self.calibration_nonconformity_scores_)
        k = ceil((n + 1) * confidence_level)

        one_sided_interval = self.calibration_nonconformity_scores_[
            # `np.argpartition` promises that the k-th smallest number will be in its final
            # sorted position smaller on the left, larger on the right (not necessarily sorted)
            np.argpartition(self.calibration_nonconformity_scores_, k)[k]
        ]

        if self.weighted_:
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
        return _ClassifierICP(
            predictor=predictor,
            fit=fit,
        )

    elif is_regressor(predictor):
        return _RegressorICP(
            predictor=predictor,
            fit=fit,
        )
    else:
        raise Exception("Only Classifiers or Regressors from `scikit-learn` are supported!")
