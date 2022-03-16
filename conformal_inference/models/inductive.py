from abc import ABC, abstractmethod
from copy import deepcopy
from logging import getLogger
from math import ceil
from typing import Any, Dict, Optional, Tuple, TypeVar, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, is_classifier, is_regressor
from sklearn.model_selection import train_test_split
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

logger = getLogger(__name__)

ICP = TypeVar("ICP", bound="InductiveConformalPredictor")


class InductiveConformalPredictor(ABC):
    """
    Inductive Conformal Predictors are originally described in Section 4.1 of:
        Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic Learning in a Random World.
    """

    calibration_nonconformity_scores_: Union[NDArray, Dict[Any, NDArray]]

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

    # TODO: max or min?
    @staticmethod
    def get_max_nonconformity(nonconformity_scores: NDArray, confidence_level: float) -> float:
        n = len(nonconformity_scores)
        k = ceil((n + 1) * confidence_level)
        max_nonconformity = nonconformity_scores[
            # `np.argpartition` promises that the k-th smallest number will be in its final
            # sorted position smaller on the left, larger on the right (not necessarily sorted)
            np.argpartition(nonconformity_scores, k)[k]
        ]

        return max_nonconformity

    @staticmethod
    @abstractmethod
    def _non_conformity_measure(y: Optional[NDArray], y_hat: NDArray) -> NDArray:
        pass

    @abstractmethod
    def fit(self, X: ArrayLike, y: ArrayLike, calibration_size: float = 0.2) -> ICP:
        pass

    @abstractmethod
    def predict(self, X: ArrayLike, confidence_level: float = 0.9) -> NDArray:
        pass


class _ClassifierICP(InductiveConformalPredictor):
    """
    TODO
    """

    @staticmethod
    def _non_conformity_measure(y: Optional[NDArray], y_hat: NDArray) -> NDArray:
        return 1 - y_hat

    def fit(
        self, X: ArrayLike, y: ArrayLike, calibration_size: float = 0.2, mondrian: bool = True
    ) -> ICP:

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
            max_nonconformity = self.get_max_nonconformity(
                self.calibration_nonconformity_scores_[label], confidence_level
            )

            prediction_sets[
                nonconformity_scores[:, index] < max_nonconformity, index
            ] = self.predictor.classes_[index]

        return prediction_sets


class _RegressorICP(InductiveConformalPredictor):
    """
    Algorithm described in:
        Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L. (2018).
        Distribution-free predictive inference for regression.
        Journal of the American Statistical Association, 113(523), 1094-1111.
    """

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
    ) -> ICP:

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
        check_is_fitted(
            self,
            attributes=["calibration_nonconformity_scores_", "mean_absolute_deviation_predictor"]
            if self._weighted
            else ["calibration_nonconformity_scores_"],
        )
        X = check_array(X, force_all_finite="allow-nan", estimator=self.predictor)

        y_hat = self.predictor.predict(X)
        one_sided_interval = self.get_max_nonconformity(
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
