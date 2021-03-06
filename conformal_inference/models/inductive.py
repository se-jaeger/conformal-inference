from __future__ import annotations

from abc import ABC, abstractmethod
from logging import getLogger
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.utils.validation import check_is_fitted

from .utils import calculate_q_hat, check_in_range

logger = getLogger(__name__)


class ConformalPredictor(ABC):
    """
    Inductive Conformal Predictors are originally described in Section 4.1 of:
        Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic Learning in a Random World.

    Other literature refer to this approach as Split Conformal Predictors.
    """

    _fit: bool
    _predictor: Any
    _conditional: bool
    calibration_nonconformity_scores_: Union[NDArray, Dict[Any, NDArray]]

    def __init__(
        self,
        predictor: Any,
        conditional: bool = True,
        fit: bool = True,
    ) -> None:

        if type(conditional) != bool:
            raise ValueError("'conditional' need to be of type bool.")

        if type(fit) != bool:
            raise ValueError("'fit' need to be of type bool.")

        self._fit = fit
        self._predictor = predictor
        self._conditional = conditional

    @abstractmethod
    def fit(
        self,
        X: ArrayLike,
        calibration_size: float = 0.2,
        y: Optional[ArrayLike] = None,
        fit_params: dict = {},
        **kwargs: Dict[str, Any],
    ) -> ConformalPredictor:

        check_in_range(calibration_size, "calibration_size")

        if type(fit_params) != dict:
            raise ValueError("'fit_params' need to be dictionary of arguments.")

        return self

    @abstractmethod
    def predict(  # type: ignore[return]
        self,
        X: ArrayLike,
        confidence_level: float = 0.9,
        predict_params: dict = {},
        **kwargs: Dict[str, Any],
    ) -> NDArray:
        check_in_range(confidence_level, "confidence_level")
        check_is_fitted(self, attributes=["calibration_nonconformity_scores_"])

        if type(predict_params) != dict:
            raise ValueError("'predict_params' need to be dictionary of arguments.")


class ConformalClassifier(ConformalPredictor):
    """
    TODO
    """

    label_to_index_: Dict[Any, int]
    index_to_label_: Dict[int, Any]
    _labels_dtype: Optional[Any] = None
    calibration_nonconformity_scores_: Dict[Any, NDArray]

    @staticmethod
    def _calculate_nonconformity_scores(y_hat: NDArray) -> NDArray:
        return 1 - y_hat

    def _create_numpy_array_for_labels_dtype(self, shape: Tuple[int, ...]) -> NDArray:
        if self._labels_dtype is None:
            are_labels_numerical = list(
                {
                    isinstance(x, (int, float, complex)) and not isinstance(x, bool)
                    for x in self.label_to_index_.keys()
                }
            )

            if len(are_labels_numerical) > 1 or not are_labels_numerical[0]:
                self._labels_dtype = object

            else:
                self._labels_dtype = float

        return np.full(shape, np.nan, dtype=self._labels_dtype)

    def fit(
        self,
        X: ArrayLike,
        calibration_size: float = 0.2,
        y: Optional[ArrayLike] = None,
        fit_params: dict = {},
        **kwargs: Dict[str, Any],
    ) -> ConformalClassifier:

        super().fit(
            X=X, calibration_size=calibration_size, y=y, fit_params=fit_params, kwargs=kwargs
        )

        (
            y_calibration,
            nonconformity_scores,
        ) = self._fit_and_calculate_calibration_nonconformity_scores(
            X=X, calibration_size=calibration_size, y=y, fit_params=fit_params, kwargs=kwargs
        )

        self.label_to_index_ = self._get_label_to_index_mapping()
        self.index_to_label_ = {index: label for label, index in self.label_to_index_.items()}

        if self._conditional:
            self.calibration_nonconformity_scores_ = {
                label: nonconformity_scores[y_calibration == label, index]
                for label, index in self.label_to_index_.items()
            }
        else:
            self.calibration_nonconformity_scores_ = {
                label: nonconformity_scores[range(len(y_calibration)), index]
                for label, index in self.label_to_index_.items()
            }

        return self

    @abstractmethod
    def _fit_and_calculate_calibration_nonconformity_scores(
        self,
        X: ArrayLike,
        calibration_size: float = 0.2,
        y: Optional[ArrayLike] = None,
        fit_params: dict = {},
        **kwargs: Dict[str, Any],
    ) -> Tuple[NDArray, NDArray]:
        pass

    @abstractmethod
    def _get_label_to_index_mapping(self) -> Dict[Any, int]:
        pass

    def predict(
        self,
        X: ArrayLike,
        confidence_level: float = 0.9,
        predict_params: dict = {},
        **kwargs: Dict[str, Any],
    ) -> NDArray:

        check_is_fitted(self, attributes=["label_to_index_", "index_to_label_"])

        super().predict(
            X=X,
            confidence_level=confidence_level,
            predict_params=predict_params,
            kwargs=kwargs,
        )

        sorted = kwargs.get("sorted", True)
        allow_empty_set = kwargs.get("allow_empty_set", True)

        y_hat, nonconformity_scores = self._predict_and_calculate_nonconformity_scores(
            X=X, predict_params=predict_params, kwargs=kwargs
        )
        y_prediction = np.array([self.index_to_label_[index] for index in np.argmax(y_hat, 1)])

        y_hats_if_in_prediction_set = np.full(nonconformity_scores.shape, np.nan)
        prediction_sets = self._create_numpy_array_for_labels_dtype(
            shape=nonconformity_scores.shape
        )

        for label, class_index in self.label_to_index_.items():
            q_hat = calculate_q_hat(self.calibration_nonconformity_scores_[label], confidence_level)

            # if calibration set does not have examples for `label`,
            # `calculate_q_hat` returns `None``
            if q_hat:
                # for now, we save both: class_label and predicted y_hat
                # if they are smaller than q_hat
                sample_mask = nonconformity_scores[:, class_index] < q_hat
                y_hats_if_in_prediction_set[sample_mask, class_index] = y_hat[
                    sample_mask, class_index
                ]
                prediction_sets[sample_mask, class_index] = label

            if not allow_empty_set:
                # enforce prediction set is at leas of size 1
                prediction_sets[y_prediction == label, class_index] = label

        if sorted:
            # descending sort the classes based on their y_hat predictions
            sorted_args = np.argsort(y_hats_if_in_prediction_set, axis=1)
            sorted_args = sorted_args[:, ::-1]
            sorted_prediction_sets_as_lists = np.take_along_axis(
                prediction_sets, sorted_args, axis=1
            ).tolist()

            # remove `NA`s so that ...
            prediction_sets_as_lists = [
                [prediction for prediction in list_of_prediction_sets if not pd.isna(prediction)]
                for list_of_prediction_sets in sorted_prediction_sets_as_lists
            ]

            # .. we can now move them to the end of the prediction sets and maintain numpy arrays
            # since it's no longer possible to use rows in matrices with different length
            prediction_sets = self._create_numpy_array_for_labels_dtype(
                shape=nonconformity_scores.shape
            )
            for idx in range(len(prediction_sets_as_lists)):
                prediction_sets[
                    idx, 0 : len(prediction_sets_as_lists[idx])  # noqa
                ] = prediction_sets_as_lists[idx]

        return prediction_sets

    @abstractmethod
    def _predict_and_calculate_nonconformity_scores(
        self,
        X: ArrayLike,
        predict_params: dict = {},
        **kwargs: Dict[str, Any],
    ) -> Tuple[NDArray, NDArray]:
        pass


class ConformalRegressor(ConformalPredictor):
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
        self,
        X: ArrayLike,
        calibration_size: float = 0.2,
        y: Optional[ArrayLike] = None,
        fit_params: dict = {},
        **kwargs: Dict[str, Any],
    ) -> ConformalRegressor:

        super().fit(
            X=X, calibration_size=calibration_size, y=y, fit_params=fit_params, kwargs=kwargs
        )

        (
            _,
            self.calibration_nonconformity_scores_,
        ) = self._fit_and_calculate_calibration_nonconformity_scores(
            X=X, calibration_size=calibration_size, y=y, fit_params=fit_params, kwargs=kwargs
        )

        return self

    @abstractmethod
    def _fit_and_calculate_calibration_nonconformity_scores(
        self,
        X: ArrayLike,
        calibration_size: float = 0.2,
        y: Optional[ArrayLike] = None,
        fit_params: dict = {},
        **kwargs: Dict[str, Any],
    ) -> Tuple[NDArray, NDArray]:
        pass

    def predict(
        self,
        X: ArrayLike,
        confidence_level: float = 0.9,
        predict_params: dict = {},
        **kwargs: Dict[str, Any],
    ) -> NDArray:

        super().predict(
            X=X,
            confidence_level=confidence_level,
            predict_params=predict_params,
            kwargs=kwargs,
        )

        y_hat, half_interval = self._predict_and_calculate_half_interval(
            X=X, confidence_level=confidence_level, predict_params=predict_params, kwargs=kwargs
        )

        y_hat_lower_bound = y_hat - half_interval
        y_hat_upper_bound = y_hat + half_interval

        return np.stack((y_hat_lower_bound, y_hat, y_hat_upper_bound), axis=1)

    @abstractmethod
    def _predict_and_calculate_half_interval(
        self,
        X: ArrayLike,
        confidence_level: float = 0.9,
        predict_params: dict = {},
        **kwargs: Dict[str, Any],
    ) -> Tuple[NDArray, float]:
        pass


class ConformalQuantileRegressor(ConformalPredictor):
    """
    Source:
        Romano, Y., Patterson, E., & Cand??s, E.J. (2019).
        Conformalized Quantile Regression. NeurIPS.
    """

    calibration_nonconformity_scores_: Union[NDArray, Dict[float, NDArray]]

    @staticmethod
    def _calculate_nonconformity_scores(y_hat: NDArray, y: NDArray) -> NDArray:
        return np.stack((y_hat[:, 0] - y, y - y_hat[:, 2]), axis=1).max(axis=1)

    @staticmethod
    def _calculate_lower_upper_quantiles(confidence_level: float) -> Tuple[float, float]:
        miscoverage_level = 1 - confidence_level
        lower_quantile = miscoverage_level / 2
        upper_quantile = 1 - miscoverage_level / 2

        return lower_quantile, upper_quantile

    def fit(
        self,
        X: ArrayLike,
        calibration_size: float = 0.2,
        y: Optional[ArrayLike] = None,
        fit_params: dict = {},
        **kwargs: Dict[str, Any],
    ) -> ConformalQuantileRegressor:

        super().fit(
            X=X, calibration_size=calibration_size, y=y, fit_params=fit_params, kwargs=kwargs
        )

        self._fit_method(
            X=X, calibration_size=calibration_size, y=y, fit_params=fit_params, kwargs=kwargs
        )
        return self

    @abstractmethod
    def _fit_method(
        self,
        X: ArrayLike,
        calibration_size: float = 0.2,
        y: Optional[ArrayLike] = None,
        fit_params: dict = {},
        **kwargs: Dict[str, Any],
    ) -> None:
        pass

    def predict(
        self,
        X: ArrayLike,
        confidence_level: Optional[float] = None,
        predict_params: dict = {},
        **kwargs: Dict[str, Any],
    ) -> NDArray:

        super().predict(
            X=X,
            confidence_level=confidence_level if confidence_level is not None else 0.5,
            predict_params=predict_params,
            kwargs=kwargs,
        )

        y_hat_quantiles, half_interval = self._predict_and_calculate_half_interval(
            X=X, confidence_level=confidence_level, predict_params=predict_params, kwargs=kwargs
        )

        y_hat_lower_bound = y_hat_quantiles[:, 0] - half_interval
        y_hat_upper_bound = y_hat_quantiles[:, 2] + half_interval

        return np.stack((y_hat_lower_bound, y_hat_quantiles[:, 1], y_hat_upper_bound), axis=1)

    @abstractmethod
    def _predict_and_calculate_half_interval(
        self,
        X: ArrayLike,
        confidence_level: Optional[float] = None,
        predict_params: dict = {},
        **kwargs: Dict[str, Any],
    ) -> Tuple[NDArray, float]:
        pass
