from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

from ..base import ConformalClassifier, InductiveConformalPredictor
from ..utils import calculate_q_hat, check_in_range


class ConformalQuantileAutoGluonRegressor(InductiveConformalPredictor):
    """
    Source:
        Romano, Y., Patterson, E., & Candès, E.J. (2019).
        Conformalized Quantile Regression. NeurIPS.
    """

    predictor: TabularPredictor
    _q_hat: float
    _confidence_level: float
    _target_column: str

    def __init__(
        self, target_column: str, confidence_level: float = 0.9, predictor_params: dict = {}
    ) -> None:

        check_in_range(confidence_level, "confidence_level")
        self._confidence_level = confidence_level
        self._target_column = target_column

        if type(predictor_params) != dict:
            raise ValueError("'predictor_params' need to be dictionary of arguments.")

        lower_quantile, upper_quantile = self._calculate_lower_upper_quantiles(confidence_level)

        super().__init__(
            TabularPredictor(
                label=target_column,
                problem_type="quantile",
                quantile_levels=[lower_quantile, 0.5, upper_quantile],
                **predictor_params,
            )
        )

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
        training_data: Union[TabularDataset, pd.DataFrame],
        calibration_size: float = 0.2,
        fit_params: dict = {},
    ) -> ConformalQuantileAutoGluonRegressor:

        check_in_range(calibration_size, "calibration_size")

        if type(fit_params) != dict:
            raise ValueError("'fit_params' need to be dictionary of arguments.")

        training_data_, calibration_data_ = train_test_split(
            training_data, test_size=calibration_size
        )

        X_calibration = calibration_data_[
            [column for column in calibration_data_.columns if column != self._target_column]
        ]
        y_calibration = calibration_data_[self._target_column]

        self.predictor.fit(training_data_, **fit_params, calibrate=False)
        non_conformity_scores = self._calculate_nonconformity_scores(
            y_hat=self.predictor.predict(X_calibration, as_pandas=False),
            y=y_calibration,
        )
        self._q_hat = calculate_q_hat(non_conformity_scores, self._confidence_level)

        return self

    def predict(self, data: Union[TabularDataset, pd.DataFrame]) -> NDArray:
        check_is_fitted(self, attributes=["_q_hat"])

        y_hat_quantiles = self.predictor.predict(data, as_pandas=False)

        y_hat_lower_bound = y_hat_quantiles[:, 0] - self._q_hat
        y_hat_upper_bound = y_hat_quantiles[:, 2] + self._q_hat

        return np.stack((y_hat_lower_bound, y_hat_quantiles[:, 1], y_hat_upper_bound), axis=1)


class ConformalAutoGluonClassifier(ConformalClassifier):

    predictor: TabularPredictor
    _labels_dtype: Optional[Any] = None

    def __init__(self, target_column: str, predictor_params: dict = {}) -> None:

        self._target_column = target_column

        if type(predictor_params) != dict:
            raise ValueError("'predictor_params' need to be dictionary of arguments.")

        super().__init__(
            TabularPredictor(
                label=target_column,
                **predictor_params,
            )
        )

    def _create_numpy_array_for_labels_dtype(self, shape: Tuple[int, ...]) -> NDArray:
        if self._labels_dtype is None:
            are_labels_numerical = list(
                {
                    isinstance(x, (int, float, complex)) and not isinstance(x, bool)
                    for x in self.predictor.class_labels
                }
            )

            if len(are_labels_numerical) > 1 or not are_labels_numerical[0]:
                self._labels_dtype = object

            else:
                self._labels_dtype = float

        return np.full(shape, np.nan, dtype=self._labels_dtype)

    # it's fine, signature changed by intention
    def fit(  # type: ignore
        self,
        training_data: Union[TabularDataset, pd.DataFrame],
        calibration_size: float = 0.2,
        fit_params: dict = {},
    ) -> ConformalAutoGluonClassifier:

        check_in_range(calibration_size, "calibration_size")

        if type(fit_params) != dict:
            raise ValueError("'fit_params' need to be dictionary of arguments.")

        training_data_, calibration_data_ = train_test_split(
            training_data, test_size=calibration_size
        )

        X_calibration = calibration_data_[
            [column for column in calibration_data_.columns if column != self._target_column]
        ]
        y_calibration = calibration_data_[self._target_column]

        self.predictor.fit(training_data_, **fit_params, calibrate=False)

        self.label_2_index_ = {x: index for index, x in enumerate(self.predictor.class_labels)}
        nonconformity_scores = self._calculate_nonconformity_scores(
            y_hat=self.predictor.predict_proba(X_calibration, as_pandas=False)
        )

        self.calibration_nonconformity_scores_ = {
            label: nonconformity_scores[y_calibration == label, index]
            for label, index in self.label_2_index_.items()
        }

        return self

    def predict(
        self, data: Union[TabularDataset, pd.DataFrame], confidence_level: float = 0.9
    ) -> NDArray:
        check_in_range(confidence_level, "confidence_level")
        check_is_fitted(self, attributes=["calibration_nonconformity_scores_", "label_2_index_"])

        y_hat = self.predictor.predict_proba(data, as_pandas=False)
        nonconformity_scores = self._calculate_nonconformity_scores(y_hat=y_hat)

        # numpy does not allow to
        y_hats_if_in_prediction_set = np.full(nonconformity_scores.shape, np.nan)
        class_labels_if_in_prediction_set = self._create_numpy_array_for_labels_dtype(
            shape=nonconformity_scores.shape
        )

        for label, class_index in self.label_2_index_.items():
            q_hat = calculate_q_hat(self.calibration_nonconformity_scores_[label], confidence_level)

            # for now, we save both: class_label and predicted y_hat if they are smaller than q_hat
            sample_mask = nonconformity_scores[:, class_index] < q_hat
            y_hats_if_in_prediction_set[sample_mask, class_index] = y_hat[sample_mask, class_index]
            class_labels_if_in_prediction_set[sample_mask, class_index] = label

        # descending sort the classes based on their y_hat predictions
        sorted_args = np.argsort(y_hats_if_in_prediction_set, axis=1)
        sorted_args = sorted_args[:, ::-1]
        list_of_prediction_set_lists = np.take_along_axis(
            class_labels_if_in_prediction_set, sorted_args, axis=1
        ).tolist()

        prediction_sets_as_lists = [
            [prediction for prediction in list_of_prediction_sets if not pd.isna(prediction)]
            for list_of_prediction_sets in list_of_prediction_set_lists
        ]

        prediction_sets = self._create_numpy_array_for_labels_dtype(
            shape=nonconformity_scores.shape
        )
        for idx in range(len(prediction_sets_as_lists)):
            prediction_sets[
                idx, 0 : len(prediction_sets_as_lists[idx])  # noqa
            ] = prediction_sets_as_lists[idx]

        return prediction_sets
