from __future__ import annotations

from typing import Tuple, Union

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
        return np.stack((y_hat[:, 0] - y, y - y_hat[:, -1]), axis=1).max(axis=1)

    @staticmethod
    def _calculate_lower_upper_quantiles(confidence_level: float) -> Tuple[float, float]:
        miscoverage_level = 1 - confidence_level
        lower_quantile = miscoverage_level / 2
        upper_quantile = 1 - miscoverage_level / 2

        return lower_quantile, upper_quantile

    def fit(
        self, training_data: Union[TabularDataset, pd.DataFrame], calibration_size: float = 0.2
    ) -> ConformalQuantileAutoGluonRegressor:

        check_in_range(calibration_size, "calibration_size")

        training_data_, calibration_data_ = train_test_split(
            training_data, test_size=calibration_size
        )

        X_calibration = calibration_data_[
            [column for column in calibration_data_.columns if column != self._target_column]
        ]
        y_calibration = calibration_data_[self._target_column]

        self.predictor.fit(training_data_)
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
        y_hat_upper_bound = y_hat_quantiles[:, -1] + self._q_hat

        return np.stack((y_hat_lower_bound, y_hat_upper_bound), axis=1)


class ConformalQuantileAutoGluonClassifier(ConformalClassifier):

    _multiclass: bool
    predictor: TabularPredictor

    def __init__(
        self, target_column: str, multiclass: bool = True, predictor_params: dict = {}
    ) -> None:

        self._multiclass = multiclass
        self._target_column = target_column

        if type(predictor_params) != dict:
            raise ValueError("'predictor_params' need to be dictionary of arguments.")

        super().__init__(
            TabularPredictor(
                label=target_column,
                problem_type="multiclass" if self._multiclass else "binary",
                **predictor_params,
            )
        )

    # it's fine signature changed by intention
    def fit(  # type: ignore
        self, training_data: Union[TabularDataset, pd.DataFrame], calibration_size: float = 0.2
    ) -> ConformalQuantileAutoGluonClassifier:

        check_in_range(calibration_size, "calibration_size")

        training_data_, calibration_data_ = train_test_split(
            training_data, test_size=calibration_size
        )

        X_calibration = calibration_data_[
            [column for column in calibration_data_.columns if column != self._target_column]
        ]
        y_calibration = calibration_data_[self._target_column]

        self.predictor.fit(training_data_)

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

        # prepare prediction sets. If class is not included in prediction, use `np.nan`
        prediction_sets = np.empty(nonconformity_scores.shape)
        prediction_sets[:] = np.nan

        for label, index in self.label_2_index_.items():
            q_hat = calculate_q_hat(self.calibration_nonconformity_scores_[label], confidence_level)

            prediction_sets[nonconformity_scores[:, index] < q_hat, index] = label

        return prediction_sets
