from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING, Any

from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

from conformal_inference.models.inductive import ConformalClassifier, ConformalQuantileRegressor
from conformal_inference.models.utils import calculate_q_hat, check_in_range

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

logger = getLogger()


class ConformalQuantileAutoGluonRegressor(ConformalQuantileRegressor):
    _predictor: TabularPredictor
    _q_hat: float
    _confidence_level: float
    _target_column: str

    def __init__(self, target_column: str, confidence_level: float = 0.9, predictor_params: dict = {}) -> None:
        check_in_range(confidence_level, "confidence_level")
        self._confidence_level = confidence_level
        self._target_column = target_column

        if type(predictor_params) != dict:
            msg = "'predictor_params' need to be dictionary of arguments."
            raise ValueError(msg)

        lower_quantile, upper_quantile = self._calculate_lower_upper_quantiles(confidence_level)

        for to_remove in ["label", "problem_type", "quantile_levels"]:
            if predictor_params.pop(to_remove, None) is not None:
                logger.warning(f"Ignoring '{to_remove}' of given 'predictor_params' " "since it is already defined.")

        super().__init__(
            TabularPredictor(
                label=target_column,
                problem_type="quantile",
                quantile_levels=[lower_quantile, 0.5, upper_quantile],
                **predictor_params,
            ),
        )

    def _fit_method(
        self,
        X: ArrayLike,
        calibration_size: float = 0.2,
        y: ArrayLike | None = None,
        fit_params: dict = {},
        **kwargs: dict[str, Any],
    ) -> None:
        if y is not None:
            msg = f"This implementation of 'ConformalClassifier' requires to be called with 'X', which should integrate the target column '{self._target_column}'."
            raise ValueError(
                msg,
            )

        training_data_, calibration_data_ = train_test_split(X, test_size=calibration_size)

        X_calibration = calibration_data_[
            [column for column in calibration_data_.columns if column != self._target_column]
        ]
        y_calibration = calibration_data_[self._target_column]

        self._predictor.fit(training_data_, **fit_params, calibrate=False)
        non_conformity_scores = self._calculate_nonconformity_scores(
            y_hat=self._predictor.predict(X_calibration, as_pandas=False),
            y=y_calibration,
        )

        # returning `None` isn't a problem here.
        # This only happens when no nonconformity scores are given.
        self._q_hat = calculate_q_hat(non_conformity_scores, self._confidence_level)  # type: ignore

        # bootstrap the necessary attributes
        self.calibration_nonconformity_scores_ = {}  # type: ignore

    def _predict_and_calculate_half_interval(
        self,
        X: ArrayLike,
        confidence_level: float | None = None,
        predict_params: dict = {},
        **kwargs: dict[str, Any],
    ) -> tuple[NDArray, float]:
        check_is_fitted(self, attributes=["_q_hat"])

        if confidence_level is not None:
            msg = f"This implementation of 'ConformalQuantileRegressor' does not allow to set 'confidence_level' for prediction. It was set to {self._confidence_level} during initialization."
            raise ValueError(
                msg,
            )

        y_hat_quantiles = self._predictor.predict(X, **predict_params, as_pandas=False)

        return y_hat_quantiles, self._q_hat


class ConformalAutoGluonClassifier(ConformalClassifier):
    _predictor: TabularPredictor

    def __init__(self, target_column: str, conditional: bool = True, predictor_params: dict = {}) -> None:
        if type(predictor_params) != dict:
            msg = "'predictor_params' need to be dictionary of arguments."
            raise ValueError(msg)

        if type(target_column) != str:
            msg = "'target_column' need to be of type 'str'."
            raise ValueError(msg)

        self._target_column = target_column

        for to_remove in ["label"]:
            if predictor_params.pop(to_remove, None) is not None:
                logger.warning(f"Ignoring '{to_remove}' of given 'predictor_params' " "since it is already defined.")

        super().__init__(
            TabularPredictor(
                label=target_column,
                **predictor_params,
            ),
            conditional=conditional,
            fit=True,
        )

    def _fit_and_calculate_calibration_nonconformity_scores(
        self,
        X: ArrayLike,
        calibration_size: float = 0.2,
        y: ArrayLike | None = None,
        fit_params: dict = {},
        **kwargs: dict[str, Any],
    ) -> tuple[NDArray, NDArray]:
        if y is not None:
            msg = f"This implementation of 'ConformalClassifier' requires to be called with 'X', which should integrate the target column '{self._target_column}'."
            raise ValueError(
                msg,
            )

        training_data_, calibration_data_ = train_test_split(X, test_size=calibration_size)

        X_calibration = calibration_data_[
            [column for column in calibration_data_.columns if column != self._target_column]
        ]

        # later on, we expect it to be a NDArray
        y_calibration = calibration_data_[self._target_column].to_numpy()

        self._predictor.fit(training_data_, **fit_params, calibrate=False)

        nonconformity_scores = self._calculate_nonconformity_scores(
            y_hat=self._predictor.predict_proba(X_calibration, as_pandas=False),
        )

        return y_calibration, nonconformity_scores

    def _get_label_to_index_mapping(self) -> dict[Any, int]:
        return {x: index for index, x in enumerate(self._predictor.class_labels)}

    def _predict_and_calculate_nonconformity_scores(
        self,
        X: ArrayLike,
        predict_params: dict = {},
        **kwargs: dict[str, Any],
    ) -> tuple[NDArray, NDArray]:
        y_hat = self._predictor.predict_proba(X, **predict_params, as_pandas=False)
        nonconformity_scores = self._calculate_nonconformity_scores(y_hat=y_hat)

        return y_hat, nonconformity_scores
