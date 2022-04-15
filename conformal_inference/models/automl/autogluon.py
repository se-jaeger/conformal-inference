from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from autogluon.tabular import TabularPredictor
from numpy.typing import ArrayLike, NDArray
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

from ..inductive import ConformalClassifier, ConformalQuantileRegressor
from ..utils import calculate_q_hat, check_in_range


class ConformalQuantileAutoGluonRegressor(ConformalQuantileRegressor):

    _predictor: TabularPredictor
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

    def _fit_method(
        self,
        X: ArrayLike,
        calibration_size: float = 0.2,
        y: Optional[ArrayLike] = None,
        fit_params: dict = {},
        **kwargs: Dict[str, Any],
    ) -> None:

        if y is not None:
            raise ValueError(
                "This implementation of 'ConformalClassifier' requires to be called with 'X', "
                f"which should integrate the target column '{self._target_column}'."
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
        self._q_hat = calculate_q_hat(non_conformity_scores, self._confidence_level)

        # bootstrap the necessary attributes
        self.calibration_nonconformity_scores_ = {}  # type: ignore

    def _predict_and_calculate_half_interval(
        self,
        X: ArrayLike,
        confidence_level: Optional[float] = None,
        predict_params: dict = {},
        **kwargs: Dict[str, Any],
    ) -> Tuple[NDArray, float]:
        check_is_fitted(self, attributes=["_q_hat"])

        if confidence_level is not None:
            raise ValueError(
                "This implementation of 'ConformalQuantileRegressor' does not allow to set "
                f"'confidence_level' for prediction. It was set to {self._confidence_level} "
                "during initialization."
            )

        y_hat_quantiles = self._predictor.predict(X, **predict_params, as_pandas=False)

        return y_hat_quantiles, self._q_hat


class ConformalAutoGluonClassifier(ConformalClassifier):

    _predictor: TabularPredictor

    def __init__(
        self, target_column: str, conditional: bool = True, predictor_params: dict = {}
    ) -> None:

        if type(predictor_params) != dict:
            raise ValueError("'predictor_params' need to be dictionary of arguments.")

        if type(target_column) != str:
            raise ValueError("'target_column' need to be of type 'str'.")

        self._target_column = target_column

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
        y: Optional[ArrayLike] = None,
        fit_params: dict = {},
        **kwargs: Dict[str, Any],
    ) -> Tuple[NDArray, NDArray]:

        if y is not None:
            raise ValueError(
                "This implementation of 'ConformalClassifier' requires to be called with 'X', "
                f"which should integrate the target column '{self._target_column}'."
            )

        training_data_, calibration_data_ = train_test_split(X, test_size=calibration_size)

        X_calibration = calibration_data_[
            [column for column in calibration_data_.columns if column != self._target_column]
        ]

        # later on, we expect it to be a NDArray
        y_calibration = calibration_data_[self._target_column].to_numpy()

        self._predictor.fit(training_data_, **fit_params, calibrate=False)

        nonconformity_scores = self._calculate_nonconformity_scores(
            y_hat=self._predictor.predict_proba(X_calibration, as_pandas=False)
        )

        return y_calibration, nonconformity_scores

    def _get_label_to_index_mapping(self) -> Dict[Any, int]:
        return {x: index for index, x in enumerate(self._predictor.class_labels)}

    def _predict_and_calculate_nonconformity_scores(
        self,
        X: ArrayLike,
        predict_params: dict = {},
        **kwargs: Dict[str, Any],
    ) -> Tuple[NDArray, NDArray]:

        y_hat = self._predictor.predict_proba(X, **predict_params, as_pandas=False)
        nonconformity_scores = self._calculate_nonconformity_scores(y_hat=y_hat)

        return y_hat, nonconformity_scores
