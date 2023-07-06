from sklearn.base import BaseEstimator, is_classifier, is_regressor
from sklearn.utils.validation import check_is_fitted

from conformal_inference.models import setup_logger

from .base import ConformalSKLearnClassifier, ConformalSKLearnRegressor, SKLearnConformalPredictor

setup_logger(__name__)


def create_SKLearnConformalPredictor(
    predictor: BaseEstimator,
    conditional: bool = True,
    fit: bool = True,
) -> SKLearnConformalPredictor:
    if not fit:
        check_is_fitted(predictor)

    if is_classifier(predictor):
        return ConformalSKLearnClassifier(
            predictor=predictor,
            conditional=conditional,
            fit=fit,
        )

    elif is_regressor(predictor):
        return ConformalSKLearnRegressor(
            predictor=predictor,
            conditional=conditional,
            fit=fit,
        )
    else:
        msg = "Only Classifiers or Regressors from `scikit-learn` are supported!"
        raise Exception(msg)
