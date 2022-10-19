from __future__ import annotations
from enum import Enum, auto
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from dataclasses import dataclass


class RegLinearModel(Enum):
    RIDGE = auto()
    LASSO = auto()


REG_LINEAR_MODELS = {
    RegLinearModel.RIDGE: Ridge,
    RegLinearModel.LASSO: Lasso
}


@dataclass
class RegressionEstimatorGenerator:
    x_train: list[float]
    y_train: list[float]
    model: RegLinearModel = RegLinearModel.RIDGE

    def get_best_parameters(self, param_grid, cv_partitions: int = 4) -> dict:

        estimator = REG_LINEAR_MODELS[self.model]()
        model_cv = GridSearchCV(estimator,
                                param_grid,
                                scoring="neg_median_absolute_error",  # "r2" or "neg_median_absolute_error"
                                cv=cv_partitions,
                                verbose=1,
                                return_train_score=True,
                                n_jobs=-2)
        model_cv.fit(self.x_train, self.y_train)
        return model_cv.best_params_

    def get_best_model(self, param_grid):

        best_params = self.get_best_parameters(param_grid)
        best_estimator = REG_LINEAR_MODELS[self.model](**best_params)
        best_estimator.fit(self.x_train, self.y_train)
        return best_estimator
