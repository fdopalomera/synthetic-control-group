from __future__ import annotations
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from typing import Any
from model_generator import RegressionEstimatorGenerator, RegLinearModel
from model_evaluator import RegModelEvaluator


@dataclass
class SyntheticControl:
    dependent_city: str
    independent_cities: list[str]
    estimator = None

    def get_train_test_datasets(self, data) -> tuple[list[float], list[float], list[float], list[float]]:

        x = data.loc[:, self.independent_cities]
        y = data[self.dependent_city]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=42)
        return x_train, x_test, y_train, y_test

    def generate_estimator(self, x_train, y_train, param_grid: dict[str, Any], base_model: RegLinearModel = RegLinearModel.RIDGE) -> None:

        estimator_generator = RegressionEstimatorGenerator(x_train, y_train)
        reg_model = estimator_generator.get_best_model(param_grid)
        self.estimator = reg_model

    def print_evaluation_report(self, x_train, x_test, y_train, y_test) -> None:
        if self.estimator is None:
            raise Exception("The SyntheticControl instance's estimator is not generated yet. "
                            "Call 'generate_estimator' with appropriate arguments before trying to get the evaluation report")

        model_eval = RegModelEvaluator()
        model_eval.compute_scores(self.estimator, x_train, x_test, y_train, y_test)
        model_eval.print_scores()

    def generate_and_evaluate_estimator(self, data, param_grid):

        # Create the datasets to train and test the model
        x_train, x_test, y_train, y_test = self.get_train_test_datasets(data)
        print(f"y_train samples: {len(y_train)}\ny_test samples: {len(y_test)}")

        self.generate_estimator(x_train, y_train, param_grid)
        print(f"{self.estimator.__str__()}")

        self.print_evaluation_report(x_train, x_test, y_train, y_test)
