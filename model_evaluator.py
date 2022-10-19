from abc import ABC, abstractmethod
from typing import Optional
from sklearn.metrics import median_absolute_error, mean_squared_error, r2_score


def float_to_pct(num: float, precision: int) -> str:
    return f"{num * 100:.{precision}f}%"


def root_squared_mean_error(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)


class ModelEvaluator(ABC):
    @abstractmethod
    def compute_scores(self, estimator, X_train, X_test, y_train, y_test) -> None:
        pass

    @abstractmethod
    def print_scores(self, float_precision) -> None:
        pass


class RegModelEvaluator(ModelEvaluator):

    REG_METRICS = {
        'R2': r2_score,
        'MAE': median_absolute_error,
        'RSME': root_squared_mean_error
    }

    def __init__(self):
        self.scores: dict[str, Optional[float, str]] | None = None

    def compute_scores(self, estimator, X_train, X_test, y_train, y_test):

        y_hat_test = estimator.predict(X_test)
        y_hat_train = estimator.predict(X_train)
        test_train_data = {
            'test': {'y_true': y_test, 'y_pred': y_hat_test},
            'train': {'y_true': y_train, 'y_pred': y_hat_train}
        }
        self.scores = {
            f"{metric_name}_{dataset}": metric_func(data['y_true'], data['y_pred'])
            for metric_name, metric_func in self.REG_METRICS.items()
            for dataset, data in test_train_data.items()
        }

    def print_scores(self, float_precision: int = 3) -> None:

        if self.scores is None:
            raise Exception("The RegModelEvaluator instance's scores are not computed yet. Call 'compute_scores' with "
                            "appropriate arguments before trying to print its values")

        for metric, score in self.scores.items():
            if 'R2' in metric:
                score_rounded = round(score, float_precision)
            else:
                score_rounded = float_to_pct(round(score, float_precision), float_precision - 2)
            print(f"{metric}: {score_rounded}")
