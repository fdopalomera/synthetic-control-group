import numpy as np
import pandas as pd
from synthetic_control import SyntheticControl


def main() -> None:

    rng = np.random.default_rng()

    DEPENDENT_CITY = "Santiago"
    INDEPENDET_CITIES = ["Valparaíso", "Concecpción"]

    param_grid = {
        "alpha": np.arange(.00, .25, .005),
        "fit_intercept": [True, False]
    }
    columns = [DEPENDENT_CITY] + INDEPENDET_CITIES
    values = rng.standard_normal((20, 3))
    data = pd.DataFrame(data=values, columns=columns)

    synth_control = SyntheticControl(DEPENDENT_CITY, INDEPENDET_CITIES)
    synth_control.generate_and_evaluate_estimator(data, param_grid)


if __name__ == '__main__':
    main()
