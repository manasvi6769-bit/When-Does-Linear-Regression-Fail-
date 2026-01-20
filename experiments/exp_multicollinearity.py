import numpy as np
import pandas as pd
from src.data_generation import generate_multicollinear
from src.models import linear_model, ridge_model, lasso_model
from src.metrics import mse

def run_experiment():
    X, y = generate_multicollinear()
    alphas = np.logspace(-3, 2, 20)
    results = []

    # Linear regression
    lr = linear_model()
    lr.fit(X, y)
    results.append({
        "model": "Linear",
        "alpha": 0,
        "coef_1": lr.coef_[0],
        "coef_2": lr.coef_[1],
        "mse": mse(y, lr.predict(X))
    })

    for alpha in alphas:
        ridge = ridge_model(alpha)
        ridge.fit(X, y)

        lasso = lasso_model(alpha)
        lasso.fit(X, y)

        results.extend([
            {
                "model": "Ridge",
                "alpha": alpha,
                "coef_1": ridge.coef_[0],
                "coef_2": ridge.coef_[1],
                "mse": mse(y, ridge.predict(X))
            },
            {
                "model": "Lasso",
                "alpha": alpha,
                "coef_1": lasso.coef_[0],
                "coef_2": lasso.coef_[1],
                "mse": mse(y, lasso.predict(X))
            }
        ])

    pd.DataFrame(results).to_csv("results/exp2_results.csv", index=False)

if __name__ == "__main__":
    run_experiment()
