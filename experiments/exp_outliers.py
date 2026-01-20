import pandas as pd
from src.data_generation import generate_linear, inject_outliers
from src.models import linear_model, ridge_model
from src.metrics import mse

def run_experiment():
    X, y = generate_linear()
    X_out, y_out = inject_outliers(X, y)

    models = {
        "Linear": linear_model(),
        "Ridge": ridge_model(alpha=1.0)
    }

    results = []

    for name, model in models.items():
        model.fit(X, y)
        clean_mse = mse(y, model.predict(X))

        model.fit(X_out, y_out)
        out_mse = mse(y_out, model.predict(X_out))

        results.append({
            "model": name,
            "clean_mse": clean_mse,
            "outlier_mse": out_mse
        })

    pd.DataFrame(results).to_csv("results/exp3_results.csv", index=False)

if __name__ == "__main__":
    run_experiment()
