import pandas as pd
from src.data_generation import generate_linear, generate_quadratic
from src.models import linear_model, polynomial_model
from src.metrics import mse

def run_experiment():
    results = []

    for data_type in ["linear", "quadratic"]:
        if data_type == "linear":
            X, y = generate_linear()
        else:
            X, y = generate_quadratic()

        models = {
            "Linear": linear_model(),
            "Poly2": polynomial_model(2),
            "Poly3": polynomial_model(3)
        }

        for name, model in models.items():
            model.fit(X, y)
            preds = model.predict(X)
            error = mse(y, preds)

            results.append({
                "data": data_type,
                "model": name,
                "mse": error
            })

    pd.DataFrame(results).to_csv("results/exp1_results.csv", index=False)

if __name__ == "__main__":
    run_experiment()
