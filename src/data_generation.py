import numpy as np

def generate_linear(n=100, noise=1.0, seed=42):
    np.random.seed(seed)
    x = np.random.uniform(-5, 5, n)
    y = 2 * x + np.random.normal(0, noise, n)
    return x.reshape(-1, 1), y

def generate_quadratic(n=100, noise=1.0, seed=42):
    np.random.seed(seed)
    x = np.random.uniform(-5, 5, n)
    y = x**2 + np.random.normal(0, noise, n)
    return x.reshape(-1, 1), y

def generate_multicollinear(n=100, noise=0.1, seed=42):
    np.random.seed(seed)
    x1 = np.random.randn(n)
    x2 = x1 + np.random.normal(0, noise, n)
    y = 3 * x1 + np.random.normal(0, 1, n)
    X = np.column_stack([x1, x2])
    return X, y

def inject_outliers(X, y, num_outliers=5, magnitude=20):
    X_out = X.copy()
    y_out = y.copy()
    idx = np.random.choice(len(y), num_outliers, replace=False)
    y_out[idx] += magnitude
    return X_out, y_out
