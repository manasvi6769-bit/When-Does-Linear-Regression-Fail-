from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

def linear_model():
    return LinearRegression()

def polynomial_model(degree):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("lr", LinearRegression())
    ])

def ridge_model(alpha):
    return Ridge(alpha=alpha)

def lasso_model(alpha):
    return Lasso(alpha=alpha, max_iter=10000)
