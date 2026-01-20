# When Does Linear Regression Fail?

## Overview
Linear regression is one of the most widely used models in machine learning due to its simplicity, interpretability, and strong theoretical foundations. However, its effectiveness relies on several assumptions about the data. When these assumptions are violated, linear regression can produce misleading or unstable results.

This project systematically studies the **failure modes of linear regression** using controlled experiments. By generating synthetic datasets with known ground truth, we isolate individual assumption violations and analyze how they affect model behavior. We also examine how regularization techniques such as Ridge and Lasso mitigate some of these failures.

The goal of this project is not to optimize performance, but to develop **research-oriented intuition** about model assumptions, limitations, and diagnostics.

---

## Research Questions
This project is guided by the following questions:

- How does linear regression behave when the true relationship is non-linear?
- What happens to coefficient stability in the presence of multicollinearity?
- How sensitive is linear regression to outliers?
- How do regularization methods (Ridge and Lasso) change these behaviors?

---

## Dataset Strategy
We primarily use **synthetic datasets**, which allows full control over data-generating processes and ground truth. This makes it possible to clearly identify when and why a model fails.

### Synthetic Data Scenarios
- **Linear relationship**: baseline case where assumptions hold  
- **Non-linear relationship**: quadratic data to violate linearity  
- **Multicollinearity**: highly correlated features to study coefficient instability  
- **Outliers**: injected extreme values to test robustness  

(Optional) A real-world regression dataset may be added for comparison.

---

## Experiments

### Experiment 1: Failure on Non-Linear Data
**Question:**  
What happens when the true relationship is non-linear but we apply linear regression?

**Models:**  
- Linear Regression  
- Polynomial Regression (degree 2 and 3)

**Analysis:**  
- Mean Squared Error (MSE)  
- Residual plots  

**Key Insight:**  
Linear regression underfits non-linear patterns, which is revealed more clearly through residual analysis than error metrics alone.

---

### Experiment 2: Multicollinearity
**Question:**  
How does multicollinearity affect coefficient stability?

**Models:**  
- Linear Regression  
- Ridge Regression  
- Lasso Regression  

**Analysis:**  
- Coefficient magnitudes  
- Prediction error vs regularization strength  

**Key Insight:**  
Multicollinearity inflates coefficient variance without significantly improving predictive performance. Ridge stabilizes coefficients, while Lasso promotes sparsity.

---

### Experiment 3: Sensitivity to Outliers
**Question:**  
How robust is linear regression to outliers?

**Models:**  
- Linear Regression  
- Ridge Regression  

**Analysis:**  
- Regression line shift  
- Residual distributions  

**Key Insight:**  
Least-squares regression is highly sensitive to outliers due to squared-error loss. Regularization offers limited robustness.

---



