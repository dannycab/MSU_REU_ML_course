---
marp: true
theme: nordic
paginate: true

title: Day 03 — Regression
description: MSU REU ML Short Course, Day 03
author: Danny Caballero
---

<!-- _class: title -->

# Day 03
## Regression with `scikit-learn`

MSU REU Machine Learning Short Course

---

# Classification vs Regression

Same pipeline. Different output.

| | Classification | Regression |
|---|---|---|
| **Output** | A category (STAR / QSO) | A number (z-band magnitude) |
| **Loss** | Misclassification rate | Mean Squared Error |
| **Evaluation** | Confusion matrix, F1 | MSE, R², residuals |
| **Example** | "Is this a star?" | "How bright is this star?" |

---

# Linear Regression

The model: **ŷ = β₀ + β₁x₁ + β₂x₂ + … + βₙxₙ**

The coefficients **β** are learned from training data by minimizing MSE.

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

Same API as KNN. That's the point of scikit-learn.

---

# How good is your fit? MSE and R²

**MSE** — average squared distance between predicted and actual values. Lower is better.

**R²** — fraction of variance explained by the model. 1.0 is perfect; 0.0 is no better than predicting the mean.

```python
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)
```

R² = 0.95 sounds great. But always look at the plots.

---

# Always visualize — Anscombe's Quartet

Four datasets. Identical means, variances, correlations, and R² values. Completely different patterns.

> Identical statistics **do not** mean identical data.
> See: [The Importance of Visualization](../notes/importance_of_visualization.ipynb)

**The lesson:** before trusting any metric, make the plots.

---

<!-- _class: img-full -->

# Three diagnostic plots

Always make these: residuals, predicted vs actual, and the fit itself.

![width:820px](./figures/residuals_scatter_hist.png)

Residuals should be centered on zero with no structure. If they fan out or curve — your model is missing something.

---

<!-- _class: img-full -->

# When linear isn't enough — Random Forest

Same data, different model. The sklearn API doesn't change.

![width:760px](./figures/stellar_rf_regression_results_star.png)

```python
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100)
```

---

# Today's Activity

Work through **Activity 03: Regression with scikit-learn**

1. Predict z-band magnitude from other photometric bands
2. Evaluate with MSE, R², and diagnostic plots
3. Add redshift — does it help regression the way it helped classification?
4. Try per-class models (STAR, GALAXY, QSO separately)
5. Compare Linear Regression vs Random Forest

> **Note:** [The Importance of Visualization](../notes/importance_of_visualization.ipynb)
