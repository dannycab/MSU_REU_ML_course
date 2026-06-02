---
marp: true
theme: nordic
paginate: true

title: Day 03 — Regression
description: MSU REU ML Short Course, Day 03
author: Danny Caballero
---

<!-- _class: title -->

# Day 03: Regression with `scikit-learn`
## Predicting continuous values

MSU REU Machine Learning Short Course

---

# Learning Goals for Today

By the end of this activity, you'll be able to:

1. **Distinguish regression from classification** — when each is the right tool
2. **Build and train a linear regression model** using the same pipeline as classification
3. **Evaluate regression models** with MSE, R², and diagnostic residual plots
4. **Diagnose what your model is missing** by reading residuals
5. **Compare different regression models** (Linear vs Random Forest)
6. **Improve a model** by adding features and per-class approaches

This is the same workflow as classification. The only difference: you're predicting a number, not a category.

---

# The Real Difference: One Question

| | Classification | Regression |
|---|---|---|
| **Question** | "What is this?" | "How much?" |
| **Output** | A category (STAR, GALAXY, QSO) | A number (z-magnitude, temperature) |
| **Example** | Classify a spectrum as a star or quasar | Predict the brightness of that star |

**Same dataset. Same features. Different target variable.**

---

# Why Regression? Real Examples

Beyond classification, we often need **continuous predictions**:

| Problem | Features | Target |
|---|---|---|
| **Stellar brightness** | Colors (u, g, r, i, z) | Magnitude (0–30) |
| **Molecular energy** | Atomic structure | Bond energy (continuous) |
| **Star temperature** | Spectral absorption lines | Temperature (K) |
| **Galaxy redshift** | Brightness + colors | Redshift z (0–7) |

**Key insight:** When you need a precise number, not just a category, regression is your tool.

---

# Linear Regression: The Simplest Model

## The idea
Fit a straight line (or plane, or hyperplane) through the data.

**Prediction formula:**
$$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_n x_n$$

Where:
- $\hat{y}$ = predicted value
- $x_i$ = features (colors, spectrum)
- $\beta_i$ = learned weights (how much each feature matters)

**Goal:** Find coefficients **β** that minimize error.

---

# Measuring Error in Regression

## What is a residual?
$$\text{residual} = \text{predicted} - \text{actual}$$

Large residual = bad prediction. Small residual = good prediction.

## Why minimize *squared* error (MSE)?
- Penalizes big mistakes harder than small ones
- Mathematically convenient (smooth, differentiable)
- Interpretable: MSE is in *squared* units of your target

**Example:** If predicting magnitude (0–30), an error of 5 is worse than 5 errors of 1.

---

# The Regression Pipeline

![width:900px](./figures/d2/regression-pipeline.svg)

**Identical to classification**: Prepare → Scale → Train → Predict → Evaluate

The only difference: your target is continuous, not categorical.

---

# Building the Model (Still the Same Pattern)

The `scikit-learn` workflow is **identical to KNN**:

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Prepare: split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. Scale: fit on training data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Train & Predict
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# 4. Evaluate
from sklearn.metrics import mean_squared_error, r2_score
print(r2_score(y_test, y_pred))
```

**Same structure, every time.**

---

# Evaluation: MSE and R²

**Mean Squared Error (MSE)** — average squared distance between predicted and actual
$$\text{MSE} = \frac{1}{n} \sum (y_{\text{true}} - y_{\text{pred}})^2$$
Lower is better. In same units as your squared target.

**R² (R-squared)** — fraction of variance explained
$$R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}$$
- 1.0 = perfect fit
- 0.0 = no better than predicting the mean
- 0.95 = good, but always verify with plots

---

# A Warning: Anscombe's Quartet

**Four datasets. Identical means, variances, correlations, AND R² values.**

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Anscombe%27s_quartet_3.svg/1280px-Anscombe%27s_quartet_3.svg.png" alt="Anscombe's Quartet" width="700">

> Identical statistics **do not** mean identical data. Always plot.

---

# Diagnostic Plots: Reading Residuals

Always visualize three things:

1. **Residuals vs Predicted** — scatter plot: do residuals look random?
2. **Histogram of residuals** — distribution: is it centered at zero?
3. **Actual vs Predicted** — 45° line should align with your points

<img src="./figures/residuals_scatter_hist.png" alt="Residuals plots" width="820">

---

# What to Look For in Residuals

![width:900px](./figures/d2/residuals-diagnostic.svg)

**Rule:** Residuals should look like **random noise** around zero.

If they have a pattern (curved, fanned, clustered), your model is missing something.

---

# When Residuals Tell You Something

## Your model doesn't fit
- **Curved pattern** → Relationship is non-linear (not a straight line)
- **Fan shape** → Error gets bigger at high values (heteroscedasticity)
- **Clustering** → Model works better for some groups than others

## What to do?
- Try a **different model** (Random Forest, polynomial)
- **Add more features** (maybe colors interact?)
- **Separate by class** (STAR and GALAXY have different color-brightness relationships)

---

# When Linear Isn't Enough

**Same data, same question. Different model.**

<img src="./figures/stellar_rf_regression_results_star.png" alt="Linear vs Random Forest" width="760">

Random Forest can capture **non-linear patterns** — curves, interactions, complexity that linear models miss.

Same `scikit-learn` API:

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
```

---

# Model Comparison: Linear vs Random Forest

![width:900px](./figures/d2/model-comparison.svg)

**Linear regression** assumes a straight line. Fast, interpretable, fails on curves.

**Random Forest** is flexible but slower and prone to overfitting.

**Which to use?** Visualize your residuals first. If they're curved, try Random Forest. If they're random, Linear might be enough.

---

# Does Redshift Dominate Here Too?

**Parallel to classification: does one feature carry all the signal?**

<img src="./figures/stellar_regression_results.png" alt="Linear vs with redshift" width="400">

| Without redshift | With redshift |
|---|---|
| Low R² (colors alone are weak predictors) | High R² (redshift correlates with magnitude) |
| Colors matter | But is it "cheating"? |

**Key question:** Adding redshift improves R², but does the model learn color-magnitude relationships, or just memorize redshift?

**Your job:** Compare residuals with and without redshift. Does one have obvious patterns?

---

# Per-Class Regression: Better than One-Size-Fits-All

**Stars, galaxies, and quasars have different color-magnitude relationships.**

Brighter stars → different color changes than brighter galaxies.

```python
# Naive: one model for all 100,000 objects
model = LinearRegression()
model.fit(X_train, y_train)

# Better: fit one model per class
models = {}
for class_name in ["STAR", "GALAXY", "QSO"]:
    X_class = X_train[y_class_train == class_name]
    y_class = y_train[y_class_train == class_name]
    
    models[class_name] = LinearRegression()
    models[class_name].fit(X_class, y_class)

# Predict with the right model for each object
for obj in test_set:
    class_pred = classifier.predict(obj)
    magnitude_pred = models[class_pred].predict(obj)
```

**Trade-off:** More flexibility (three models, each fit to its own patterns) vs less training data per model (~30k examples each instead of 100k).

---

# Today's Activity: Regression with scikit-learn

Work through **Activity 03** step by step:

## Part 1: Baseline Linear Model
- Load SDSS data (100k objects, 17 features)
- Predict **z-band magnitude** from colors (u, g, r, i)
- Evaluate with **MSE, R²**

## Part 2: Diagnose with Residuals
- Plot residuals vs predicted
- Is there a pattern? Curved? Fanned out?
- What does that tell you?

## Part 3: Add Information
- Add **redshift** as a feature
- Compare R² before and after
- Do residuals look better? Why?

## Part 4: Different Approach
- Build **per-class models** (STAR, GALAXY, QSO separately)
- Compare accuracy per class
- Trade-off: flexibility vs sample size per model

## Part 5: Beyond Linear
- Try **Random Forest Regression**
- Compare performance to Linear
- Why might it be better? Overfitting risk?

> **The goal:** Understand the whole workflow. Pick one question that interests you and dig deep.

> **Notes:** [The Importance of Visualization](../notes/importance_of_visualization.ipynb) · [Methods & Validation](../notes/methods_and_validation.ipynb)
