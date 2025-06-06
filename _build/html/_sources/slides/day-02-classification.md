---
marp: true
theme: king
paginate: true

title: Day 02 - Classification
description: Slides for ML Short Course Summer 2025, Day 02: Classification
author: Prof. Danny Caballero <caball14@msu.edu>
keywords: classification, machine learning, physics, MSU
url: https://dannycaballero.info/MSU_REU_ML_course/slides/day-02-classification.html

---

# Day 02 - Classification

<img src="https://upload.wikimedia.org/wikipedia/commons/7/78/KNN_decision_surface_animation.gif" alt="KNN Decision Surface Animation" width="80%">

---

# SDSS Data Set

<img src="../activities/figures/stellar_color_diagrams.png" alt="Stellar Color Diagrams" width="60%">

---

# SDSS Data Set

<img src="../activities/figures/stellar_histograms.png" alt="Stellar Histograms" width="70%">

---

# SDSS Data Set

<img src="../activities/figures/stellar_redshift_distribution.png" alt="Stellar Redshift Distribution" width="70%">

---

# Classification Task

- Using the SDSS data set, we will classify objects as stars or quasars.
    - At first, we will only use the color information (u-g, g-r, r-i) to classify objects.
    - Later, we will add the redshift information (z) to improve our classification.
- Then, we will perform a 3-class classification to distinguish between stars, quasars, and galaxies; here we will use all available features including redshift.

---

<img src="./figures/ml.png" alt="Machine Learning" width="100%">

---

# Sci-Kit-Learn Classification

<img src="https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png" alt="Sci-Kit-Learn Logo" width="30%">

- Sci-Kit-Learn is a powerful Python library for machine learning.
- It provides a wide range of classification algorithms, including:
    - k-Nearest Neighbors (kNN) & Logistic Regression
    - Decision Trees & Random Forests
    - Support Vector Machines (SVM)
- It also includes tools for model evaluation, such as cross-validation and confusion matrices.

<https://scikit-learn.org/stable/index.html>

---

<img src="./figures/scikit.png" alt="Scikit-learn" width="75%">

---

# K-Nearest Neighbors (kNN)

- kNN is a simple and intuitive classification algorithm.
- It classifies a data point based on the majority class of its k nearest neighbors in the feature space.
- The distance metric (e.g., Euclidean distance) is used to determine the nearest neighbors.
- kNN is a non-parametric method, meaning it makes no assumptions about the underlying data distribution.
- It is sensitive to the choice of k and the distance metric.

---

# K-Nearest Neighbors (kNN)

<img src="https://upload.wikimedia.org/wikipedia/commons/7/78/KNN_decision_surface_animation.gif" alt="KNN Decision Surface Animation" width="80%">

---

# Today's Activity

- We will implement a kNN classifier using Sci-Kit-Learn to classify stars and quasars from the SDSS data set.
- We will:
    1. Load the SDSS data set and preprocess it.
    2. Split the data into training and testing sets.
    3. Train a kNN classifier on the training set.
    4. Evaluate the classifier's performance on the testing set.
- We will focus on the evaluation metrics and visualizations to understand the classifier's performance.