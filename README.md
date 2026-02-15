# Adult Income Classification using Machine Learning

## a. Problem Statement

The objective of this project is to build and compare multiple machine
learning classification models to predict whether an individual's annual
income exceeds \$50K based on demographic and employment-related
attributes.

The task is a binary classification problem where the target variable
represents income class: - \<=50K - \>50K

The goal is to evaluate and compare different classification algorithms
using multiple performance metrics and deploy the best-performing model
using a Streamlit web application.

------------------------------------------------------------------------

## b. Dataset Description

The dataset used is the Adult Income Dataset (UCI Repository).

Dataset Characteristics: - Total Instances (after cleaning): 30,162 -
Total Features (excluding target): 14 - Target Variable: income -
Classification Type: Binary Classification

Data Cleaning Performed: - Missing values represented as "?" were
replaced with NaN. - Rows containing missing values were removed. -
Final cleaned dataset shape: (30162, 15)

Train-Test Split: - Training set size: 24,129 instances - Test set size:
6,033 instances - Split ratio: 80% training, 20% testing - Stratified
split used to maintain class balance

------------------------------------------------------------------------

## c. Models Used and Performance Comparison

The following six classification models were implemented and evaluated
on the same dataset:

1.  Logistic Regression\
2.  Decision Tree Classifier\
3.  K-Nearest Neighbors (KNN)\
4.  Naive Bayes (Gaussian)\
5.  Random Forest (Ensemble)\
6.  XGBoost (Ensemble)

Evaluation Metrics Used: - Accuracy\
- AUC Score\
- Precision\
- Recall\
- F1 Score\
- Matthews Correlation Coefficient (MCC)

### Model Performance Comparison Table

  --------------------------------------------------------------------------------------
  ML Model Name    Accuracy    AUC        Precision     Recall     F1         MCC
  ---------------- ----------- ---------- ------------- ---------- ---------- ----------
  Logistic         0.854301    0.913589   0.750201      0.621838   0.680015   0.591088
  Regression                                                                  

  Decision Tree    0.815183    0.751002   0.630303      0.623169   0.626716   0.503925

  KNN              0.834079    0.867278   0.683248      0.621838   0.651098   0.543609

  Naive Bayes      0.601028    0.830016   0.379494      0.948735   0.542134   0.387561

  Random Forest    0.856290    0.910527   0.749020      0.635819   0.687793   0.598636
  (Ensemble)                                                                  

  XGBoost          0.872866    0.934065   0.789598      0.667111   0.723205   0.645282
  (Ensemble)                                                                  
  --------------------------------------------------------------------------------------

------------------------------------------------------------------------

## d. Observations About Model Performance

  -----------------------------------------------------------------------
  ML Model Name        Observation about Model Performance
  -------------------- --------------------------------------------------
  Logistic Regression  Performs strongly with high AUC and good overall
                       balance. Indicates dataset has moderately linear
                       decision boundaries.

  Decision Tree        Lower AUC compared to other models. Likely
                       overfitting due to high variance.

  KNN                  Performs moderately well but sensitive to feature
                       scaling and dimensionality.

  Naive Bayes          High recall but very low precision, indicating
                       strong bias and independence assumption
                       limitations.

  Random Forest        Improves over single Decision Tree by reducing
  (Ensemble)           variance. Strong overall performance.

  XGBoost (Ensemble)   Best performing model across all metrics. Boosting
                       effectively captures complex patterns and improves
                       predictive power.
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## Conclusion

Among all implemented models, XGBoost achieved the highest Accuracy
(0.8729) and AUC (0.9341), making it the best-performing model for this
dataset.

Ensemble-based methods (Random Forest and XGBoost) outperformed
individual classifiers, demonstrating the effectiveness of ensemble
learning techniques for structured tabular data.
