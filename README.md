# COSC2669/COSC2816 — Individual Task 1 & Task 2 Data Analysis

This repository contains the code used to produce the machine learning
analysis in Individual Task 1 (Part 1.3) and Individual Task 2 (Part 2)
for Case Studies in Data Science, RMIT.

**Author:** Darshan Nagaraja (s4188277)

## Overview

**Task 1 — Part 1.3:** Two machine learning algorithms — **Logistic
Regression** and **Random Forest** — were trained and evaluated
separately on two datasets relevant to a credit risk / fraud detection
data scientist role:

1. **Credit Card Fraud Detection** (Kaggle/ULB)
   Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. **Statlog German Credit Data** (UCI Machine Learning Repository)
   Source: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

**Task 2 — Part 2:** Extends the Task 1 analysis with a deliberation on
bias and methodology, specifically:
- 5-fold stratified cross-validation, to test whether the original
  70/30 train-test split gave an unbiased performance estimate
- Learning curve analysis, showing model performance (F1-score) across
  varying training set sizes
- A fairness/bias audit using **Fairlearn**, using `foreign_worker` as
  a sensitive attribute on the German Credit dataset

The raw dataset files are **not included** in this repository due to file
size (the fraud dataset is ~98MB) and licensing — download them directly
from the source links above and place them in the repository root as:
- `creditcard.csv`
- `german_credit_named.csv` (see preprocessing note below)

## Files

**Task 1:**
- `run_analysis.py` — main script: loads both datasets, preprocesses,
  trains Logistic Regression and Random Forest models, evaluates with
  precision/recall/F1/ROC-AUC, and saves results to `outputs/results.json`.
- `outputs/results.json` — saved model evaluation results.
- `outputs/fig_model_comparison.png` — bar chart comparing model metrics
  across both datasets.

**Task 2:**
- `run_cv_analysis.py` — 5-fold stratified cross-validation for both
  models on both datasets. Saves results to `outputs/cv_results.json`.
- `run_learning_curve.py` — learning curve analysis (Random Forest,
  F1-score vs. training set size) on both datasets. Saves results to
  `outputs/learning_curve_results.json` and `outputs/learning_curves.png`.
- `run_fairlearn.py` — Fairlearn bias analysis on the German Credit
  dataset, using `foreign_worker` as the sensitive attribute. Saves
  results to `outputs/fairlearn_results.json` and
  `outputs/fairlearn_chart.png`.

## Reproducing the analysis

```bash
pip install pandas scikit-learn matplotlib numpy fairlearn
python run_analysis.py
python run_cv_analysis.py
python run_learning_curve.py
python run_fairlearn.py
```

## Preprocessing notes

- The German Credit dataset was downloaded in its raw `.data` form and
  assigned column names based on the official Statlog attribute
  description (`german.names`), then saved as `german_credit_named.csv`
  before running any script.
- Task 1 used a 70/30 stratified train-test split with
  `class_weight='balanced'` in both models to address class imbalance.
- Task 2's cross-validation and learning curve analysis used a reduced
  Random Forest configuration (fewer estimators, limited max depth) for
  the Fraud dataset specifically, to keep runtime tractable on the
  284,807-row dataset; the primary Task 1 results use the original,
  full-complexity configuration.

## Key findings (Task 2)

- Cross-validation results closely matched the original Task 1 single-split
  results for both datasets, with low variance on the Fraud dataset and
  higher variance on the smaller German Credit dataset.
- Learning curves showed the German Credit model overfitting (large,
  persistent train-validation gap), consistent with its small size (1,000
  records).
- The Fairlearn bias analysis on `foreign_worker` returned a large
  equalized odds difference (0.193), but this must be interpreted with
  caution: the minority group had only 9 individuals in the test set,
  making the result statistically unreliable rather than firm evidence
  of bias.

## AI usage disclosure

Code in this repository was structured with the assistance of Claude
(Anthropic), under the Condition 3 Bounded Process permitted in the
assignment specification. Model outputs (metrics, feature importances,
fairness metrics) are real results from code execution, not fabricated.
See the AI Attribution statement and Condition 3 Declaration Form
submitted with the main reports for full details.
