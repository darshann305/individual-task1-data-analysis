# COSC2669/COSC2816 — Individual Task 1: Part 1.3 Data Analysis

This repository contains the code used to produce the machine learning
analysis in Part 1.3 of Individual Task 1 (Case Studies in Data Science, RMIT).

**Author:** Darshan Nagaraja (s4188277)

## Overview

Two machine learning algorithms — **Logistic Regression** and **Random
Forest** — were trained and evaluated separately on two datasets relevant
to a credit risk / fraud detection data scientist role:

1. **Credit Card Fraud Detection** (Kaggle/ULB)
   Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. **Statlog German Credit Data** (UCI Machine Learning Repository)
   Source: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

The raw dataset files are **not included** in this repository due to file
size (the fraud dataset is ~98MB) and licensing — download them directly
from the source links above and place them in the repository root as:
- `creditcard.csv`
- `german_credit_named.csv` (see preprocessing note below)

## Files

- `run_analysis.py` — main script: loads both datasets, preprocesses,
  trains Logistic Regression and Random Forest models, evaluates with
  precision/recall/F1/ROC-AUC, and saves results to `outputs/results.json`.
- `outputs/results.json` — saved model evaluation results from the run
  used in the final report.
- `outputs/fig_model_comparison.png` — bar chart comparing model metrics
  across both datasets, used in the report (Figure 1, Section 3).

## Reproducing the analysis

```bash
pip install pandas scikit-learn matplotlib numpy
python run_analysis.py
```

## Preprocessing notes

- The German Credit dataset was downloaded in its raw `.data` form and
  assigned column names based on the official Statlog attribute
  description (`german.names`), then saved as `german_credit_named.csv`
  before running the script.
- Both datasets used a 70/30 stratified train-test split with
  `class_weight='balanced'` in both models to address class imbalance.

## AI usage disclosure

Code in this repository was generated with the assistance of Claude
(Anthropic), under the Condition 3 Bounded Process permitted in the
assignment specification. Model outputs (metrics, feature importances)
are real results from code execution, not fabricated. See the AI
Attribution statement and Condition 3 Declaration Form submitted with
the main report for full details.
