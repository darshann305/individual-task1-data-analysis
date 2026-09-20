"""
Individual Task 2 - Part 2: Bias analysis using Fairlearn
Sensitive attribute: foreign_worker (A201 = yes, A202 = no)
Model: Random Forest on the German Credit dataset
"""
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from fairlearn.metrics import (
    MetricFrame, demographic_parity_difference, demographic_parity_ratio,
    equalized_odds_difference, equalized_odds_ratio,
    selection_rate, false_positive_rate, false_negative_rate
)
from sklearn.metrics import accuracy_score, recall_score, precision_score

RANDOM_STATE = 42

df = pd.read_csv('german_credit_named.csv')
df['target'] = df['class'].map({1: 0, 2: 1})  # 1 = bad credit risk
sensitive = df['foreign_worker'].map({'A201': 'foreign_worker_yes', 'A202': 'foreign_worker_no'})

X = pd.get_dummies(df.drop(columns=['class', 'target', 'foreign_worker']), drop_first=True)
y = df['target']

X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
    X, y, sensitive, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)

rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("Test set sensitive group counts:")
print(sens_test.value_counts())

# Overall metrics by group
metric_frame = MetricFrame(
    metrics={
        'accuracy': accuracy_score,
        'selection_rate': selection_rate,
        'recall': recall_score,
        'precision': precision_score,
        'false_positive_rate': false_positive_rate,
        'false_negative_rate': false_negative_rate,
    },
    y_true=y_test, y_pred=y_pred, sensitive_features=sens_test
)

print("\nBy-group metrics:")
print(metric_frame.by_group)

dp_diff = demographic_parity_difference(y_test, y_pred, sensitive_features=sens_test)
dp_ratio = demographic_parity_ratio(y_test, y_pred, sensitive_features=sens_test)
eo_diff = equalized_odds_difference(y_test, y_pred, sensitive_features=sens_test)
eo_ratio = equalized_odds_ratio(y_test, y_pred, sensitive_features=sens_test)

print(f"\nDemographic parity difference: {dp_diff:.4f}")
print(f"Demographic parity ratio: {dp_ratio:.4f}")
print(f"Equalized odds difference: {eo_diff:.4f}")
print(f"Equalized odds ratio: {eo_ratio:.4f}")

results = {
    'test_group_counts': sens_test.value_counts().to_dict(),
    'by_group_metrics': metric_frame.by_group.round(4).to_dict(),
    'overall_metrics': metric_frame.overall if isinstance(metric_frame.overall, dict) else dict(metric_frame.overall.round(4)),
    'demographic_parity_difference': round(float(dp_diff), 4),
    'demographic_parity_ratio': round(float(dp_ratio), 4),
    'equalized_odds_difference': round(float(eo_diff), 4),
    'equalized_odds_ratio': round(float(eo_ratio), 4),
}

with open('outputs/fairlearn_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

# Chart: selection rate and recall by group
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
groups = metric_frame.by_group.index.tolist()

axes[0].bar(groups, metric_frame.by_group['selection_rate'], color=['#1B998B', '#C9A227'])
axes[0].set_title('Selection Rate\n(predicted "bad credit risk")')
axes[0].set_ylim(0, 1)
axes[0].tick_params(axis='x', rotation=15)

axes[1].bar(groups, metric_frame.by_group['recall'], color=['#1B998B', '#C9A227'])
axes[1].set_title('Recall by Group\n(bad-risk cases correctly flagged)')
axes[1].set_ylim(0, 1)
axes[1].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig('outputs/fairlearn_chart.png', dpi=150)
print("\nSaved fairlearn_chart.png")
