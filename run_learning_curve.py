"""
Individual Task 2 - Part 2: Learning curve analysis
Random Forest performance (F1) across varying training set sizes,
for the Fraud Detection dataset (primary) and German Credit dataset.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42

# ---------- Fraud dataset ----------
df1 = pd.read_csv('creditcard.csv')
X1 = df1.drop(columns=['Class'])
y1 = df1['Class']

# ---------- Credit dataset ----------
df2 = pd.read_csv('german_credit_named.csv')
df2['target'] = df2['class'].map({1: 0, 2: 1})
df2 = df2.drop(columns=['class'])
X2 = pd.get_dummies(df2.drop(columns=['target']), drop_first=True)
y2 = df2['target']

train_sizes = np.linspace(0.1, 1.0, 6)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

# Fraud - Random Forest (reduced complexity for tractable runtime)
rf_fraud = RandomForestClassifier(n_estimators=30, max_depth=10, class_weight='balanced',
                                   random_state=RANDOM_STATE, n_jobs=1)
sizes1, train_scores1, val_scores1 = learning_curve(
    rf_fraud, X1, y1, train_sizes=train_sizes, cv=3, scoring='f1',
    n_jobs=1, random_state=RANDOM_STATE
)
axes[0].plot(sizes1, train_scores1.mean(axis=1), 'o-', color='#1B998B', label='Training score')
axes[0].plot(sizes1, val_scores1.mean(axis=1), 'o-', color='#C9A227', label='Validation score')
axes[0].fill_between(sizes1, train_scores1.mean(axis=1)-train_scores1.std(axis=1),
                      train_scores1.mean(axis=1)+train_scores1.std(axis=1), alpha=0.15, color='#1B998B')
axes[0].fill_between(sizes1, val_scores1.mean(axis=1)-val_scores1.std(axis=1),
                      val_scores1.mean(axis=1)+val_scores1.std(axis=1), alpha=0.15, color='#C9A227')
axes[0].set_title('Fraud Detection — Random Forest')
axes[0].set_xlabel('Training set size')
axes[0].set_ylabel('F1-score')
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

# Credit - Random Forest
rf_credit = RandomForestClassifier(n_estimators=50, max_depth=12, class_weight='balanced',
                                    random_state=RANDOM_STATE, n_jobs=1)
sizes2, train_scores2, val_scores2 = learning_curve(
    rf_credit, X2, y2, train_sizes=train_sizes, cv=5, scoring='f1',
    n_jobs=1, random_state=RANDOM_STATE
)
axes[1].plot(sizes2, train_scores2.mean(axis=1), 'o-', color='#1B998B', label='Training score')
axes[1].plot(sizes2, val_scores2.mean(axis=1), 'o-', color='#C9A227', label='Validation score')
axes[1].fill_between(sizes2, train_scores2.mean(axis=1)-train_scores2.std(axis=1),
                      train_scores2.mean(axis=1)+train_scores2.std(axis=1), alpha=0.15, color='#1B998B')
axes[1].fill_between(sizes2, val_scores2.mean(axis=1)-val_scores2.std(axis=1),
                      val_scores2.mean(axis=1)+val_scores2.std(axis=1), alpha=0.15, color='#C9A227')
axes[1].set_title('German Credit — Random Forest')
axes[1].set_xlabel('Training set size')
axes[1].set_ylabel('F1-score')
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/learning_curves.png', dpi=150)
print("Saved learning_curves.png")

# Save raw numbers too
import json
lc_results = {
    'fraud_rf': {
        'train_sizes': sizes1.tolist(),
        'train_f1_mean': train_scores1.mean(axis=1).round(4).tolist(),
        'val_f1_mean': val_scores1.mean(axis=1).round(4).tolist(),
    },
    'credit_rf': {
        'train_sizes': sizes2.tolist(),
        'train_f1_mean': train_scores2.mean(axis=1).round(4).tolist(),
        'val_f1_mean': val_scores2.mean(axis=1).round(4).tolist(),
    }
}
with open('outputs/learning_curve_results.json', 'w') as f:
    json.dump(lc_results, f, indent=2)
print(json.dumps(lc_results, indent=2))
