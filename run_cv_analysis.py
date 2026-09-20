"""
Individual Task 2 - Part 2: Deliberation on Task 1
Cross-validation evaluation for Random Forest and Logistic Regression
on the Credit Card Fraud and German Credit datasets.
"""
import pandas as pd
import numpy as np
import json
import time

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
results = {}

df1 = pd.read_csv('creditcard.csv')
X1 = df1.drop(columns=['Class'])
y1 = df1['Class']

df2 = pd.read_csv('german_credit_named.csv')
df2['target'] = df2['class'].map({1: 0, 2: 1})
df2 = df2.drop(columns=['class'])
X2 = pd.get_dummies(df2.drop(columns=['target']), drop_first=True)
y2 = df2['target']

datasets = {'credit': (X2, y2), 'fraud': (X1, y1)}  # credit (small) first

models = {
    'lr': lambda: Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE))
    ]),
    'rf': lambda: RandomForestClassifier(n_estimators=50, max_depth=12, class_weight='balanced',
                                          random_state=RANDOM_STATE, n_jobs=-1),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = ['precision', 'recall', 'f1', 'roc_auc']

for ds_name, (X, y) in datasets.items():
    for model_name, model_fn in models.items():
        t0 = time.time()
        model = model_fn()
        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=2)
        key = f"{ds_name}_{model_name}_cv"
        results[key] = {
            metric: {
                'mean': round(float(np.mean(cv_results[f'test_{metric}'])), 4),
                'std': round(float(np.std(cv_results[f'test_{metric}'])), 4),
                'folds': [round(float(v), 4) for v in cv_results[f'test_{metric}']]
            }
            for metric in scoring
        }
        print(f"{ds_name} - {model_name}: done in {time.time()-t0:.1f}s", flush=True)
        with open('outputs/cv_results.json', 'w') as f:
            json.dump(results, f, indent=2)

print("ALL DONE")
