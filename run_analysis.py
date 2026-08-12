"""
Individual Task 1 - Part 1.3 Data Analysis
Random Forest + Logistic Regression on:
  1. Credit Card Fraud Detection (Kaggle/ULB)
  2. Statlog German Credit Data (UCI)
"""
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, average_precision_score
)

RANDOM_STATE = 42
results = {}

def evaluate_model(name, y_true, y_pred, y_proba):
    return {
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
        "pr_auc": round(average_precision_score(y_true, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

# ============================================================
# DATASET 1: Credit Card Fraud Detection
# ============================================================
print("="*60)
print("DATASET 1: Credit Card Fraud Detection")
print("="*60)

df1 = pd.read_csv('creditcard.csv')
X1 = df1.drop(columns=['Class'])
y1 = df1['Class']

X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1, y1, test_size=0.3, random_state=RANDOM_STATE, stratify=y1
)

scaler1 = StandardScaler()
X1_train_scaled = scaler1.fit_transform(X1_train)
X1_test_scaled = scaler1.transform(X1_test)

# Logistic Regression (class_weight balanced due to severe imbalance)
lr1 = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE)
lr1.fit(X1_train_scaled, y1_train)
y1_pred_lr = lr1.predict(X1_test_scaled)
y1_proba_lr = lr1.predict_proba(X1_test_scaled)[:, 1]
results['fraud_lr'] = evaluate_model('LR-Fraud', y1_test, y1_pred_lr, y1_proba_lr)
print("Logistic Regression:", results['fraud_lr'])

# Random Forest
rf1 = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
rf1.fit(X1_train, y1_train)
y1_pred_rf = rf1.predict(X1_test)
y1_proba_rf = rf1.predict_proba(X1_test)[:, 1]
results['fraud_rf'] = evaluate_model('RF-Fraud', y1_test, y1_pred_rf, y1_proba_rf)
print("Random Forest:", results['fraud_rf'])

# Feature importance (RF) - top 5
feat_imp1 = pd.Series(rf1.feature_importances_, index=X1.columns).sort_values(ascending=False)
results['fraud_rf_top_features'] = feat_imp1.head(5).round(4).to_dict()
print("Top features (fraud RF):", results['fraud_rf_top_features'])

# ============================================================
# DATASET 2: Statlog German Credit Data
# ============================================================
print("\n" + "="*60)
print("DATASET 2: Statlog German Credit Data")
print("="*60)

df2 = pd.read_csv('german_credit_named.csv')
# Target: 1 = good credit, 2 = bad credit -> convert to 0/1 (1 = bad/risk, matching "positive class = risk" convention)
df2['target'] = df2['class'].map({1: 0, 2: 1})  # 1 = bad credit risk
df2 = df2.drop(columns=['class'])

X2 = pd.get_dummies(df2.drop(columns=['target']), drop_first=True)
y2 = df2['target']

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.3, random_state=RANDOM_STATE, stratify=y2
)

scaler2 = StandardScaler()
X2_train_scaled = scaler2.fit_transform(X2_train)
X2_test_scaled = scaler2.transform(X2_test)

lr2 = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE)
lr2.fit(X2_train_scaled, y2_train)
y2_pred_lr = lr2.predict(X2_test_scaled)
y2_proba_lr = lr2.predict_proba(X2_test_scaled)[:, 1]
results['credit_lr'] = evaluate_model('LR-Credit', y2_test, y2_pred_lr, y2_proba_lr)
print("Logistic Regression:", results['credit_lr'])

rf2 = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
rf2.fit(X2_train, y2_train)
y2_pred_rf = rf2.predict(X2_test)
y2_proba_rf = rf2.predict_proba(X2_test)[:, 1]
results['credit_rf'] = evaluate_model('RF-Credit', y2_test, y2_pred_rf, y2_proba_rf)
print("Random Forest:", results['credit_rf'])

feat_imp2 = pd.Series(rf2.feature_importances_, index=X2.columns).sort_values(ascending=False)
results['credit_rf_top_features'] = feat_imp2.head(5).round(4).to_dict()
print("Top features (credit RF):", results['credit_rf_top_features'])

# dataset sizes for reference
results['meta'] = {
    'fraud_n': len(df1), 'fraud_fraud_rate': round(y1.mean(), 5),
    'credit_n': len(df2), 'credit_bad_rate': round(y2.mean(), 4)
}

with open('outputs/results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n\nDone. Results saved to outputs/results.json")
