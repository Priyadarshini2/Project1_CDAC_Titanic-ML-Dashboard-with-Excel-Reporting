# your Streamlit code here
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pandas.api.types import is_numeric_dtype

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)

# =====================================================
# TITLE
# =====================================================

st.set_page_config(page_title="Titanic Classification App")

st.title("Titanic Survival Prediction")

# =====================================================
# LOAD DATA
# =====================================================

df = sns.load_dataset("titanic")

# Add Serial Number
df.insert(0, "Serial_Number", range(1, len(df) + 1))

st.subheader("Raw Titanic Dataset")

st.dataframe(df.head())

# =====================================================
# DATA CLEANING
# =====================================================

data = df.copy()

# Missing Value Treatment

for col in data.columns:

    if is_numeric_dtype(data[col]):

        data[col] = data[col].fillna(
            data[col].median()
        )

    else:

        data[col] = data[col].fillna(
            data[col].mode()[0]
        )

# Convert bool columns to integer

bool_cols = data.select_dtypes(include=['bool']).columns

for col in bool_cols:
    data[col] = data[col].astype(int)

# Encode categorical variables

le = LabelEncoder()

cat_cols = data.select_dtypes(
    include=['object', 'category', 'string']
).columns

for col in cat_cols:

    data[col] = le.fit_transform(
        data[col].astype(str)
    )

# =====================================================
# FEATURES & TARGET
# =====================================================

target = "survived"

X = data.drop(
    columns=[
        "survived",
        "Serial_Number"
    ]
)

y = data[target]

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Save train/test data including serial number

train_data = data.loc[X_train.index].copy()

test_data = data.loc[X_test.index].copy()

# =====================================================
# MODEL
# =====================================================

model = LogisticRegression(
    max_iter=5000
)

model.fit(X_train, y_train)

# =====================================================
# PREDICTIONS
# =====================================================

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]

# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(y_test, y_pred)

TN, FP, FN, TP = cm.ravel()

# User Requested Format

cm_df = pd.DataFrame(
    [
        [TP, FN],
        [FP, TN]
    ],
    index=["Actual_1", "Actual_0"],
    columns=["Predicted_1", "Predicted_0"]
)

# =====================================================
# METRICS
# =====================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

fpr_value = FP / (FP + TN)

tpr_value = TP / (TP + FN)

roc_auc = roc_auc_score(y_test, y_prob)

metrics_df = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "FPR",
        "TPR",
        "ROC-AUC Score"
    ],
    "Value": [
        accuracy,
        precision,
        recall,
        f1,
        fpr_value,
        tpr_value,
        roc_auc
    ]
})

# =====================================================
# ROC CURVE
# =====
