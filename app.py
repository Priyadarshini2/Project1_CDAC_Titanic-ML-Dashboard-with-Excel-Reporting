import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)

from io import BytesIO


# ==========================
# Page Configuration
# ==========================

st.set_page_config(
    page_title="Titanic Survival Prediction",
    layout="wide"
)

st.title("Titanic Survival Prediction Dashboard")


# ==========================
# Load Dataset
# ==========================

df = sns.load_dataset("titanic")

df.insert(0, "Serial_Number", range(1, len(df) + 1))

raw_df = df.copy()


# ==========================
# Sidebar
# ==========================

st.sidebar.title("Dashboard Controls")

model_name = st.sidebar.selectbox(
    "Select Machine Learning Model",
    [
        "Random Forest",
        "Logistic Regression",
        "Decision Tree",
        "SVM"
    ]
)

test_size = st.sidebar.slider(
    "Select Test Size",
    min_value=0.10,
    max_value=0.40,
    value=0.20,
    step=0.05
)


# ==========================
# Dataset Overview
# ==========================

st.header("1. Dataset Overview")

st.subheader("Dataset Used")

st.write("Dataset Name : Titanic Dataset")
st.write("Source : Seaborn Built-in Dataset")
st.write(f"Rows : {raw_df.shape[0]}")
st.write(f"Columns : {raw_df.shape[1]}")

st.subheader("Original Dataset")

st.dataframe(
    raw_df,
    use_container_width=True
)

csv = raw_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Original Titanic Dataset",
    data=csv,
    file_name="Titanic_Dataset.csv",
    mime="text/csv"
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Rows", raw_df.shape[0])

with col2:
    st.metric("Total Columns", raw_df.shape[1])

with col3:
    st.metric("Target Column", "survived")

st.subheader("Missing Values")

missing_df = pd.DataFrame({
    "Column": raw_df.columns,
    "Missing Values": raw_df.isnull().sum().values
})

st.dataframe(
    missing_df,
    use_container_width=True
)


# ==========================
# Data Visualization
# ==========================

st.header("2. Data Visualization")

col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots()
    sns.countplot(data=raw_df, x="survived", ax=ax1)
    ax1.set_title("Survival Count")
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots()
    sns.countplot(data=raw_df, x="sex", hue="survived", ax=ax2)
    ax2.set_title("Survival by Gender")
    st.pyplot(fig2)

col3, col4 = st.columns(2)

with col3:
    fig3, ax3 = plt.subplots()
    sns.histplot(data=raw_df, x="age", hue="survived", kde=True, ax=ax3)
    ax3.set_title("Age Distribution by Survival")
    st.pyplot(fig3)

with col4:
    fig4, ax4 = plt.subplots()
    sns.countplot(data=raw_df, x="pclass", hue="survived", ax=ax4)
    ax4.set_title("Survival by Passenger Class")
    st.pyplot(fig4)

st.subheader("Correlation Heatmap")

fig5, ax5 = plt.subplots(figsize=(10, 5))
sns.heatmap(
    raw_df.select_dtypes(include=np.number).corr(),
    annot=True,
    cmap="coolwarm",
    ax=ax5
)
ax5.set_title("Correlation Heatmap")
st.pyplot(fig5)


# ==========================
# Data Preprocessing
# ==========================

target = "survived"

drop_cols = [
    "alive",
    "who",
    "adult_male",
    "deck",
    "embark_town",
    "class"
]

df = df.drop(columns=drop_cols)

y = df[target]

X = df.drop(columns=[target])

serial_numbers = X["Serial_Number"]

X = X.drop(columns=["Serial_Number"])

le_dict = {}

for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = X[col].astype(str)
    X[col] = le.fit_transform(X[col])
    le_dict[col] = le

num_cols = X.select_dtypes(include=np.number).columns

imputer = SimpleImputer(strategy="median")
X[num_cols] = imputer.fit_transform(X[num_cols])


# ==========================
# Train Test Split
# ==========================

X_train, X_test, y_train, y_test, sn_train, sn_test = train_test_split(
    X,
    y,
    serial_numbers,
    test_size=test_size,
    random_state=42,
    stratify=y
)


# ==========================
# Model Selection
# ==========================

if model_name == "Random Forest":
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

elif model_name == "Logistic Regression":
    model = LogisticRegression(
        max_iter=1000
    )

elif model_name == "Decision Tree":
    model = DecisionTreeClassifier(
        random_state=42
    )

else:
    model = SVC(
        probability=True,
        random_state=42
    )

model.fit(X_train, y_train)


# ==========================
# Predictions
# ==========================

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_prob = model.predict_proba(X_train)[:, 1]
test_prob = model.predict_proba(X_test)[:, 1]


# ==========================
# Metrics
# ==========================

cm = confusion_matrix(y_test, test_pred)

tn, fp, fn, tp = cm.ravel()

accuracy = accuracy_score(y_test, test_pred)
precision = precision_score(y_test, test_pred)
recall = recall_score(y_test, test_pred)
f1 = f1_score(y_test, test_pred)

fpr_metric = fp / (fp + tn)
tpr_metric = tp / (tp + fn)

roc_auc = roc_auc_score(y_test, test_prob)

fpr, tpr, thresholds = roc_curve(y_test, test_prob)


# ==========================
# Model Evaluation
# ==========================

st.header("3. Model Evaluation")

st.success(f"Selected Model: {model_name}")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Accuracy", round(accuracy, 4))

with col2:
    st.metric("Precision", round(precision, 4))

with col3:
    st.metric("Recall", round(recall, 4))

with col4:
    st.metric("F1 Score", round(f1, 4))

col5, col6, col7 = st.columns(3)

with col5:
    st.metric("FPR", round(fpr_metric, 4))

with col6:
    st.metric("TPR", round(tpr_metric, 4))

with col7:
    st.metric("ROC-AUC", round(roc_auc, 4))


# ==========================
# Confusion Matrix and ROC
# ==========================

col1, col2 = st.columns(2)

with col1:
    st.subheader("Confusion Matrix")

    fig6, ax6 = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax6
    )
    ax6.set_xlabel("Predicted")
    ax6.set_ylabel("Actual")
    st.pyplot(fig6)

with col2:
    st.subheader("ROC Curve")

    fig7, ax7 = plt.subplots()
    ax7.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}")
    ax7.plot([0, 1], [0, 1], linestyle="--")
    ax7.set_xlabel("False Positive Rate")
    ax7.set_ylabel("True Positive Rate")
    ax7.set_title("ROC Curve")
    ax7.legend()
    st.pyplot(fig7)


# ==========================
# Train and Test Results
# ==========================

st.header("4. Prediction Results")

train_df = X_train.copy()
train_df.insert(0, "Serial_Number", sn_train.values)
train_df["Actual"] = y_train.values
train_df["Predicted"] = train_pred
train_df["Probability"] = train_prob

test_df = X_test.copy()
test_df.insert(0, "Serial_Number", sn_test.values)
test_df["Actual"] = y_test.values
test_df["Predicted"] = test_pred
test_df["Probability"] = test_prob

st.subheader("Training Prediction Data")
st.dataframe(train_df.head(20), use_container_width=True)

st.subheader("Testing Prediction Data")
st.dataframe(test_df.head(20), use_container_width=True)


# ==========================
# Model Comparison
# ==========================

st.header("5. Model Comparison")

models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

comparison_results = []

for name, clf in models.items():
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    prob = clf.predict_proba(X_test)[:, 1]

    comparison_results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1 Score": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, prob)
    })

comparison_df = pd.DataFrame(comparison_results)

st.dataframe(comparison_df, use_container_width=True)

fig8, ax8 = plt.subplots(figsize=(10, 5))
comparison_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]].plot(
    kind="bar",
    ax=ax8
)
ax8.set_title("Model Comparison")
ax8.set_ylabel("Score")
ax8.set_ylim(0, 1)
st.pyplot(fig8)


# ==========================
# Excel Export
# ==========================

cm_df = pd.DataFrame(
    cm,
    columns=["Predicted_0", "Predicted_1"],
    index=["Actual_0", "Actual_1"]
)

eval_df = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "False Positive Rate",
        "True Positive Rate",
        "ROC-AUC"
    ],
    "Value": [
        accuracy,
        precision,
        recall,
        f1,
        fpr_metric,
        tpr_metric,
        roc_auc
    ]
})

roc_df = pd.DataFrame({
    "FPR": fpr,
    "TPR": tpr,
    "Threshold": thresholds
})


def create_excel():

    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:

        raw_df.to_excel(writer, sheet_name="Raw_Data", index=False)
        train_df.to_excel(writer, sheet_name="Train_Data", index=False)
        test_df.to_excel(writer, sheet_name="Test_Data", index=False)
        cm_df.to_excel(writer, sheet_name="Confusion_Matrix")
        eval_df.to_excel(writer, sheet_name="Model_Evaluation", index=False)
        roc_df.to_excel(writer, sheet_name="ROC_Curve_Data", index=False)
        comparison_df.to_excel(writer, sheet_name="Model_Comparison", index=False)

    output.seek(0)

    return output


excel_file = create_excel()

st.header("6. Download Report")

st.download_button(
    label="Download Excel Report",
    data=excel_file,
    file_name="Titanic_Model_Report.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
