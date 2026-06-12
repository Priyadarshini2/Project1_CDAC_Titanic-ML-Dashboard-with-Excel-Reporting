# Titanic Survival Prediction Dashboard

## Project Objective

The objective of this project is to predict passenger survival on the Titanic dataset using Machine Learning techniques and present the results through an interactive Streamlit dashboard.

## Dataset

* Dataset: Titanic Dataset (Seaborn)
* Total Records: 891
* Target Variable: Survived (0 = No, 1 = Yes)

## Data Preprocessing

* Missing value treatment
* Categorical variable encoding
* Feature selection
* Serial Number creation for tracking records
* Removal of target variable from predictor set

## Train-Test Split

* Training Data: 80%
* Testing Data: 20%
* Random State: 42

## Machine Learning Model

### Random Forest Classifier

The Random Forest algorithm was used to build the classification model because it:

* Handles complex relationships effectively
* Reduces overfitting through ensemble learning
* Provides strong classification performance

## Model Evaluation

### Confusion Matrix

| Actual \ Predicted | 0  | 1  |
| ------------------ | -- | -- |
| Actual 0           | 96 | 14 |
| Actual 1           | 18 | 51 |

### Performance Metrics

| Metric                    | Value  |
| ------------------------- | ------ |
| Accuracy                  | 82.12% |
| Precision                 | 78.46% |
| Recall                    | 73.91% |
| F1 Score                  | 76.12% |
| False Positive Rate (FPR) | 12.73% |
| True Positive Rate (TPR)  | 73.91% |

## ROC-AUC Analysis

ROC Curve was generated using prediction probabilities from the Random Forest model to evaluate classification performance across different thresholds.

## Streamlit Dashboard Features

* Titanic Dataset Visualization
* Model Training
* Prediction Generation
* Confusion Matrix Display
* Accuracy, Precision, Recall and F1 Score Calculation
* ROC-AUC Curve Visualization
* Excel Report Generation

## Excel Reporting

The application automatically exports:

### Sheet 1

Raw_Data

### Sheet 2

Train_Data

### Sheet 3

Test_Data

### Sheet 4

Confusion_Matrix

### Sheet 5

Model_Evaluation

### Sheet 6

ROC_Curve_Data

## Technologies Used

* Python
* Pandas
* NumPy
* Seaborn
* Matplotlib
* Scikit-Learn
* Streamlit
* XlsxWriter
* GitHub

## Author

Priyadarshini Behera

CDAC Machine Learning Project
