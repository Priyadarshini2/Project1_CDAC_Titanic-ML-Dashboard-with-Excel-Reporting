# Titanic Survival Prediction Dashboard

## Project Objective

The objective of this project is to predict passenger survival on the Titanic dataset using Machine Learning Classification techniques and present the results through an interactive Streamlit Dashboard. The application allows users to explore the dataset, visualize patterns, train multiple machine learning models, compare model performance, and download detailed Excel reports.

### Dataset Source

The dashboard supports two dataset sources:

1. Built-in Titanic Dataset (Seaborn)
2. User Uploaded CSV Dataset

### Titanic Dataset Information

* Dataset Name: Titanic Dataset
* Source: Seaborn Built-in Dataset
* Total Records: 891
* Target Variable: survived

  * 0 = Did Not Survive
  * 1 = Survived

## Data Preprocessing

The following preprocessing steps are performed:

* Missing Value Treatment using Median Imputation
* Categorical Variable Encoding using Label Encoder
* Feature Selection
* Serial Number Creation for Record Tracking
* Removal of Target Variable from Predictor Set
* Train-Test Data Splitting

## Exploratory Data Analysis (EDA)

The dashboard provides the following visualizations:

### Survival Count

Displays the distribution of survived and non-survived passengers.

### Survival by Gender

Shows survival patterns based on passenger gender.

### Age Distribution

Visualizes age distribution with respect to survival.

### Passenger Class Analysis

Shows survival distribution across passenger classes.

### Correlation Heatmap

Displays correlations among numerical variables.

## Train-Test Split

* Training Data: 80%
* Testing Data: 20%
* Random State: 42
* Stratified Sampling Applied

Users can dynamically change the test size using the dashboard controls.
## Machine Learning Models

The dashboard supports multiple classification algorithms:

### Random Forest Classifier

* Ensemble Learning Technique
* Reduces Overfitting
* High Predictive Accuracy

### Logistic Regression

* Linear Classification Model
* Probability-Based Prediction

### Decision Tree Classifier

* Easy to Interpret
* Tree-Based Classification

### Support Vector Machine (SVM)

* Effective for Complex Classification Boundaries
* Supports Probability Predictions

## Model Evaluation Metrics

The dashboard calculates and displays:

* Accuracy
* Precision
* Recall
* F1 Score
* False Positive Rate (FPR)
* True Positive Rate (TPR)
* ROC-AUC Score

## Confusion Matrix

The dashboard displays an interactive confusion matrix showing:

* True Positives (TP)
* True Negatives (TN)
* False Positives (FP)
* False Negatives (FN)

## ROC Curve Analysis

ROC Curve is generated using prediction probabilities.

The ROC-AUC score helps evaluate classification performance across different threshold values.

## Model Comparison

The dashboard compares all available machine learning models and displays:

* Accuracy Comparison
* Precision Comparison
* Recall Comparison
* F1 Score Comparison
* ROC-AUC Comparison

A visual comparison chart is also provided.

## Streamlit Dashboard Features

### Dataset Management

* Built-in Dataset Selection
* CSV File Upload Option
* Dataset Preview
* Dataset Download

### Data Analysis

* Missing Value Summary
* Dataset Statistics
* Exploratory Data Analysis (EDA)

### Machine Learning

* Model Selection
* Model Training
* Prediction Generation
* Performance Evaluation
* Model Comparison

### Visualization

* Survival Count Plot
* Gender-Based Analysis
* Age Distribution Plot
* Passenger Class Analysis
* Correlation Heatmap
* Confusion Matrix
* ROC Curve

### Reporting

* Excel Report Generation
* Dataset Export
* Prediction Export

## Excel Report Sheets

The generated Excel report contains the following sheets:

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

### Sheet 7

Model_Comparison

## Technologies Used

* Python
* Pandas
* NumPy
* Seaborn
* Matplotlib
* Scikit-Learn
* Streamlit
* OpenPyXL
* GitHub
* Streamlit Community Cloud

## Project Workflow

Dataset Selection
→ Data Preprocessing
→ Feature Engineering
→ Train-Test Split
→ Model Training
→ Prediction
→ Model Evaluation
→ Visualization
→ Model Comparison
→ Excel Report Generation
## Author
Priyadarshini Behera

CDAC Machine Learning Project

Titanic Survival Prediction Dashboard using Streamlit and Machine Learning
