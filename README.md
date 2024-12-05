# Employee Burnout Analysis and Prediction

This project focuses on analyzing and predicting employee burnout using various employee features. The goal is to identify key factors contributing to burnout and develop a predictive model to help organizations proactively address this issue.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Data Preprocessing and Feature Engineering](#data-preprocessing-and-feature-engineering)
5. [Model Development](#model-development)
6. [Model Evaluation](#model-evaluation)
7. [Visualization and Results](#visualization-and-results)
8. [Conclusion](#conclusion)
9. [Future Work](#future-work)
10. [Setup and Installation](#setup-and-installation)


## Project Overview

The aim of this project is to analyze the factors that contribute to employee burnout and develop a predictive model using machine learning. Burnout is a significant issue for organizations, leading to reduced productivity, increased turnover, and poor employee well-being. By predicting burnout levels, companies can take proactive steps to mitigate the risks and improve employee satisfaction.

## Dataset Description

The dataset used in this project contains employee data, including attributes such as:

- **Employee ID**: Unique identifier for each employee.
- **Company Type**: The type of company the employee works for.
- **Gender**: The gender of the employee.
- **Work Hours**: The number of hours worked by the employee.
- **Employee Satisfaction**: Satisfaction level of the employee.
- **Date of Joining**: The date the employee joined the company.
- **Burn Rate**: Target variable representing the level of burnout.

The data is cleaned, preprocessed, and transformed to create meaningful features that can be used for prediction.

### Features and Target Variable
- **Features**: Employee-related data such as work hours, employee satisfaction, company type, and other categorical variables.
- **Target Variable**: `Burn Rate`, which is the level of employee burnout.

### Data Format and Structure
- The data is in Excel format (`data.xlsx`), with rows representing individual employees and columns representing various attributes of each employee.

## Exploratory Data Analysis (EDA)

### Data Inspection
We performed an initial inspection of the data to understand its structure, look for missing values, and identify potential outliers. We explored the dataset with summary statistics, value counts, and data types.

### Missing Values and Correlations
During EDA, we examined missing values and handled them appropriately. We also performed correlation analysis to understand relationships between numeric features and the target variable (`Burn Rate`).

### Visualizations
Various visualizations were created, including:
- Pair plots to analyze relationships between different features.
- Correlation heatmaps to identify strong relationships with the target variable.
- Bar plots to visualize the trend of employee burnout over time.

## Data Preprocessing and Feature Engineering

### Handling Missing Values
We handled missing data by dropping rows with missing values or imputing where necessary. This step ensured that the dataset was clean and ready for model training.

### Encoding Categorical Features
Categorical features such as `Company Type`, `WFH Setup Available`, and `Gender` were encoded using one-hot encoding.

### Feature Selection
We performed feature selection by analyzing feature importance through correlation matrices and identifying which features were most relevant for predicting burnout.

## Model Development

We chose a **Linear Regression** model to predict employee burnout, as it provides a straightforward approach to understand feature impacts and interpretability. The dataset was split into training and testing sets, and the features were scaled using `StandardScaler` to normalize the values.

The model was trained on the processed data, and predictions were made on the test set to evaluate its performance.

## Model Evaluation

### Performance Metrics
The model's performance was evaluated using the following metrics:
- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**
- **R-squared (R2)**

These metrics provided insights into how well the model was able to predict employee burnout.

### Insights
From the model evaluation, we learned which features contributed most to burnout prediction and assessed the model’s accuracy.

## Visualization and Results

### Burnout Trends
We visualized the trends of burnout over time, identifying whether certain months had higher burnout rates and what factors were contributing to these trends.

### Correlation Heatmaps and Feature Insights
The heatmap visualizations helped in understanding how different features correlated with burnout levels, allowing for a clearer picture of which factors were most influential.

## Conclusion

The project successfully analyzed the factors contributing to employee burnout and built a predictive model. Key features such as `Work Hours`, `Employee Satisfaction`, and `Company Type` were found to significantly influence the level of burnout. The model developed can help organizations predict burnout and take proactive steps to mitigate it.

## Future Work

- **Exploring Advanced Models**: Experimenting with advanced models such as Random Forest or Gradient Boosting to improve prediction accuracy.
- **Hyperparameter Tuning**: Fine-tuning model parameters to enhance performance.
- **Incorporating Additional Data**: Adding qualitative data like employee feedback for better predictions.
- **Real-Time Predictions**: Deploying the model for real-time predictions to proactively manage employee well-being.

## Setup and Installation

### Prerequisites

- Python 3.x
- Required libraries (listed below)


## Usage

Follow the steps below to use the project and execute the model training, evaluation, and predictions.

### Step 1: Data Preprocessing

Before training the model, you need to preprocess the raw dataset. This step includes handling missing values, encoding categorical features, and scaling the numeric features.

Run the `data_preprocessing.py` script: 

```bash
python data_preprocessing.py
```

This will:
- Load the dataset (`data.xlsx`).
- Handle missing values.
- Encode categorical features.
- Save the preprocessed data as CSV files (`X_train_processed.csv`, `y_train_processed.csv`) in the `../data/processed/` directory.

### Step 2: Model Training

Once the data is preprocessed, you can train the Linear Regression model. The training script loads the preprocessed data, splits it into training and testing sets, and trains the model.

Run the `train_model.py` script:

```bash
python train_model.py
```

This will:
- Load the preprocessed training data (`X_train_processed.csv` and `y_train_processed.csv`).
- Train a Linear Regression model.
- Save the trained model as `linear_model.pkl` in the `../models/` directory.
- Save the scaler as `scaler.pkl` to ensure consistent scaling during future predictions.

### Step 3: Model Evaluation

After training the model, you can evaluate its performance on the test data.

Run the `evaluate_model.py` script:

```bash
python evaluate_model.py
```


This will:
- Load the saved model and scaler.
- Load the preprocessed test data.
- Make predictions using the trained model.
- Calculate and display evaluation metrics (Mean Squared Error, Mean Absolute Error, R-squared).

### Step 4: Generating Predictions

Once the model is trained and evaluated, you can use it to predict employee burnout on new data.

Run the `predict_burnout.py` script:

```bash
python predict_burnout.py
```

This will:
- Load the trained model and scaler.
- Accept new employee data (must be preprocessed and scaled).
- Predict the burnout level using the trained model.
- Output the predicted burnout value.




