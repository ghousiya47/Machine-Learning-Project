# Weather Analysis – Machine Learning Regression (Open Source Project)

This project explores the use of machine learning regression models to predict **average monthly rainfall** based on atmospheric conditions such as temperature, humidity, pressure, and wind-related features.

## Problem Statement
To analyze historical weather data and evaluate different supervised machine learning models for predicting **average rainfall per month**.

## Dataset
Source: Kaggle  
https://www.kaggle.com/datasets/mastmustu/weather-analysis  

- 3,902 records
- 22 meteorological features

## Approach
- Data cleaning and preprocessing
- Exploratory data analysis (EDA) and feature analysis
- Training and evaluation of multiple regression models
- Model comparison using **MSE** and **R²** metrics

## Machine Learning Models Used
- Random Forest Regression  
- XGBoost Regression  
- Linear Regression  
- Ridge Regression  
- Elastic Net Regression  
- Decision Tree Regression  
- Neural Network Regression (TensorFlow / Keras)

**MODELS Evaluation Results**

| Model                     | MSE_train | R2_train | MSE_test  | R2_test   |
|---------------------------|-----------|----------|-----------|-----------|
| Random Forest Regression  | 7.03      | 0.93     | 57.90     | 0.51      |
| Linear Regression         | 17.09     | 0.84     | 60.72     | 0.49      |
| Ridge Regression          | 85.65     | 0.22     | 96.16     | 0.20      |
| Elastic Net Regression    | 105.0     | 0.04     | 114.7     | 0.047     |
| Decision Tree Regression  | 0.00      | 1.00     | 61.30     | 0.49      |
| Deep NN                   | 34.29     | 0.04     | 114.7     | 0.0471    |

## Observations
- Ensemble-based models such as **Random Forest and XGBoost** showed lower error and more stable performance on the test set.
- Decision Tree models achieved very high training performance but showed signs of overfitting.
- Neural Network regression did not generalize well on this dataset.

## Tools & Libraries
Python, Pandas, NumPy, Scikit-learn, XGBoost, TensorFlow, Keras, Matplotlib, Seaborn

## Original Repository
[ML-Crate — Weather Analysis](https://github.com/abhisheks008/ML-Crate/tree/main/Weather%20Analysis)

> My contributions are visible in the commit history of the original repository.

**Author**

*Ghousiya Begum*

[![LinkedIn](https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ghousiya-begum-a9b634258/)  [![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ghousiya47)

