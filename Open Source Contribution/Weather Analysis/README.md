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
|Random Forest Regression	  | 0.0126    |	0.965291 | 0.082938	 | 0.773470  |
|XGBoost Regression	        | 0.0056    |	0.984504 | 0.089369	 | 0.755905  |
|Decision Tree	            | 0.58e-34  | 1.000000 | 0.144070	 | 0.606500  |
|Riddge Regression	        | 3.58e-34	| 1.000000 | 0.144070  | 0.606500  |
|Linear Regression	        | 0.274    	| 0.243614 | 0.281541  | 0.231021  |
|Elastic Net Regression	    | 2.94e-01	| 0.190594 | 0.302724	 | 0.173166  |
|Neural Network Regression	| 0.358     | 0.076272 | 0.405645	 |-0.107945  |


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

