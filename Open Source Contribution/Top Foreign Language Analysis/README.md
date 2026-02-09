# Top Foreign Languages Analysis – Machine Learning Regression (Collaborative Project)

This project applies machine learning regression techniques to analyze factors influencing the **USD price per lesson** for foreign language tutors.

## Problem Statement
To evaluate supervised ML models for predicting lesson pricing based on tutor attributes such as ratings, reviews, student engagement, and teaching experience.

## Dataset
Source: Kaggle  
https://www.kaggle.com/datasets/timmofeyy/top-foreign-languages-preply-tutors  

- 8 CSV files merged into a single dataset
- 34,442 records
- 47 features

## Approach
- Dataset merging and preprocessing
- Feature analysis and correlation study
- Training and evaluation of regression models
- Performance comparison using **MSE** and **R²**

## Machine Learning Models Used
- Random Forest Regression  
- Linear Regression  
- Ridge Regression  
- Elastic Net Regression  
- Decision Tree Regression  
- Deep Neural Network (TensorFlow / Keras)

## Observations
- **Random Forest and Linear Regression** showed comparatively better generalization on the test dataset.
- Decision Tree models achieved strong training performance but showed overfitting.
- Neural Network regression exhibited higher error and limited predictive performance for this dataset.

## Tools & Libraries
Python, Pandas, NumPy, Scikit-learn, XGBoost, TensorFlow, Keras, Seaborn, Plotly

## Original Repository
[ML-Crate — Top Foreign Languages Analysis](https://github.com/abhisheks008/ML-Crate/tree/main/Top%20Foreign%20Languages%20Analysis)

> My contributions are visible in the commit history of the original repository.
