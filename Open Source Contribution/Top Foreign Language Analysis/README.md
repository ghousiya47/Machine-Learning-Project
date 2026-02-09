# Top Foreign Languages Analysis – Machine Learning Regression (Open Source Project)

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

## Model Evaluation Results
| Model                     | MSE_train | R2_train | MSE_test  | R2_test   |
|---------------------------|-----------|----------|-----------|-----------|
| Random Forest Regression  | 7.03      | 0.93     | 57.90     | 0.51      |
| Linear Regression         | 17.09     | 0.84     | 60.72     | 0.49      |
| Ridge Regression          | 85.65     | 0.22     | 96.16     | 0.20      |
| Elastic Net Regression    | 105.0     | 0.04     | 114.7     | 0.047     |
| Decision Tree Regression  | 0.00      | 1.00     | 61.30     | 0.49      |
| Deep NN                   | 34.29     | 0.04     | 114.7     | 0.0471    |


## Observations
- **Random Forest and Linear Regression** showed comparatively better generalization on the test dataset.
- Decision Tree models achieved strong training performance but showed overfitting.
- Neural Network regression exhibited higher error and limited predictive performance for this dataset.

## Tools & Libraries
Python, Pandas, NumPy, Scikit-learn, XGBoost, TensorFlow, Keras, Seaborn, Plotly

## Original Repository
[ML-Crate — Top Foreign Languages Analysis](https://github.com/abhisheks008/ML-Crate/tree/main/Top%20Foreign%20Languages%20Analysis)

> My contributions are visible in the commit history of the original repository.


**Author**

*Ghousiya Begum*

[![LinkedIn](https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ghousiya-begum-a9b634258/)  [![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ghousiya47)

