# Predict Bike Sharing Demand with AutoGluon

This project focuses on predicting bike sharing demand using **AutoGluon AutoML**, leveraging temporal and weather-related features. The objective was to explore automated model selection, feature engineering, and hyperparameter tuning to improve prediction accuracy.

This work was completed as part of the **AWS AI & ML Scholarship Program (Amazon)** and was implemented using **AWS SageMaker Studio**.

---

## Problem Statement
Predict the number of bike rentals based on historical usage patterns, date-time information, and environmental factors. The project follows a Kaggle-style workflow with iterative model improvements and evaluation using RMSE.

---

## Approach
- Trained baseline regression models using AutoGluon TabularPredictor
- Performed feature engineering by extracting year, month, day, and hour from datetime
- Converted categorical features (season, weather) to appropriate data types
- Applied hyperparameter optimization (HPO) on multiple model types
- Evaluated models using RMSE and Kaggle leaderboard scores

---

## Model Performance Summary

| Training Stage        | RMSE (Kaggle) |
|----------------------|---------------|
| Baseline Model       | 1.80276       |
| With Feature Engineering | 0.70335       |
| With Hyperparameter Tuning | **0.49091** |

**Best Performing Model:**  
AutoGluon **WeightedEnsemble_L3 (HPO)** achieved the lowest RMSE and the best Kaggle submission score.

---

## Tools & Technologies
- Python
- AutoGluon (Tabular AutoML)
- Pandas, NumPy
- AWS SageMaker Studio
- Kaggle

---

## Detailed Report
For a complete explanation of experiments, feature engineering, hyperparameter tuning, and results:
- 📄 [`REPORT.md`](./REPORT.md)

## Author
**YOUR NAME**

*Ghousiya Begum*

[![LinkedIn](https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ghousiya-begum-a9b634258/)  [![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ghousiya47)
