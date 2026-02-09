# Report: Predict Bike Sharing Demand with AutoGluon
**Author:** Ghousiya Begum  
**Platform:** AWS SageMaker Studio  
**Program:** AWS AI & ML Scholarship (Amazon)

---

## Initial Model Training
During the initial submission process, it was observed that predictions needed to follow Kaggle’s strict submission format. This required ensuring correct index alignment with the test dataset and proper output formatting before submission.

The baseline model was trained using AutoGluon’s default settings.

**Initial Performance:**
- Kaggle RMSE: **1.80276**

---

## Best Performing Model
The top-performing model was produced using AutoGluon’s hyperparameter optimization framework.

- **Model:** WeightedEnsemble_L3  
- **Best Kaggle Score:** **0.49091**

This ensemble combined multiple optimized base models and consistently outperformed individual models.

---

## Exploratory Data Analysis & Feature Engineering
Exploratory analysis revealed that the `datetime` feature contained valuable temporal patterns. To capture this information, the following features were engineered:
- Year
- Month
- Day
- Hour

Additionally:
- `season` and `weather` were converted to categorical data types
- `casual` and `registered` features were excluded to reduce multicollinearity

### Impact on Performance
Feature engineering significantly improved generalization and model accuracy.

- RMSE improved from **1.80608 → 0.70335**

---

## Hyperparameter Optimization
Hyperparameter tuning was applied to Gradient Boosting and Neural Network models using AutoGluon’s built-in HPO framework.

Although improvements from HPO were smaller compared to feature engineering, they provided meaningful gains.

- Kaggle RMSE improved from **0.70335 → 0.49091**

---

## Model Comparison Summary

| Model Stage        | Description                                | Kaggle RMSE |
|--------------------|--------------------------------------------|-------------|
| Baseline           | Default AutoGluon configuration            | 1.80276 |
| Feature Engineering| Datetime expansion & categorical fixes     | 0.70335 |
| Hyperparameter Tuning | Optimized ensemble (WeightedEnsemble_L3) | **0.49091** |

---

## Training Performance Across Experiments
The following plot shows the RMSE trends across different training stages:

![Training RMSE Comparison](images/train_img.png)

---

## Kaggle Submission Performance
The following plot illustrates improvements in Kaggle submission scores across experiments:

![Kaggle Score Comparison](images/test_img.png)

---

## Summary
This project demonstrates the application of AutoGluon AutoML for regression tasks in a real-world Kaggle-style workflow. Feature engineering had the greatest impact on performance, while ensemble learning and hyperparameter tuning provided further improvements.

The final model achieved a **Kaggle RMSE of 0.49091**, highlighting the effectiveness of automated model selection combined with domain-aware feature engineering.

The project provided hands-on experience with:
- AutoML workflows
- Feature engineering
- Model evaluation using RMSE
- AWS SageMaker Studio

## Key Learnings
- Feature engineering had the largest impact on performance improvement
- Proper handling of categorical variables improved model generalization
- AutoGluon’s ensemble models consistently outperformed individual models
- Hyperparameter tuning provided incremental but meaningful gains

---

## Conclusion
This project demonstrates the effectiveness of AutoML frameworks such as AutoGluon in rapidly developing competitive regression models. By combining feature engineering with automated hyperparameter tuning, the final model achieved a strong Kaggle score with minimal manual model selection.

The project also provided practical experience with AWS SageMaker and real-world ML evaluation workflows.

---


## Future Improvements
Given more time, future work could include:
- Increasing the number of HPO trials
- Experimenting with additional feature transformations
- Exploring custom evaluation metrics
- Testing alternative ensemble strategies
