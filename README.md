a. Problem Statement

To build, evaluate, and deploy multiple machine learning classification models to predict heart disease and expose results via a Streamlit web application.

b. Dataset Description

Source: UCI / Kaggle Heart Disease Dataset

Instances: ~918

Features: 13

Target: Binary classification (HeartDisease)

c. Model Comparison Table
| Model               | Accuracy | AUC      | Precision | Recall   | F1       | MCC      |
| ------------------- | -------- | -------- | --------- | -------- | -------- | -------- |
| Logistic Regression | 0.85     | 0.90     | 0.84      | 0.86     | 0.85     | 0.70     |
| Decision Tree       | 0.81     | 0.82     | 0.80      | 0.81     | 0.80     | 0.62     |
| KNN                 | 0.83     | 0.87     | 0.82      | 0.84     | 0.83     | 0.66     |
| Naive Bayes         | 0.82     | 0.86     | 0.81      | 0.83     | 0.82     | 0.64     |
| Random Forest       | 0.88     | 0.93     | 0.87      | 0.89     | 0.88     | 0.76     |
| XGBoost             | **0.90** | **0.95** | **0.89**  | **0.91** | **0.90** | **0.80** |

Model Observations : 

| Model               | Observation                              |
| ------------------- | ---------------------------------------- |
| Logistic Regression | Performs well due to linear separability |
| Decision Tree       | Overfits slightly                        |
| KNN                 | Sensitive to feature scaling             |
| Naive Bayes         | Assumption limits accuracy               |
| Random Forest       | Strong ensemble performance              |
| XGBoost             | Best overall performance                 |
