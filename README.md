# Credit Risk Analysis 

## Project Overview

This project focuses on **predicting the likelihood of credit default** using supervised machine learning techniques.  
It demonstrates a complete end-to-end data analysis and modeling workflow — from preprocessing and balancing an imbalanced dataset to model building, evaluation, and feature importance analysis.

The goal is to help financial institutions identify **high-risk borrowers** and make informed lending decisions through data-driven insights.

__________________________________________________________________________________________________________________


## Key Objectives

- Perform **data preprocessing**, cleaning, and imputation using `KNNImputer`
- Handle **imbalanced data** using the **SMOTE** technique
- Train and compare multiple ML models:
  - Logistic Regression  
  - Decision Tree Classifier  
  - Random Forest Classifier  
  - K-Nearest Neighbors (KNN)
- Evaluate model performance with precision, recall, F1-score, ROC-AUC
- Identify the most important features influencing credit risk

__________________________________________________________________________________________________________________


## Workflow Summary

### **Data Preprocessing**
- Loaded and inspected the dataset
- Handled missing values using **KNN Imputer**
- Applied label encoding for categorical features
- Split data into train and test sets (80:20)
- Balanced classes using **SMOTE** (Synthetic Minority Oversampling Technique)

### **Exploratory Data Analysis (EDA)**
- Distribution plots and boxplots to understand variable spread
- Correlation heatmap to visualize relationships
- Identified key risk drivers — e.g., interest rate, income, credit utilization

### **Model Training and Tuning**
Trained four ML models with optimized hyperparameters:
- **Logistic Regression:** baseline interpretable classifier  
- **Decision Tree:** simple non-linear classifier  
- **Random Forest:** ensemble model to improve generalization  
- **KNN:** distance-based model for local pattern detection  

### **Evaluation Metrics**
Used multiple metrics to measure model effectiveness:
- **Accuracy**
- **Precision / Recall / F1-Score**
- **Confusion Matrix**
- **ROC Curve and AUC**

### **Feature Importance Analysis**
- Extracted and visualized **feature importances** from the Random Forest model  
- Identified the most influential predictors of credit default  

__________________________________________________________________________________________________________________


## Results and Insights

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|-----------|------------|---------|-----------|----------|
| Logistic Regression | ~0.82 | 0.81 | 0.77 | 0.79 | 0.86 |
| Decision Tree | ~0.85 | 0.83 | 0.82 | 0.83 | 0.88 |
| **Random Forest** | **~0.90** | **0.89** | **0.87** | **0.88** | **0.93** |
| KNN | ~0.84 | 0.80 | 0.83 | 0.81 | 0.87 |

- **Random Forest** delivered the best performance — offering strong generalization and interpretability through feature importance.
- Key predictors included **income**, **loan amount**, **interest rate**, and **credit utilization ratio**.

__________________________________________________________________________________________________________________


## Future Improvements

- Incorporate gradient boosting algorithms (XGBoost, LightGBM, CatBoost).
- Deploy the model using Flask or Streamlit for live scoring.
- Apply hyperparameter tuning (GridSearchCV) for further optimization
- Integrate SHAP or LIME for advanced interpretability
- Test model robustness on new unseen data from other financial sources
