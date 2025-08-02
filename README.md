# Breast Cancer Tumour Stage Prediction

This project aims to predict **breast cancer tumour stages** using machine learning models trained on protein expression data and receptor status. Two classifiers are used: **Random Forest** (with hyperparameter tuning) and **XGBoost**. The model also supports prediction on new samples.



## 🔍 Project Overview

- **Input Features**:
  - Patient age
  - Protein expression levels (Protein1–Protein4)
  - Hormone receptor statuses (ER, PR, HER2)

- **Target Variable**:
  - Tumour Stage (`Stage I`, `Stage II`, `Stage III`)

- **Models Used**:
  - Random Forest (baseline and tuned using GridSearchCV)
  - XGBoost Classifier

- **Evaluation Metrics**:
  - Accuracy
  - Classification Report
  - Cross-validation score
  - Feature Importance Visualization



## 📊 Dataset

The dataset used (`jvtm241.csv`) is a **small-scale example dataset** with limited samples. Due to difficulty in accessing larger real-world cancer datasets, this project demonstrates **model pipeline design** more than performance accuracy.



## 👨🏻‍💻 Result

