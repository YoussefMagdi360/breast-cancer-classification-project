# Breast Cancer Tumor Classification

## Project Objective
This graduation project builds a machine learning classification model that predicts whether a breast tumor is **benign** or **malignant** using the Breast Cancer Wisconsin dataset available in scikit-learn.

The goal is to follow the full machine learning workflow:

1. Select a suitable dataset
2. Perform Exploratory Data Analysis (EDA)
3. Clean and preprocess the data
4. Train a machine learning model
5. Evaluate the model using classification metrics
6. Deploy the trained model with Streamlit as a bonus

---

## Dataset
The project uses the built-in `load_breast_cancer` dataset from scikit-learn.

Dataset summary:

- Task type: Classification
- Number of samples: 569
- Number of features: 30
- Target classes:
  - Benign
  - Malignant

Because the dataset is built into scikit-learn, no external CSV file is required.

---

## Model Used
The main model used is **Logistic Regression** inside a scikit-learn Pipeline.

The pipeline includes:

1. `StandardScaler` for feature scaling
2. `LogisticRegression` for classification

Logistic Regression is suitable here because this is a binary classification problem and the model is easy to explain.

---

## Evaluation Metrics
The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- Classification Report

---

## Project Structure

```text
breast_cancer_ml_project/
│
├── notebooks/
│   └── breast_cancer_classification.ipynb
│
├── src/
│   └── train_model.py
│
├── models/
│   └── model.joblib  # created after training
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## How to Run the Project

### 1. Install the required libraries

```bash
pip install -r requirements.txt
```

### 2. Open and run the notebook

```bash
jupyter notebook notebooks/breast_cancer_classification.ipynb
```

### 3. Train and save the model

```bash
python src/train_model.py
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

---

## Important Note
This project is for educational purposes only. It should not be used as a real medical diagnosis tool.
