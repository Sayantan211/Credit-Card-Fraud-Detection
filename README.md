# Credit-Card-Fraud-Detection

This project aims to detect fraudulent credit card transactions from a **real-world, highly imbalanced dataset**. What sets this project apart is the implementation of **Logistic Regression from scratch using NumPy**, along with proper handling of class imbalance using **SMOTE**.

---

## 🔍 Problem Statement

> Detect fraudulent transactions in real-time from a credit card dataset where only **0.17%** of the entries are fraud. The goal is to ensure **high recall and precision** for fraud detection while maintaining balanced performance across all classes.

---

## 📦 Dataset Overview

- **Total Transactions:** 284,807 
- **Fraudulent Transactions:** ~0.17%  
- **Features:** 30 anonymized features + `Time`, `Amount`, and `Class`  
- **Target:** `Class` → 1 for Fraud, 0 for Legit

Dataset Source: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## 🔁 Full Workflow

### 1️⃣ Exploratory Data Analysis (EDA)
- Checked for missing values and data types  
- Visualized class distribution (heavily skewed toward non-fraud)  

### 2️⃣ Data Preprocessing
- **Feature Scaling:** Used `StandardScaler` for `Amount` and `Time`  
- **Train-Test Split:** 70-30 split on the scaled dataset  
- **Handling Imbalance:** Applied **SMOTE** to oversample minority class (fraud) in training set

### 3️⃣ Custom Logistic Regression (from Scratch)
- Implemented using **NumPy only**  
- Functions include: sigmoid activation, binary cross-entropy loss, gradient descent  
- Written as a modular Python class `MyLogReg`

### 4️⃣ Model Training
- Trained :
  - **Custom Logistic Regression**  

### 5️⃣ Evaluation Metrics
- Used `classification_report` and `confusion_matrix`
- Focused on **Accuracy, Precision, Recall, F1-Score**

---

## 📊 Final Results

## 📊 Final Evaluation Metrics

| Class           | Precision | Recall | F1-Score | Support |
|-----------------|-----------|--------|----------|---------|
| Non-Fraud (0)   | 0.91      | 0.97   | 0.94     | 55,073  |
| Fraud (1)       | 0.97      | 0.91   | 0.94     | 55,003  |
| **Accuracy**    |           |        | **0.94** | 110,076 |
| **Macro Avg**   | 0.94      | 0.94   | 0.94     | 110,076 |
| **Weighted Avg**| 0.94      | 0.94   | 0.94     | 110,076 |


✅ These metrics were **balanced across both fraud and non-fraud classes**, thanks to SMOTE oversampling.

---

## 🧠 Tech Stack

- Python 3.10  
- NumPy (custom ML logic)  
- pandas, scikit-learn  
- imbalanced-learn (SMOTE)  
- matplotlib, seaborn (EDA)

---

## 🧮 Custom Logistic Regression Class (NumPy)

```python
class MyLogReg:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        for _ in range(self.epochs):
            linear = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear)
            dw = (1 / len(y)) * np.dot(X.T, (y_pred - y))
            db = (1 / len(y)) * np.sum(y_pred - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear)
        return np.where(y_pred >= 0.5, 1, 0)
