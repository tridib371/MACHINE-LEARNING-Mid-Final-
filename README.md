# 🤖 Machine Learning – Mid & Final

This repository contains all materials for **Machine Learning** including **theory, assignments, and projects** implemented in **Python** using **Google Colab**.

---

## 📚 Course Overview

**Machine Learning (ML)** is a subset of Artificial Intelligence (AI) that enables systems to learn from data and improve from experience without being explicitly programmed.

### Key Learning Objectives:
- Understand **supervised, unsupervised, and reinforcement learning**.  
- Implement ML algorithms using **Python libraries**: `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`.  
- Evaluate models using **accuracy, precision, recall, F1-score, and confusion matrix**.  
- Apply ML concepts to **real-world projects and datasets**.  

---

## 📝 Theory Topics

### 1. **Introduction to Machine Learning**
- Definition, history, and applications  
- Types of ML: Supervised, Unsupervised, Reinforcement Learning  
- Steps in ML pipeline: Data collection → Preprocessing → Model training → Evaluation → Deployment  

### 2. **Data Preprocessing**
- Handling missing values  
- Encoding categorical variables (`LabelEncoder`, `OneHotEncoder`)  
- Feature scaling (`StandardScaler`, `MinMaxScaler`)  
- Train/Test split  

### 3. **Supervised Learning**
- Regression Algorithms: Linear Regression, Polynomial Regression  
- Classification Algorithms: Logistic Regression, K-Nearest Neighbors, Decision Trees, Random Forest, Support Vector Machine  

### 4. **Unsupervised Learning**
- Clustering: K-Means, Hierarchical Clustering  
- Dimensionality Reduction: PCA (Principal Component Analysis)  

### 5. **Model Evaluation**
- Metrics: Accuracy, Precision, Recall, F1-score  
- Confusion Matrix  
- Cross-validation  
- Overfitting vs Underfitting  

### 6. **Advanced Topics**
- Ensemble Methods: Bagging, Boosting  
- Neural Networks (Basic Overview)  
- Feature Importance and Selection  

---

## 🖥️ Lab / Practical Exercises (Python in Google Colab)

### 1. **Setup in Google Colab**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
