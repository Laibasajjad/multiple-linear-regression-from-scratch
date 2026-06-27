# 📊 Multiple Linear Regression from Scratch

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![NumPy](https://img.shields.io/badge/NumPy-ML-orange)
![Scikit-learn](https://img.shields.io/badge/Sklearn-Comparison-green)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)

---

## 🧩 Problem Statement

Real-world housing prices are influenced by multiple factors such as:

- 📏 House size  
- 🛏 Number of bedrooms  
- 🏠 Age of the house  

The objective of this project is to **predict house prices using Multiple Linear Regression**, while also deeply understanding how the algorithm works internally.

> 🔥 Instead of using built-in libraries directly, we implement the entire learning process from scratch using NumPy.

We also compare our results with **Scikit-learn’s Linear Regression** to validate correctness.

---

## 🎯 Objectives

- Build Multiple Linear Regression from scratch
- Understand Gradient Descent optimization
- Apply Feature Scaling for faster convergence
- Visualize cost reduction and predictions
- Compare with Scikit-learn implementation

---

## 🧠 Approach (ML Pipeline)

### 1️⃣ Data Preparation
- Define dataset (Size, Bedrooms, Age → Price)
- Convert into NumPy arrays

### 2️⃣ Feature Scaling
- Standardization using mean and standard deviation
- Improves gradient descent performance

### 3️⃣ Model Building (From Scratch)
We implement:

- Hypothesis function  
- Cost function (MSE)  
- Gradient computation  
- Gradient descent optimization  

### 4️⃣ Training
- 3000 iterations (epochs)
- Learning rate tuning

### 5️⃣ Prediction
- Predict unseen house prices

### 6️⃣ Visualization
- Cost function convergence
- Regression fit
- Feature relationships

### 7️⃣ Benchmarking
- Compare with Scikit-learn model

---

## 📊 Dataset

| Feature  | Description          |
|----------|----------------------|
| Size     | House area (sq ft)   |
| Bedrooms | Number of rooms      |
| Age      | Age of house (years) |
| Price    | Target variable      |

---

## 🧮 Mathematical Foundation

### Hypothesis Function
\[
h(x) = w^T x + b
\]

### Cost Function
\[
J(w,b) = \frac{1}{2m} \sum (h(x) - y)^2
\]

### Gradient Descent Updates
\[
w := w - \alpha \frac{\partial J}{\partial w}
\]

\[
b := b - \alpha \frac{\partial J}{\partial b}
\]

---

## 📈 Results & Visualizations

### 🔹 Cost Function Convergence
Model learns by minimizing error over time:

![Cost Curve](images/cost_curve.png)

---

### 🔹 Regression Fit (Size vs Price)
Model predictions closely match real values:

![Regression Fit](images/regression_fit.png)

---

### 🔹 Feature Analysis

**Size vs Price**
![Size vs Price](images/size_vs_price.png)

**Bedrooms vs Price**
![Bedrooms vs Price](images/bedrooms_vs_price.png)

**Age vs Price**
![Age vs Price](images/age_vs_price.png)

---

## 🤖 Scikit-learn Benchmark

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x, y)

model.predict([[1600, 3, 6]])


✔ Results are consistent with our scratch implementation

---

## 📊 Key Insights

* Feature scaling is critical for stable convergence
* Cost function decreases smoothly → correct learning behavior
* Scratch implementation builds deep intuition
* Scikit-learn abstracts internal optimization steps

---

## 🛠 Tech Stack

* Python 🐍
* NumPy
* Matplotlib
* Scikit-learn

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

```bash
python src/linear_regression.py
```

---

## 📂 Project Structure

```
src/
 └── linear_regression.py   # From-scratch ML model

notebooks/
 └── exploration.ipynb      # Experiments & visualizations

images/
 ├── cost_curve.png
 ├── regression_fit.png
 ├── size_vs_price.png
 ├── bedrooms_vs_price.png
 └── age_vs_price.png
```

---

## 👩‍💻 Author

**Laiba Sajjad**
BS Artificial Intelligence Student
Machine Learning Enthusiast 🤖

---

## ⭐ Final Note

This project demonstrates how machine learning works **from first principles**, bridging the gap between theory and practical implementation.

If you found this useful, consider starring ⭐ the repository.

```
