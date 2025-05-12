import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Step 1: Generate synthetic binary classification data
np.random.seed(42)
X = np.linspace(1, 10, 30).reshape(-1, 1)
y = (X.flatten() > 5).astype(int)  # Labels: 0 if <= 5, else 1

# Step 2: Bias-Variance Tradeoff
degrees = [1, 2, 3, 4, 5]
train_scores = []
val_scores = []

for d in degrees:
    poly = PolynomialFeatures(degree=d)
    X_poly = poly.fit_transform(X)
    model = LogisticRegression(max_iter=500)
    model.fit(X_poly, y)
    train_scores.append(model.score(X_poly, y))
    val_scores.append(cross_val_score(model, X_poly, y, cv=3).mean())

# Step 3: Plotting
plt.figure(figsize=(8, 5))
plt.plot(degrees, train_scores, label='Training Accuracy', marker='o')
plt.plot(degrees, val_scores, label='Validation Accuracy', marker='s')
plt.xlabel("Polynomial Degree")
plt.ylabel("Accuracy")
plt.title("Bias-Variance Tradeoff")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
