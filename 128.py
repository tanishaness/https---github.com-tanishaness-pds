# 1. Data Collection
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'Age': [25, 30, np.nan, 45, 50],
    'Salary': [50000, 60000, 55000, 65000, 70000],
    'Purchased': [0, 1, 0, 1, 1]
})
print("Collected Data:\n", df)

# 2. Data Cleaning
df_cleaned = df.drop_duplicates()
df_cleaned = df_cleaned.dropna(subset=['Salary'])
print("After Cleaning:\n", df_cleaned)
# 3. Data Transformation
# 3. Data Integration
df_api = pd.DataFrame({'Age': [35], 'Salary': [58000], 'Purchased': [1]})
df_merged = pd.concat([df_cleaned, df_api], ignore_index=True)
print("After Integration:\n", df_merged)
# 4. Data Preprocessing
from sklearn.preprocessing import StandardScaler

df_encoded = pd.get_dummies(df_merged, columns=['Purchased'], drop_first=True)
scaler = StandardScaler()
df_encoded[['Age', 'Salary']] = scaler.fit_transform(df_encoded[['Age', 'Salary']])
print("After Preprocessing:\n", df_encoded)
# 5. Missing Value Imputation
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=2)
df_imputed = pd.DataFrame(imputer.fit_transform(df_encoded), columns=df_encoded.columns)
print("After Imputation:\n", df_imputed)
# 6. Feature Selection
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif

X_fs = df_imputed.drop('Purchased_1', axis=1)
y_fs = df_imputed['Purchased_1']

model_fs = ExtraTreesClassifier()
model_fs.fit(X_fs, y_fs)

selector = SelectKBest(score_func=f_classif, k='all')
X_selected = selector.fit_transform(X_fs, y_fs)
print("Selected Features Shape:", X_selected.shape)
# 7. Feature Engineering
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

df_fe = df_imputed.copy()
df_fe['Age_log'] = np.log1p(df_fe['Age'])

poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df_fe[['Age', 'Salary']])

df_poly = pd.concat([df_fe, pd.DataFrame(poly_features, columns=[f'poly_{i}' for i in range(poly_features.shape[1])])], axis=1)
print("After Feature Engineering:\n", df_poly.head())
# 8. Data Splitting
from sklearn.model_selection import train_test_split

X = df_imputed.drop('Purchased_1', axis=1)
y = df_imputed['Purchased_1']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
# 9. Model Training - Supervised
from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier()
model_rf.fit(X_train, y_train)
print("Training Accuracy:", model_rf.score(X_train, y_train))
# 10. Model Training - Unsupervised (KMeans)
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X_unsupervised = df_imputed.drop('Purchased_1', axis=1)

kmeans = KMeans(n_clusters=2, random_state=42)
df_imputed['Cluster'] = kmeans.fit_predict(X_unsupervised)

print("Cluster Assignments:\n", df_imputed[['Age', 'Salary', 'Cluster']])

# Plot clusters
plt.scatter(df_imputed['Age'], df_imputed['Salary'], c=df_imputed['Cluster'], cmap='viridis')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.title('KMeans Clustering')
plt.grid(True)
plt.show()
# 11. Model Validation with Grid Search
from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators': [10, 50, 100]}
grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=3)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
val_accuracy = best_model.score(X_test, y_test)
print("Best Estimator:", best_model)
print("Validation Accuracy:", val_accuracy)
# 12. Model Testing
test_accuracy = best_model.score(X_test, y_test)
print("Test Accuracy:", test_accuracy)
# 13. Cross Validation with KFold
from sklearn.model_selection import cross_val_score

kf_scores = cross_val_score(RandomForestClassifier(), X, y, cv=5)
print("KFold CV Scores:", kf_scores)
print("Mean CV Accuracy:", np.mean(kf_scores))
# 14. Regularization Demo
from sklearn.linear_model import LogisticRegression

logreg_model = LogisticRegression(penalty='l2', solver='liblinear')
reg_scores = cross_val_score(logreg_model, X, y, cv=5)
print("Regularization (L2) Accuracy:", np.mean(reg_scores))
# 15. Underfitting Solution
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
logreg = LogisticRegression(max_iter=500)
logreg.fit(X_poly, y)
train_acc = logreg.score(X_poly, y)

print("Polynomial Transformed Training Accuracy (Underfitting Fix):", train_acc)
# 16. Confusion Matrix Visualization
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.title("Confusion Matrix")
plt.grid(False)
plt.show()
# 17. Precision Score
from sklearn.metrics import precision_score

precision = precision_score(y_test, y_pred)
print("Precision Score:", precision)
# 18. Recall Score
from sklearn.metrics import recall_score

recall = recall_score(y_test, y_pred)
print("Recall Score:", recall)
# 19. Accuracy with TP, TN, FP, FN
tp = sum((y_test == 1) & (y_pred == 1))
tn = sum((y_test == 0) & (y_pred == 0))
fp = sum((y_test == 0) & (y_pred == 1))
fn = sum((y_test == 1) & (y_pred == 0))

accuracy = (tp + tn) / (tp + tn + fp + fn)
print(f"Accuracy: {accuracy:.2f}, TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
# 20. ROC Curve
from sklearn.metrics import roc_curve

y_scores = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_scores)

plt.plot(fpr, tpr, label='ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()
# 21. AUC Calculation
from sklearn.metrics import roc_auc_score

auc = roc_auc_score(y_test, y_scores)
print("AUC Score:", auc)
# 22. Cross Validation (KFold again, same as 13)
from sklearn.model_selection import KFold, cross_val_score

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(RandomForestClassifier(), X, y, cv=kf)
print("Cross-Validation Scores:", cv_scores)
# 23. Manual KFold Loop
kf = KFold(n_splits=5)
scores = []
for train_idx, test_idx in kf.split(X):
    X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
    y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
    model = RandomForestClassifier()
    model.fit(X_train_fold, y_train_fold)
    scores.append(model.score(X_test_fold, y_test_fold))

print("Manual KFold Scores:", scores)
# 24. Leave-One-Out Cross-Validation
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
model = RandomForestClassifier()
loo_scores = []

for train_idx, test_idx in loo.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    model.fit(X_train, y_train)
    loo_scores.append(model.score(X_test, y_test))

print("LOOCV Mean Accuracy:", np.mean(loo_scores))
# 25. Feature Engineering Scaling + Transformation
from sklearn.preprocessing import MinMaxScaler

# Log, square, sqrt transformations
df_fe = df_imputed.copy()
df_fe['Age_log'] = np.log1p(df_fe['Age'])
df_fe['Salary_sq'] = df_fe['Salary'] ** 2
df_fe['Salary_sqrt'] = np.sqrt(df_fe['Salary'])

# MinMax Scaling
scaler = MinMaxScaler()
df_fe[['Age_scaled', 'Salary_scaled']] = scaler.fit_transform(df_fe[['Age', 'Salary']])

print("Feature Engineered Data:\n", df_fe.head())
# 26. Overfitting Demo
model = RandomForestClassifier()
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Training Accuracy: {train_score}")
print(f"Test Accuracy: {test_score}")
print("Overfitting if training >> test accuracy")
# 27. Underfitting Demo
model = LogisticRegression()
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Training Accuracy: {train_score}")
print(f"Test Accuracy: {test_score}")
print("Underfitting if both are low")
# 28. Bias-Variance Tradeoff
degrees = [1, 2, 3, 4, 5]
train_scores, test_scores = [], []

for d in degrees:
    poly = PolynomialFeatures(degree=d)
    X_poly = poly.fit_transform(X)
    model = LogisticRegression(max_iter=500)
    model.fit(X_poly, y)
    train_scores.append(model.score(X_poly, y))
    test_scores.append(cross_val_score(model, X_poly, y, cv=3).mean())

plt.plot(degrees, train_scores, label='Train Accuracy')
plt.plot(degrees, test_scores, label='Validation Accuracy')
plt.xlabel('Model Complexity (Polynomial Degree)')
plt.ylabel('Accuracy')
plt.title('Bias-Variance Tradeoff')
plt.legend()
plt.grid(True)
plt.show()
