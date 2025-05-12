import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, LeaveOneOut, cross_val_score
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, 
    recall_score, roc_curve, roc_auc_score
)
import matplotlib.pyplot as plt

class DataPipeline:
    def __init__(self):
        self.data = None

    def collect_data(self):
        self.data = pd.DataFrame({
            'Age': [25, 30, np.nan, 45, 50],
            'Salary': [50000, 60000, 55000, 65000, 70000],
            'Purchased': [0, 1, 0, 1, 1]
        })
        return self.data

    def clean_data(self):
        self.data.drop_duplicates(inplace=True)
        self.data.dropna(subset=['Salary'], inplace=True)
        return self.data

    def integrate_data(self):
        df_api = pd.DataFrame({'Age': [35], 'Salary': [58000], 'Purchased': [1]})
        self.data = pd.concat([self.data, df_api], ignore_index=True)
        return self.data

    def preprocess_data(self):
        self.data = pd.get_dummies(self.data, columns=['Purchased'], drop_first=True)
        scaler = StandardScaler()
        self.data[['Age', 'Salary']] = scaler.fit_transform(self.data[['Age', 'Salary']])
        return self.data

    def handle_missing_values(self):
        imputer = KNNImputer(n_neighbors=2)
        self.data = pd.DataFrame(imputer.fit_transform(self.data), columns=self.data.columns)
        return self.data

    def feature_selection(self):
        X = self.data.drop('Purchased_1', axis=1)
        y = self.data['Purchased_1']
        model = ExtraTreesClassifier()
        model.fit(X, y)
        selector = SelectKBest(score_func=f_classif, k='all')
        X_new = selector.fit_transform(X, y)
        return X_new

    def feature_engineering(self):
        self.data['Age_log'] = np.log1p(self.data['Age'])
        poly = PolynomialFeatures(degree=2, include_bias=False)
        poly_features = poly.fit_transform(self.data[['Age', 'Salary']])
        self.data = pd.concat([self.data, pd.DataFrame(poly_features)], axis=1)
        return self.data

    def split_data(self):
        X = self.data.drop('Purchased_1', axis=1)
        y = self.data['Purchased_1']
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_supervised_model(self, X_train, y_train):
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        return model

    def train_unsupervised_model(self):
        pass  # Skipped for brevity

    def validate_model(self, X_train, y_train, X_val, y_val):
        param_grid = {'n_estimators': [10, 50, 100]}
        grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=3)
        grid.fit(X_train, y_train)
        preds = grid.best_estimator_.predict(X_val)
        return grid.best_estimator_, accuracy_score(y_val, preds)

    def test_model(self, model, X_test, y_test):
        return accuracy_score(y_test, model.predict(X_test))

    def cross_validate_model(self, X, y):
        model = RandomForestClassifier()
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        return cross_val_score(model, X, y, cv=kf)

    def regularization_demo(self, X, y):
        model = LogisticRegression(penalty='l2', solver='liblinear')
        return cross_val_score(model, X, y, cv=5).mean()

    def underfitting_solution(self, X, y):
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        model = LogisticRegression()
        model.fit(X_poly, y)
        return model.score(X_poly, y)

    def confusion_matrix_demo(self, model, X_test, y_test):
        cm = confusion_matrix(y_test, model.predict(X_test))
        ConfusionMatrixDisplay(confusion_matrix=cm).plot()
        plt.show()
        return cm

    def precision_evaluation(self, y_true, y_pred):
        return precision_score(y_true, y_pred)

    def recall_evaluation(self, y_true, y_pred):
        return recall_score(y_true, y_pred)

    def classification_accuracy(self, y_true, y_pred):
        tp = sum((yt == 1 and yp == 1) for yt, yp in zip(y_true, y_pred))
        tn = sum((yt == 0 and yp == 0) for yt, yp in zip(y_true, y_pred))
        fp = sum((yt == 0 and yp == 1) for yt, yp in zip(y_true, y_pred))
        fn = sum((yt == 1 and yp == 0) for yt, yp in zip(y_true, y_pred))
        acc = (tp + tn) / (tp + tn + fp + fn)
        return acc, tp, tn, fp, fn

    def roc_curve_plot(self, y_true, y_scores):
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        plt.plot(fpr, tpr, label="ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend()
        plt.grid(True)
        plt.show()

    def calculate_auc(self, y_true, y_scores):
        return roc_auc_score(y_true, y_scores)

    def kfold_cross_validation(self, X, y):
        return self.cross_validate_model(X, y)

    def manual_kfold_loop(self, X, y):
        kf = KFold(n_splits=5)
        scores = []
        model = RandomForestClassifier()
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            model.fit(X_train, y_train)
            scores.append(model.score(X_test, y_test))
        return scores

    def leave_one_out_cv(self, X, y):
        loo = LeaveOneOut()
        model = RandomForestClassifier()
        scores = []
        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            model.fit(X_train, y_train)
            scores.append(model.score(X_test, y_test))
        return scores

    def feature_engineering_extended(self):
        self.data['Age_log'] = np.log1p(self.data['Age'])
        self.data['Salary_squared'] = self.data['Salary'] ** 2
        self.data['Salary_sqrt'] = np.sqrt(self.data['Salary'])
        return self.data

    def overfitting_demo(self, X_train, X_test, y_train, y_test):
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        return model.score(X_train, y_train), model.score(X_test, y_test)

    def underfitting_demo(self, X_train, X_test, y_train, y_test):
        model = LogisticRegression()
        model.fit(X_train, y_train)
        return model.score(X_train, y_train), model.score(X_test, y_test)

    def bias_variance_tradeoff_plot(self, X, y):
        degrees = [1, 2, 3, 4, 5]
        train_scores, test_scores = [], []
        for d in degrees:
            poly = PolynomialFeatures(degree=d)
            X_poly = poly.fit_transform(X)
            model = LogisticRegression(max_iter=500)
            model.fit(X_poly, y)
            train_scores.append(model.score(X_poly, y))
            test_scores.append(cross_val_score(model, X_poly, y, cv=3).mean())
        plt.plot(degrees, train_scores, label='Train')
        plt.plot(degrees, test_scores, label='Validation')
        plt.xlabel("Degree")
        plt.ylabel("Accuracy")
        plt.title("Bias-Variance Tradeoff")
        plt.legend()
        plt.grid(True)
        plt.show()
