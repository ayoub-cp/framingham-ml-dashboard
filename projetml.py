# ============================================================
# FRAMINGHAM ML PROJECT — FULL LOCAL VERSION (VS CODE)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge
from sklearn.decomposition import PCA
from sklearn import svm

from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    mean_squared_error, precision_score,
    recall_score, f1_score
)

# ============================================================
# LOAD DATA
# ============================================================

DATA_PATH = "framingham.csv"
df = pd.read_csv(DATA_PATH)

print("=== DATA OVERVIEW ===")
print(df.head())
print(df.info())

# ============================================================
# EDA
# ============================================================

df_numeric = df.select_dtypes(include=['float64', 'int64'])

df_numeric.hist(bins=30, figsize=(14, 10))
plt.suptitle('Distribution des variables')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# ============================================================
# PREPROCESSING
# ============================================================

df = df.drop_duplicates()

cols_missing = ['education', 'cigsPerDay', 'BPMeds',
                'totChol', 'BMI', 'heartRate', 'glucose']

imputer = SimpleImputer(strategy='median')
df[cols_missing] = imputer.fit_transform(df[cols_missing])

cols_scale = ['age', 'cigsPerDay', 'totChol',
              'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']

scaler = MinMaxScaler()
df[cols_scale] = scaler.fit_transform(df[cols_scale])

# ============================================================
# FEATURE SELECTION
# ============================================================

X = df.drop('TenYearCHD', axis=1)
y = df['TenYearCHD']

selector = SelectKBest(k=5)
X_new = selector.fit_transform(X, y)

print("\n=== FEATURE SCORES ===")
for feature, score in zip(X.columns, selector.scores_):
    print(f"{feature}: {score:.2f}")

# ============================================================
# OUTLIER REMOVAL
# ============================================================

cols_outliers = ['totChol', 'BMI', 'glucose']

for col in cols_outliers:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df = df[(df[col] >= lower) & (df[col] <= upper)]

print("\nDataset after outlier removal:", df.shape)

# ============================================================
# CLASSIFICATION (KNN + DECISION TREE)
# ============================================================

X = df[['age','sysBP','diaBP','totChol','BMI','heartRate','glucose']]
y = df['TenYearCHD']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

scaler_std = StandardScaler()
X_train = scaler_std.fit_transform(X_train)
X_test = scaler_std.transform(X_test)

print("\n=== CLASSIFICATION RESULTS ===")

for k in [1, 3, 5]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    print(f"KNN (k={k}) Accuracy:", accuracy_score(y_test, pred))

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
pred_dt = dt.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, pred_dt))

# ============================================================
# LOGISTIC REGRESSION
# ============================================================

X = df[['age','sysBP','totChol','BMI','glucose']]
y = df['TenYearCHD']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

log_model = LogisticRegression(max_iter=1000, class_weight='balanced')
log_model.fit(X_train, y_train)

pred = log_model.predict(X_test)

print("\n=== LOGISTIC REGRESSION ===")
print("Accuracy:", accuracy_score(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))

# ============================================================
# LINEAR REGRESSION
# ============================================================

X = df[['sysBP']]
y = df['glucose']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

lin = LinearRegression()
lin.fit(X_train, y_train)

pred = lin.predict(X_test)

print("\n=== LINEAR REGRESSION ===")
print("MSE:", mean_squared_error(y_test, pred))

# ============================================================
# LASSO & RIDGE
# ============================================================

lasso = Lasso(alpha=1.0)
ridge = Ridge(alpha=1.0)

lasso.fit(X_train, y_train)
ridge.fit(X_train, y_train)

pred_lasso = lasso.predict(X_test)
pred_ridge = ridge.predict(X_test)

print("\n=== LASSO vs RIDGE ===")
print("Lasso MSE:", mean_squared_error(y_test, pred_lasso))
print("Ridge MSE:", mean_squared_error(y_test, pred_ridge))

# ============================================================
# SVM + PCA
# ============================================================

X = df.drop('TenYearCHD', axis=1)
y = df['TenYearCHD']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.3, random_state=42)

model = svm.SVC(kernel='linear', class_weight='balanced')
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("\n=== SVM + PCA ===")
print("Accuracy:", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred))
print("Recall:", recall_score(y_test, pred))
print("F1 Score:", f1_score(y_test, pred))