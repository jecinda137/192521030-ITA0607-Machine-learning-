# Install (Run this only once in Google Colab)
# !pip install pandas scikit-learn

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# CREATE DATASET
# ==========================================

X, y = make_classification(
    n_samples=2000,
    n_features=6,
    n_informative=5,
    n_redundant=1,
    n_classes=2,
    class_sep=2,
    random_state=42
)

columns = [
    "Length",
    "Width",
    "Weight",
    "Thickness",
    "SurfaceScore",
    "ColorVariation"
]

df = pd.DataFrame(X, columns=columns)
df["Quality"] = y

print("First 5 Rows")
print(df.head())

# ==========================================
# INPUT & OUTPUT
# ==========================================

X = df.drop("Quality", axis=1)
y = df["Quality"]

# ==========================================
# SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================
# PERCEPTRON MODEL
# ==========================================

print("\n===== PERCEPTRON MODEL =====")

perceptron = Perceptron(random_state=42)
perceptron.fit(X_train, y_train)

pred1 = perceptron.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, pred1) * 100, 2), "%")
print(classification_report(y_test, pred1))
print(confusion_matrix(y_test, pred1))

# ==========================================
# MLP MODEL
# ==========================================

print("\n===== MLP MODEL =====")

mlp = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    activation="relu",
    max_iter=1000,
    random_state=42
)

mlp.fit(X_train, y_train)

pred2 = mlp.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, pred2) * 100, 2), "%")
print(classification_report(y_test, pred2))
print(confusion_matrix(y_test, pred2))

# ==========================================
# MODEL COMPARISON
# ==========================================

print("\n===== MODEL COMPARISON =====")

print("Perceptron Accuracy :", round(accuracy_score(y_test, pred1) * 100, 2), "%")
print("MLP Accuracy        :", round(accuracy_score(y_test, pred2) * 100, 2), "%")

# ==========================================
# SAMPLE PREDICTION
# ==========================================

print("\n===== PRODUCT PREDICTION =====")

sample = pd.DataFrame(
    [[2.5, 1.8, 3.2, 2.1, 0.8, 1.5]],
    columns=columns
)

sample_scaled = scaler.transform(sample)

prediction = mlp.predict(sample_scaled)

if prediction[0] == 1:
    print("Prediction: GOOD PRODUCT")
else:
    print("Prediction: DEFECTIVE PRODUCT")
