import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv("data.csv")
print("First 5 rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
X = df.drop("target", axis=1)
y = df["target"]
from sklearn.model_selection import train_test_split, cross_val_score
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LogisticRegression(
    max_iter=5000,
    class_weight='balanced'  
)
from sklearn.linear_model import LogisticRegression
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nModel Performance")
print("---------------------------")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
from sklearn.metrics import classification_report, confusion_matrix, recall_score, accuracy_score
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
import matplotlib.pyplot as plt
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
cv_scores = cross_val_score(
    model,
    scaler.fit_transform(X),
    y,
    cv=5,
    scoring="recall"
)
print("\nCross Validation Recall Scores:", cv_scores)
print("Average CV Recall:", cv_scores.mean())