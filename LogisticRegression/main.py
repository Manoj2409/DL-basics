# Import necessary libraries
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Diabetes dataset
data = load_diabetes()
X = data.data  # Features
y = (data.target > data.target.mean()).astype(int)  # Binarize target for classification (0: below mean, 1: above mean)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Logistic Regression model
model = LogisticRegression(max_iter=10000, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix Visualization
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Below Mean", "Above Mean"], yticklabels=["Below Mean", "Above Mean"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Feature importance
coefficients = model.coef_[0]
feature_importance = sorted(zip(data.feature_names, coefficients), key=lambda x: abs(x[1]), reverse=True)
print("\nTop Features Contributing to the Model:")
for feature, coef in feature_importance:
    print(f"{feature}: {coef:.4f}")
