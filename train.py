
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("data/drug200.csv")

# Separate features and target
X = data.drop("Drug", axis=1)
y = data["Drug"]

# Encode categorical columns
X = pd.get_dummies(X)
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Save model and encoders
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/drug_model.joblib")
joblib.dump(encoder, "model/label_encoder.joblib")
joblib.dump(X.columns.tolist(), "model/feature_columns.joblib")

print("\nModel and preprocessing files saved successfully.")
