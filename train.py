
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA = Path("data/drug200.csv")
MODEL_DIR = Path("model")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# 1. Load dataset
df = pd.read_csv(DATA)
X = df.drop(columns=["Drug"])
y = df["Drug"]

# 2. Define features
categorical = ["Sex", "BP", "Cholesterol"]
numerical = ["Age", "Na_to_K"]

# 3. Preprocessing
preprocessor = ColumnTransformer(
    [
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numerical", "passthrough", numerical),
    ]
)

# 4. Create pipeline
pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        )),
    ]
)

# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# 6. Train
pipeline.fit(X_train, y_train)

# 7. Evaluate
predictions = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="weighted")

print(f"Accuracy: {accuracy:.4f}")
print(f"Weighted F1-score: {f1:.4f}")

# 8. Save joblib model
joblib.dump(pipeline, MODEL_DIR / "drug_model.joblib")

# 9. Save metrics
with open(RESULTS_DIR / "metrics.txt", "w", encoding="utf-8") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Weighted F1-score: {f1:.4f}\n\n")
    f.write(classification_report(y_test, predictions))

# 10. Save confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test, predictions, xticks_rotation="vertical"
)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "confusion_matrix.png")
plt.close()

# 11. Define ONNX inputs in dataset column order
initial_types = [
    ("Age", FloatTensorType([None, 1])),
    ("Sex", StringTensorType([None, 1])),
    ("BP", StringTensorType([None, 1])),
    ("Cholesterol", StringTensorType([None, 1])),
    ("Na_to_K", FloatTensorType([None, 1])),
]

# 12. Export ONNX
onnx_model = convert_sklearn(
    pipeline,
    initial_types=initial_types,
    target_opset=17,
    options={id(pipeline): {"zipmap": False}},
)

# 13. Save ONNX file
with open(MODEL_DIR / "drug_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("Joblib model saved successfully.")
print("ONNX model exported successfully.")
print("Training and evaluation completed.")
