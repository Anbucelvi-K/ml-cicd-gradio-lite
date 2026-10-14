import gradio as gr
import pandas as pd
import joblib

model = joblib.load("model/drug_model.joblib")
encoder = joblib.load("model/label_encoder.joblib")
feature_columns = joblib.load("model/feature_columns.joblib")


def predict(age, sex, bp, cholesterol, na_to_k):
    if age < 0 or na_to_k < 0:
        return "Enter non-negative values."

    data = pd.DataFrame(
        [[age, sex, bp, cholesterol, na_to_k]],
        columns=["Age", "Sex", "BP", "Cholesterol", "Na_to_K"],
    )

    data = pd.get_dummies(data)
    data = data.reindex(columns=feature_columns, fill_value=0)

    prediction = model.predict(data)
    drug = encoder.inverse_transform(prediction)[0]

    return f"Predicted Drug Class: {drug}"


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Age", value=30),
        gr.Dropdown(["F", "M"], label="Sex"),
        gr.Dropdown(["LOW", "NORMAL", "HIGH"], label="Blood Pressure"),
        gr.Dropdown(["NORMAL", "HIGH"], label="Cholesterol"),
        gr.Number(label="Na_to_K", value=15),
    ],
    outputs=gr.Textbox(label="Predicted Drug"),
    title="Drug Classification",
    description="Predict a drug class using the trained Random Forest model.",
)

if __name__ == "__main__":
    demo.launch()
