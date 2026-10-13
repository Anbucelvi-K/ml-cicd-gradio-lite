
import gradio as gr


def predict(age, na_to_k):
    if age < 0 or na_to_k < 0:
        return "Enter non-negative values."

    return (
        "Demo interface working. "
        "Connect the trained model for real predictions."
    )


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Age"),
        gr.Number(label="Na_to_K"),
    ],
    outputs=gr.Textbox(label="Status"),
    title="Drug Classification - CI/CD Demo",
    description="Testing the Gradio interface.",
)

if __name__ == "__main__":
    demo.launch()
