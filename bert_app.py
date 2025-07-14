from transformers import pipeline
from pathlib import Path
import gradio as gr

# Resolve to absolute path (prevents Hugging Face repo name validation)
model_path = Path("./toxic-bert").resolve()

# Load the model and tokenizer from local disk only
classifier = pipeline(
    "text-classification",
    model=str(model_path),
    tokenizer=str(model_path),
    local_files_only=True
)

# Classify function
def classify(text):
    result = classifier(text)[0]
    label = "Toxic 🚨" if result['label'] == "LABEL_1" else "Not Toxic ✅"
    return f"{label} ({result['score']:.2%})"

# Launch Gradio app
gr.Interface(
    fn=classify,
    inputs=gr.Textbox(lines=2, placeholder="Enter a comment..."),
    outputs="text",
    title="🧠 Toxic Comment Detector (BERT)",
    description="Predicts whether a comment is toxic using a fine-tuned DistilBERT model."
).launch()
