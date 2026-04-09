import csv
import os
import time
from datetime import datetime

import gradio as gr
import gdown

from ai_assistant import BreedAIAssistant
from breed_predictor import BreedPredictor


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "cattle_breed_classifier.pth")

# 🔥 Download model if not present
if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1VMimaWINZCXnh-sbjhuT2UdsMvfYMFd3"
    gdown.download(url, MODEL_PATH, quiet=False)

FEEDBACK_FILE = os.path.join(SCRIPT_DIR, "prediction_feedback.csv")


# -------------------- Load --------------------
try:
    predictor = BreedPredictor(MODEL_PATH, None)
    assistant = BreedAIAssistant()
except Exception as e:
    print("Error loading model:", e)
    predictor = None
    assistant = BreedAIAssistant()


# -------------------- Logic --------------------

def predict(image):
    result = predictor.predict(image)
    report = assistant.generate_breed_report(result.predicted_breed, result.confidence)

    return (
        result.top_predictions,
        f"### Prediction\n**Breed:** {result.predicted_breed}\n**Confidence:** {result.confidence:.2%}",
        f"### AI Explanation\n**Description:** {report['description']}\n\n**Purpose:** {report['purpose']}\n\n**Care:** {report['care']}",
        result.predicted_breed,
        result.confidence,
        [{"role": "assistant", "content": f"Ask about {result.predicted_breed}"}],
    )


def add_user(msg, hist):
    hist = hist or []
    return "", hist + [{"role": "user", "content": msg}]


def stream(hist, breed):
    hist.append({"role": "assistant", "content": ""})
    reply = assistant.chat(hist[-2]["content"], breed, hist)

    for i in range(len(reply)):
        hist[-1]["content"] = reply[: i + 1]
        yield hist
        time.sleep(0.01)


def clear_chat():
    return []


def clear_image():
    return None


def save_feedback(fb, breed, conf):
    if not fb or not breed:
        return "Give prediction first"

    with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now(), breed, conf, fb])

    return "Saved"


# -------------------- UI --------------------

with gr.Blocks(title="Cattle Classifier") as demo:

    gr.Markdown("# 🐄 Indian Cattle & Buffalo Breed Classifier")

    breed_state = gr.State("")
    conf_state = gr.State(0.0)

    with gr.Row():
        with gr.Column():
            image = gr.Image(type="pil")
            predict_btn = gr.Button("Predict")

        with gr.Column():
            pred_label = gr.Label()
            pred_text = gr.Markdown()

    explanation = gr.Markdown()
    chatbot = gr.Chatbot()

    chat_input = gr.Textbox(placeholder="Ask about breed...")
    send_btn = gr.Button("Send")

    predict_btn.click(
        predict,
        inputs=image,
        outputs=[pred_label, pred_text, explanation, breed_state, conf_state, chatbot],
    )

    send_btn.click(add_user, inputs=[chat_input, chatbot], outputs=[chat_input, chatbot]).then(
        stream, inputs=[chatbot, breed_state], outputs=chatbot
    )


if __name__ == "__main__":
    demo.launch()