import csv
import os
import time
from datetime import datetime
from typing import Dict, List

import gradio as gr
import gdown  # 🔥 ADDED

from ai_assistant import BreedAIAssistant
from breed_predictor import BreedPredictor


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "cattle_breed_classifier.pth")

# 🔥 ADDED: Auto-download model if not present
if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/file/d/1VMimaWINZCXnh-sbjhuT2UdsMvfYMFd3/view?usp=sharing"  # 🔴 REPLACE THIS
    gdown.download(url, MODEL_PATH, quiet=False)

DATA_DIR = os.path.join(SCRIPT_DIR, "IndianCattleBuffaloeBreeds-Dataset", "breeds", "train")
FEEDBACK_FILE = os.path.join(SCRIPT_DIR, "prediction_feedback.csv")


# -------------------- Helpers --------------------

def build_example_paths():
    paths = [
        os.path.join(SCRIPT_DIR, "IndianCattleBuffaloeBreeds-Dataset", "breeds", "test", "Gir", "Gir_001.jpg"),
        os.path.join(SCRIPT_DIR, "IndianCattleBuffaloeBreeds-Dataset", "breeds", "test", "Murrah", "Murrah_001.jpg"),
        os.path.join(SCRIPT_DIR, "IndianCattleBuffaloeBreeds-Dataset", "breeds", "test", "Sahiwal", "Sahiwal_001.jpg"),
    ]
    return [p for p in paths if os.path.exists(p)]


def format_prediction(predicted_breed, confidence):
    return f"### Prediction\n**Breed:** {predicted_breed}\n**Confidence:** {confidence:.2%}"


def format_explanation(report):
    return (
        f"### AI Explanation\n"
        f"**Description:** {report['description']}\n\n"
        f"**Purpose:** {report['purpose']}\n\n"
        f"**Care:** {report['care']}"
    )


# -------------------- Load --------------------

try:
    predictor = BreedPredictor(MODEL_PATH, DATA_DIR)
    assistant = BreedAIAssistant()
except:
    predictor = None
    assistant = BreedAIAssistant()


# -------------------- Logic --------------------

def predict(image):
    result = predictor.predict(image)
    report = assistant.generate_breed_report(result.predicted_breed, result.confidence)

    return (
        result.top_predictions,
        format_prediction(result.predicted_breed, result.confidence),
        format_explanation(report),
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
    gr.Markdown("Upload image → Get prediction → Ask AI")

    breed_state = gr.State("")
    conf_state = gr.State(0.0)

    # ----------- TOP SECTION -----------
    with gr.Row():

        # LEFT
        with gr.Column():
            image = gr.Image(type="pil", label="Upload Image")

            with gr.Row():
                clear_btn = gr.Button("Clear")
                predict_btn = gr.Button("Submit")

            gr.Examples(build_example_paths(), inputs=image)

        # RIGHT
        with gr.Column():
            pred_label = gr.Label(num_top_classes=5, label="Top 5 Predictions")
            pred_text = gr.Markdown()

    # ----------- BELOW -----------
    explanation = gr.Markdown()
    feedback = gr.Radio(["Correct", "Incorrect"], label="Feedback")
    feedback_status = gr.Textbox(label="Status")
    chatbot = gr.Chatbot(label="AI Assistant")

    # 🔥 UPDATED CHAT UI
    with gr.Row():
        chat_input = gr.Textbox(
            placeholder="Ask about breed...",
            show_label=False,
            scale=6
        )

        send_btn = gr.Button(
            "Send",
            scale=1,
            min_width=90
        )

        clear_chat_btn = gr.Button(
            "Clear",
            scale=1,
            min_width=90
        )

    

    # ----------- ACTIONS -----------

    predict_btn.click(
        predict,
        inputs=image,
        outputs=[
            pred_label,
            pred_text,
            explanation,
            breed_state,
            conf_state,
            chatbot,
        ],
    )

    # 🔥 ENTER KEY SUPPORT
    chat_input.submit(
        add_user,
        inputs=[chat_input, chatbot],
        outputs=[chat_input, chatbot],
    ).then(
        stream,
        inputs=[chatbot, breed_state],
        outputs=chatbot,
    )

    # BUTTON CLICK
    send_btn.click(
        add_user,
        inputs=[chat_input, chatbot],
        outputs=[chat_input, chatbot],
    ).then(
        stream,
        inputs=[chatbot, breed_state],
        outputs=chatbot,
    )

    clear_chat_btn.click(clear_chat, outputs=chatbot)

    # 🔥 CLEAR IMAGE FIX
    clear_btn.click(
        fn=clear_image,
        outputs=image
    )

    feedback.change(
        save_feedback,
        inputs=[feedback, breed_state, conf_state],
        outputs=feedback_status,
    )


# -------------------- RUN --------------------

if __name__ == "__main__":
    demo.launch()