import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
import os
from cattle_model import CattleBreedClassifier

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "cattle_breed_classifier.pth")
DATA_DIR = os.path.join(SCRIPT_DIR, "IndianCattleBuffaloeBreeds-Dataset", "breeds", "train")

IMG_SIZE = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Model and Class Names ---
try:
    class_names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    num_classes = len(class_names)
    
    model = CattleBreedClassifier(num_classes=num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("PyTorch model loaded successfully.")

except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- Image Transformation ---
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# --- Prediction Function ---
def predict(inp_image):
    """
    Takes a PIL image, preprocesses it, and returns a dictionary of
    breed names to probabilities.
    """
    if model is None:
        return {"Error": "Model not loaded. Please check the console."}
        
    # Convert Gradio image (numpy) to PIL Image
    if not isinstance(inp_image, Image.Image):
        inp_image = Image.fromarray(inp_image)

    img = inp_image.convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    
    # Create a dictionary of {class_name: probability}
    confidences = {class_names[i]: float(probabilities[i]) for i in range(num_classes)}
    
    return confidences

# --- Gradio Interface ---
import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
import os
import requests
import sys
from cattle_model import CattleBreedClassifier

# --- Configuration ---
# IMPORTANT: You must upload your 'cattle_breed_classifier.pth' file to a public file hosting service
# and replace the URL below with your direct download link.
MODEL_URL = "YOUR_MODEL_URL_HERE"  # <--- REPLACE WITH YOUR MODEL'S DIRECT DOWNLOAD URL
MODEL_PATH = "cattle_breed_classifier.pth"
IMG_SIZE = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Class Names ---
# This list is derived from the training dataset directory structure.
# If you retrain your model with different breeds, you must update this list.
class_names = [
    'Alambadi', 'Amritmahal', 'Ayrshire', 'Banni', 'Bargur', 'Bhadawari', 
    'Brown_Swiss', 'Dangi', 'Deoni', 'Gir', 'Guernsey', 'Hallikar', 'Hariana', 
    'Holstein_Friesian', 'Jaffrabadi', 'Jersey', 'Kangayam', 'Kankrej', 
    'Kasargod', 'Kenkatha', 'Kherigarh', 'Khillari', 'Krishna_Valley', 
    'Malnad_gidda', 'Mehsana', 'Murrah', 'Nagori', 'Nagpuri', 'Nili_Ravi', 
    'Nimari', 'Ongole', 'Pulikulam', 'Rathi', 'Red_Dane', 'Red_Sindhi', 
    'Sahiwal', 'Surti', 'Tharparkar', 'Toda', 'Umblachery', 'Vechur'
]
num_classes = len(class_names)

# --- Model Loading ---
def download_file(url, dest):
    """Downloads a file from a URL to a destination, showing progress."""
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(dest, 'wb') as f:
                if total_size == 0:
                    f.write(r.content)
                else:
                    chunk_size = 8192
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
        print(f"Successfully downloaded {dest}")
        return dest
    except requests.exceptions.RequestException as e:
        print(f"Error downloading model: {e}", file=sys.stderr)
        return None

model = None
if MODEL_URL == "YOUR_MODEL_URL_HERE":
    print("ERROR: Model URL is not set. Please edit app.py and replace 'YOUR_MODEL_URL_HERE' with the actual URL.", file=sys.stderr)
else:
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found locally. Downloading from {MODEL_URL}...")
        download_file(MODEL_URL, MODEL_PATH)
    
    if os.path.exists(MODEL_PATH):
        try:
            model = CattleBreedClassifier(num_classes=num_classes)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model.to(device)
            model.eval()
            print("PyTorch model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}", file=sys.stderr)
            model = None
    else:
        print("Model file could not be downloaded or found.", file=sys.stderr)


# --- Image Transformation ---
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# --- Prediction Function ---
def predict(inp_image):
    """
    Takes a PIL image, preprocesses it, and returns a dictionary of
    breed names to probabilities.
    """
    if model is None:
        return {"Error": "Model not loaded. Please check logs for details."}
        
    if not isinstance(inp_image, Image.Image):
        inp_image = Image.fromarray(inp_image)

    img = inp_image.convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    
    confidences = {class_names[i]: float(probabilities[i]) for i in range(num_classes)}
    
    return confidences

# --- Gradio Interface ---
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload Cattle/Buffalo Image"),
    outputs=gr.Label(num_top_classes=5, label="Top 5 Predictions"),
    title="🐄 Indian Cattle & Buffalo Breed Classifier",
    description="Upload a clear image of an Indian cattle or buffalo to classify its breed. The model will return the top 5 most likely breeds.",
    examples=[], # Examples removed as they require local file paths
    article="<p style='text-align: center;'>Built with PyTorch and Gradio</p>"
)

# --- Launch the App ---
if __name__ == "__main__":
    if model is not None:
        print("Launching Gradio app on 0.0.0.0:7860...")
        iface.launch(server_name="0.0.0.0", server_port=7860)
    else:
        print("Could not start Gradio app because the model failed to load.", file=sys.stderr)

# --- Launch the App ---
if __name__ == "__main__":
    if model is not None:
        print("Launching Gradio app...")
        iface.launch()
    else:
        print("Could not start Gradio app because the model failed to load.")