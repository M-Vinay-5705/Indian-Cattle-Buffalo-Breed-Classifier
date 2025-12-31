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
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload Cattle/Buffalo Image"),
    outputs=gr.Label(num_top_classes=5, label="Top 5 Predictions"),
    title="🐄 Indian Cattle & Buffalo Breed Classifier",
    description="Upload a clear image of an Indian cattle or buffalo to classify its breed. The model will return the top 5 most likely breeds.",
    examples=[
        os.path.join(SCRIPT_DIR, "IndianCattleBuffaloeBreeds-Dataset", "breeds", "test", "Gir", "Gir_001.jpg"),
        os.path.join(SCRIPT_DIR, "IndianCattleBuffaloeBreeds-Dataset", "breeds", "test", "Murrah", "Murrah_001.jpg"),
        os.path.join(SCRIPT_DIR, "IndianCattleBuffaloeBreeds-Dataset", "breeds", "test", "Sahiwal", "Sahiwal_001.jpg"),
    ],
    article="<p style='text-align: center;'>Built with PyTorch and Gradio</p>"
)

# --- Launch the App ---
if __name__ == "__main__":
    if model is not None:
        print("Launching Gradio app...")
        iface.launch()
    else:
        print("Could not start Gradio app because the model failed to load.")