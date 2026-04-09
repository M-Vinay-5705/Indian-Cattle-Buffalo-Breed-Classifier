import torch
from PIL import Image
from torchvision import transforms
from dataclasses import dataclass
from typing import Dict

from cattle_model import CattleBreedClassifier


@dataclass
class PredictionResult:
    predicted_breed: str
    confidence: float
    top_predictions: Dict[str, float]


class BreedPredictor:
    def __init__(self, model_path, data_dir=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 🔥 Hardcoded class names
        self.class_names = [
            "Gir",
            "Murrah",
            "Sahiwal",
            "Holstein Friesian",
            "Jersey",
            "Red Sindhi",
        ]

        self.model = CattleBreedClassifier(num_classes=len(self.class_names))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3),
            ]
        )

    def predict(self, image_input):
        if image_input is None:
            raise ValueError("Upload image")

        if not isinstance(image_input, Image.Image):
            image_input = Image.fromarray(image_input)

        image = image_input.convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]

        top_predictions = {
            self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))
        }

        idx = int(torch.argmax(probs))

        return PredictionResult(
            predicted_breed=self.class_names[idx],
            confidence=float(probs[idx]),
            top_predictions=top_predictions,
        )