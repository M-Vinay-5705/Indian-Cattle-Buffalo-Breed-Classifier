import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from PIL import Image
from torchvision import transforms

from cattle_model import CattleBreedClassifier


@dataclass
class PredictionResult:
    predicted_breed: str
    confidence: float
    top_predictions: Dict[str, float]


class BreedPredictor:
    """Loads the trained CNN model and performs breed prediction."""

    def __init__(
        self,
        model_path: str,
        data_dir: str,
        img_size: int = 128,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model_path = model_path
        self.data_dir = data_dir
        self.img_size = img_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = self._load_class_names()
        self.model = self._load_model()
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def _load_class_names(self) -> List[str]:
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"Dataset folder not found: {self.data_dir}")
        return sorted(
            [item for item in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, item))]
        )

    def _load_model(self) -> CattleBreedClassifier:
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        model = CattleBreedClassifier(num_classes=len(self.class_names))
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def predict(self, image_input) -> PredictionResult:
        """Returns the top prediction and all class confidences for a given image."""
        if image_input is None:
            raise ValueError("Please upload an image before predicting.")

        if not isinstance(image_input, Image.Image):
            image_input = Image.fromarray(image_input)

        image = image_input.convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

        top_predictions = {
            self.class_names[index]: float(probabilities[index])
            for index in range(len(self.class_names))
        }
        predicted_index = int(torch.argmax(probabilities).item())

        return PredictionResult(
            predicted_breed=self.class_names[predicted_index],
            confidence=float(probabilities[predicted_index]),
            top_predictions=top_predictions,
        )
