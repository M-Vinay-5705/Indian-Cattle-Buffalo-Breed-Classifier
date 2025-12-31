import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

from cattle_model import CattleBreedClassifier

def plot_training_history(history):
    """Plots the training and validation accuracy and loss."""
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    train_acc = history['train_acc']
    val_acc = history['val_acc']
    
    epochs = range(1, len(train_acc) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(epochs, train_acc, 'bo', label='Training acc')
    ax1.plot(epochs, val_acc, 'b', label='Validation acc')
    ax1.set_title('Training and validation accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(epochs, train_loss, 'ro', label='Training loss')
    ax2.plot(epochs, val_loss, 'r', label='Validation loss')
    ax2.set_title('Training and validation loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("\nTraining history saved to 'training_history.png'")

def predict_single_image(model_path, img_path, class_names, device, img_size=128):
    """Loads a model and predicts a single image."""
    try:
        # Load the model architecture and state
        num_classes = len(class_names)
        model = CattleBreedClassifier(num_classes=num_classes)
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model.eval()
        print(f"\nModel loaded from {model_path}")

        # Image transformations
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_class_index = torch.max(probabilities, 1)

        predicted_breed = class_names[predicted_class_index.item()]
        
        print("\n" + "="*50)
        print("Making sample prediction...")
        print("="*50)
        print(f"Image: {os.path.basename(img_path)}")
        print(f"Predicted Breed: {predicted_breed}")
        print(f"Confidence: {confidence.item() * 100:.2f}%")
        print("\nAll probabilities:")
        for i, prob in enumerate(probabilities[0]):
            print(f"  {class_names[i]}: {prob.item()*100:.2f}%")

    except Exception as e:
        print(f"Error during prediction: {e}")

# Main execution
if __name__ == "__main__":
    # --- Configuration ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.join(SCRIPT_DIR, "IndianCattleBuffaloeBreeds-Dataset", "breeds")
    
    if not os.path.exists(BASE_DIR):
        print(f"Warning: Dataset not found at relative path, checking absolute...")
        BASE_DIR = r"C:\GIET_Study\Sem-6\project\t1\IndianCattleBuffaloeBreeds-Dataset\breeds"

    TRAIN_DIR = os.path.join(BASE_DIR, "train")
    TEST_DIR = os.path.join(BASE_DIR, "test")

    IMG_SIZE = 128
    BATCH_SIZE = 32
    EPOCHS = 15
    LEARNING_RATE = 0.001
    
    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading and Augmentation ---
    print("Preparing data loaders...")
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize for 3 channels
        ]),
        'val': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]),
    }

    image_datasets = {
        'train': datasets.ImageFolder(TRAIN_DIR, data_transforms['train']),
        'val': datasets.ImageFolder(TEST_DIR, data_transforms['val'])
    }
    
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
        'val': DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    }

    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}\n")

    # --- Model, Optimizer, and Loss Function ---
    model = CattleBreedClassifier(num_classes=num_classes).to(device)
    print("Model architecture:")
    print(model)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Training Loop ---
    print("\nStarting training...")
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

    # --- Plotting and Saving ---
    plot_training_history(history)
    
    model_save_path = os.path.join(SCRIPT_DIR, "cattle_breed_classifier.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"\nModel saved to {model_save_path}")

    # --- Prediction Example ---
    sample_img_path = None
    if os.path.exists(TEST_DIR):
        for breed_dir in os.listdir(TEST_DIR):
            breed_path = os.path.join(TEST_DIR, breed_dir)
            if os.path.isdir(breed_path):
                for img_file in os.listdir(breed_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        sample_img_path = os.path.join(breed_path, img_file)
                        break
            if sample_img_path:
                break
    
    if sample_img_path:
        predict_single_image(model_save_path, sample_img_path, class_names, device, img_size=IMG_SIZE)
    else:
        print("\nCould not find a sample image to test prediction.")

    print("\n" + "="*50)
    print("Process Complete!")
    print("="*50)