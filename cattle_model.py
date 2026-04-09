import torch
import torch.nn as nn
import torch.nn.functional as F

class CattleBreedClassifier(nn.Module):
    """
    PyTorch model for cattle breed classification.
    """
    def __init__(self, num_classes):
        super(CattleBreedClassifier, self).__init__()
        # Input channels = 3 (for RGB images)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # The image size is 128x128. After 3 pooling layers, it becomes 128 / 2^3 = 16.
        # So the flattened size will be 128 * 16 * 16
        self.fc1 = nn.Linear(128 * 16 * 16, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        
        # Flatten the tensor for the fully connected layer
        x = x.view(-1, 128 * 16 * 16)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x