"""
Language Classification Model Architecture

This module contains the CNN model for spoken language detection.
"""

import torch
import torch.nn as nn
from typing import Tuple


class LanguageClassifier(nn.Module):
    """
    Convolutional Neural Network for spoken language classification.
    
    Architecture:
    - 4 Convolutional layers with increasing filters (32 → 64 → 128 → 256)
    - MaxPooling layers for dimensionality reduction
    - Adaptive average pooling for handling variable input lengths
    - Fully connected layers with dropout for classification
    
    Args:
        num_classes (int): Number of language classes to predict (default: 6)
        dropout_rate (float): Dropout probability (default: 0.5)
    """
    
    def __init__(self, num_classes: int = 6, dropout_rate: float = 0.5):
        super(LanguageClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(1, 32, kernel_size=50, stride=5)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool1d(kernel_size=3, stride=2)
        
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, stride=1)
        self.relu4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool1d(kernel_size=3, stride=2)
        
        # Adaptive pooling to handle variable input lengths
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        
        # Fully connected layers
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.relu_fc = nn.ReLU()
        self.fc_out = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 1, sequence_length]
            
        Returns:
            torch.Tensor: Output logits of shape [batch_size, num_classes]
        """
        # Convolutional layers with pooling
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)
       
        # Adaptive pooling and flattening
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        
        # Fully connected layers with dropout
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)  
        x = self.relu_fc(x)
        x = self.fc_out(x)

        return x
    
    def get_model_info(self) -> dict:
        """
        Get information about the model architecture.
        
        Returns:
            dict: Model information including parameter count
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate
        }


def create_model(num_classes: int = 6, dropout_rate: float = 0.5) -> LanguageClassifier:
    """
    Factory function to create a LanguageClassifier model.
    
    Args:
        num_classes (int): Number of language classes to predict
        dropout_rate (float): Dropout probability
        
    Returns:
        LanguageClassifier: Initialized model
    """
    return LanguageClassifier(num_classes=num_classes, dropout_rate=dropout_rate)


if __name__ == "__main__":
    # Test the model
    model = create_model()
    print("Model created successfully!")
    print(f"Model info: {model.get_model_info()}")
    
    # Test with dummy input
    dummy_input = torch.randn(1, 1, 40000)  # 5 seconds at 8kHz
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")