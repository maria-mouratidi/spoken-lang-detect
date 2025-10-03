"""
Configuration management for the spoken language detection project.

This module contains all configurable parameters for training, model architecture,
and data processing.
"""

import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    num_classes: int = 6
    dropout_rate: float = 0.5
    
    # Convolutional layer parameters
    conv1_out_channels: int = 32
    conv1_kernel_size: int = 50
    conv1_stride: int = 5
    
    conv2_out_channels: int = 64
    conv2_kernel_size: int = 3
    conv2_stride: int = 1
    
    conv3_out_channels: int = 128
    conv3_kernel_size: int = 3
    conv3_stride: int = 1
    
    conv4_out_channels: int = 256
    conv4_kernel_size: int = 3
    conv4_stride: int = 1
    
    # Pooling parameters
    pool_kernel_size: int = 3
    pool_stride: int = 2
    
    # Fully connected layer parameters
    fc1_out_features: int = 128
    fc2_out_features: int = 64


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    num_epochs: int = 60
    batch_size: int = 1
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    
    # Optimizer settings
    optimizer: str = "adam"  # "adam" or "sgd"
    momentum: float = 0.9  # for SGD
    
    # Learning rate scheduling
    use_scheduler: bool = False
    scheduler_step_size: int = 20
    scheduler_gamma: float = 0.1
    
    # Early stopping
    use_early_stopping: bool = False
    patience: int = 10
    min_delta: float = 0.001
    
    # Validation
    validation_split: float = 0.2
    shuffle_train: bool = True
    shuffle_test: bool = False
    
    # Logging
    log_interval: int = 10  # Log every N steps
    save_interval: int = 10  # Save model every N epochs


@dataclass
class DataConfig:
    """Configuration for data processing."""
    data_dir: str = "data"
    normalize: bool = True
    
    # Audio parameters
    sampling_rate: int = 8000
    clip_duration: int = 5  # seconds
    sequence_length: int = 40000  # sampling_rate * clip_duration
    
    # Data loading
    num_workers: int = 0
    pin_memory: bool = False
    
    # Languages
    languages: tuple = ("de", "en", "es", "fr", "nl", "pt")
    language_names: dict = None
    
    def __post_init__(self):
        if self.language_names is None:
            self.language_names = {
                0: "German",
                1: "English", 
                2: "Spanish",
                3: "French",
                4: "Dutch",
                5: "Portuguese"
            }


@dataclass
class SystemConfig:
    """Configuration for system settings."""
    device: str = "auto"  # "auto", "cuda", "cpu"
    seed: int = 42
    deterministic: bool = True
    
    # Paths
    model_save_dir: str = "saved_models"
    log_dir: str = "logs"
    output_dir: str = "outputs"
    
    # Model saving
    save_best_only: bool = True
    model_filename: str = "language_classifier.pt"
    
    def get_device(self) -> torch.device:
        """Get the appropriate device for computation."""
        if self.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
                print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = torch.device("cpu")
                print("Using CPU")
        else:
            device = torch.device(self.device)
            print(f"Using device: {device}")
        
        return device


@dataclass
class Config:
    """Main configuration class combining all configs."""
    model: ModelConfig = None
    training: TrainingConfig = None
    data: DataConfig = None
    system: SystemConfig = None
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.system is None:
            self.system = SystemConfig()
    
    def setup_environment(self):
        """Setup the environment for reproducible training."""
        if self.system.seed is not None:
            torch.manual_seed(self.system.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.system.seed)
                torch.cuda.manual_seed_all(self.system.seed)
        
        if self.system.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def get_device(self) -> torch.device:
        """Get the computation device."""
        return self.system.get_device()
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'system': self.system.__dict__
        }


def get_default_config() -> Config:
    """Get the default configuration."""
    return Config()


def get_binary_classification_config() -> Config:
    """Get configuration for binary classification (English vs Spanish)."""
    config = get_default_config()
    config.model.num_classes = 2
    return config


def get_debug_config() -> Config:
    """Get configuration for debugging (fewer epochs, smaller model)."""
    config = get_default_config()
    config.training.num_epochs = 5
    config.training.log_interval = 1
    config.model.conv1_out_channels = 16
    config.model.conv2_out_channels = 32
    config.model.conv3_out_channels = 64
    config.model.conv4_out_channels = 128
    return config


if __name__ == "__main__":
    # Test configuration
    config = get_default_config()
    print("Default configuration:")
    print(f"Model classes: {config.model.num_classes}")
    print(f"Training epochs: {config.training.num_epochs}")
    print(f"Data directory: {config.data.data_dir}")
    print(f"Device: {config.get_device()}")
    
    print("\nConfiguration as dict:")
    import pprint
    pprint.pprint(config.to_dict())