"""
Data loading and preprocessing utilities for spoken language detection.

This module handles loading the audio dataset, creating PyTorch datasets,
and data preprocessing including normalization.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Optional


# Constants
LANGUAGES = ("de", "en", "es", "fr", "nl", "pt")
LANGUAGE_DICT = {lang: i for i, lang in enumerate(LANGUAGES)}
LANGUAGE_NAMES = {
    0: "German",
    1: "English", 
    2: "Spanish",
    3: "French",
    4: "Dutch",
    5: "Portuguese"
}
SAMPLING_RATE = 8000
CLIP_DURATION = 5  # seconds
SEQUENCE_LENGTH = SAMPLING_RATE * CLIP_DURATION  # 40,000


class LanguageDataset(Dataset):
    """
    PyTorch Dataset for spoken language detection.
    
    This dataset loads preprocessed audio data and applies normalization
    if specified.
    """
    
    def __init__(self, 
                 data: torch.Tensor, 
                 targets: torch.Tensor,
                 normalize: bool = True,
                 mean: Optional[float] = None,
                 std: Optional[float] = None):
        """
        Initialize the dataset.
        
        Args:
            data (torch.Tensor): Audio data tensor
            targets (torch.Tensor): Language labels tensor
            normalize (bool): Whether to apply normalization
            mean (float, optional): Mean for normalization (computed if None)
            std (float, optional): Std for normalization (computed if None)
        """
        self.data = data.clone()
        self.targets = targets.clone()
        self.normalize = normalize
        
        if normalize:
            if mean is None or std is None:
                self.mean = torch.mean(self.data)
                self.std = torch.std(self.data)
            else:
                self.mean = mean
                self.std = std
            
            # Apply normalization
            self.data = (self.data - self.mean) / (self.std + 1e-8)
        else:
            self.mean = 0.0
            self.std = 1.0
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single data sample.
        
        Args:
            index (int): Index of the sample to retrieve
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (audio_data, target_label)
        """
        x = self.data[index]
        y = self.targets[index]
        
        # Apply normalization if requested
        if self.normalize:
            x = (x - self.mean) / (self.std + 1e-8)  # Add small epsilon for numerical stability
            
        return x, y


def load_data(data_dir: str = "data") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load training and test data from numpy files.
    
    Args:
        data_dir (str): Directory containing the data files
        
    Returns:
        Tuple containing (X_train, y_train, X_test, y_test)
        
    Raises:
        FileNotFoundError: If data files are not found in the specified directory
    """
    try:
        X_train = np.load(os.path.join(data_dir, "inputs_train_fp16.npy"))
        y_train = np.load(os.path.join(data_dir, "targets_train_int8.npy"))
        X_test = np.load(os.path.join(data_dir, "inputs_test_fp16.npy"))
        y_test = np.load(os.path.join(data_dir, "targets_test_int8.npy"))
        
        # Convert to float32 for better performance
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        
        return X_train, y_train, X_test, y_test
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Data files not found in {data_dir}. Expected files: "
                              "inputs_train_fp16.npy, targets_train_int8.npy, "
                              "inputs_test_fp16.npy, targets_test_int8.npy") from e


def create_datasets(
    data_dir: str = "data",
    normalize: bool = True
) -> Tuple[LanguageDataset, LanguageDataset]:
    """
    Create training and test datasets.
    
    Args:
        data_dir (str): Directory containing the data files
        normalize (bool): Whether to apply normalization
        
    Returns:
        Tuple[LanguageDataset, LanguageDataset]: (train_dataset, test_dataset)
    """
    # Load data
    X_train, y_train, X_test, y_test = load_data(data_dir)
    
    # Convert to PyTorch tensors
    X_train = torch.from_numpy(X_train)
    X_test = torch.from_numpy(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    
    # Calculate normalization statistics from training data
    if normalize:
        mean = torch.mean(X_train)
        std = torch.std(X_train)
    else:
        mean = None
        std = None
    
    # Create datasets
    train_dataset = LanguageDataset(X_train, y_train, normalize=normalize, mean=mean, std=std)
    test_dataset = LanguageDataset(X_test, y_test, normalize=normalize, mean=mean, std=std)
    
    return train_dataset, test_dataset


def create_data_loaders(
    train_dataset: LanguageDataset,
    test_dataset: LanguageDataset,
    batch_size: int = 1,
    shuffle_train: bool = True,
    shuffle_test: bool = False,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders from datasets.
    
    Args:
        train_dataset (LanguageDataset): Training dataset
        test_dataset (LanguageDataset): Test dataset
        batch_size (int): Batch size for data loading
        shuffle_train (bool): Whether to shuffle training data
        shuffle_test (bool): Whether to shuffle test data
        num_workers (int): Number of worker processes for data loading
        
    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, test_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle_test,
        num_workers=num_workers
    )
    
    return train_loader, test_loader


def get_dataset_info(train_dataset: LanguageDataset, test_dataset: LanguageDataset) -> dict:
    """
    Get information about the datasets.
    
    Args:
        train_dataset (LanguageDataset): Training dataset
        test_dataset (LanguageDataset): Test dataset
        
    Returns:
        dict: Dataset information
    """
    # Get sample shapes
    sample_x, sample_y = train_dataset[0]
    
    # Get unique classes
    train_classes = torch.unique(train_dataset.targets)
    test_classes = torch.unique(test_dataset.targets)
    
    return {
        'train_size': len(train_dataset),
        'test_size': len(test_dataset),
        'sample_shape': sample_x.shape,
        'num_classes': len(train_classes),
        'train_classes': train_classes.tolist(),
        'test_classes': test_classes.tolist(),
        'normalization': {
            'mean': train_dataset.mean.item() if train_dataset.mean is not None else None,
            'std': train_dataset.std.item() if train_dataset.std is not None else None
        }
    }


# Language mapping constants
LANGUAGES = ["de", "en", "es", "fr", "nl", "pt"]
LANGUAGE_DICT = {lang: i for i, lang in enumerate(LANGUAGES)}
LANGUAGE_NAMES = {
    0: "German",
    1: "English", 
    2: "Spanish",
    3: "French",
    4: "Dutch",
    5: "Portuguese"
}

# Audio parameters
SAMPLING_RATE = 8000  # 8 kHz
CLIP_DURATION = 5     # 5 seconds
SEQUENCE_LENGTH = SAMPLING_RATE * CLIP_DURATION  # 40,000 samples


if __name__ == "__main__":
    # Test data loading
    print("Testing data loading...")
    
    try:
        train_dataset, test_dataset = create_datasets(data_dir="../data")
        train_loader, test_loader = create_data_loaders(train_dataset, test_dataset)
        
        print("Data loaded successfully!")
        print(f"Dataset info: {get_dataset_info(train_dataset, test_dataset)}")
        
        # Test a batch
        for batch_x, batch_y in train_loader:
            print(f"Batch shape: {batch_x.shape}, {batch_y.shape}")
            break
            
    except Exception as e:
        print(f"Error loading data: {e}")