"""
Training script for the spoken language detection model.

This script handles the complete training pipeline including data loading,
model creation, training loop, and model saving.
"""

import os
import sys
import time
from typing import Tuple, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import LanguageClassifier
from data import create_datasets, create_data_loaders, get_dataset_info
from utils.config import Config, get_default_config


class Trainer:
    """Trainer class for the language classification model."""
    
    def __init__(self, config: Config):
        """
        Initialize the trainer.
        
        Args:
            config (Config): Configuration object
        """
        self.config = config
        self.device = config.get_device()
        
        # Setup environment for reproducibility
        config.setup_environment()
        
        # Initialize tracking variables
        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []
        self.best_test_accuracy = 0.0
        
        # Create model
        self.model = self._create_model()
        
        # Create optimizer and loss function
        self.optimizer = self._create_optimizer()
        self.criterion = nn.CrossEntropyLoss()
        
        # Create data loaders
        self.train_loader, self.test_loader = self._create_data_loaders()
        
        print(f"Trainer initialized successfully!")
        print(f"Model parameters: {self.model.get_model_info()}")
    
    def _create_model(self) -> LanguageClassifier:
        """Create and initialize the model."""
        model = LanguageClassifier(
            num_classes=self.config.model.num_classes,
            dropout_rate=self.config.model.dropout_rate
        )
        model.to(self.device)
        return model
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create the optimizer."""
        if self.config.training.optimizer.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer.lower() == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                momentum=self.config.training.momentum,
                weight_decay=self.config.training.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.training.optimizer}")
    
    def _create_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create training and test data loaders."""
        train_dataset, test_dataset = create_datasets(
            data_dir=self.config.data.data_dir,
            normalize=self.config.data.normalize
        )
        
        print("Dataset information:")
        print(get_dataset_info(train_dataset, test_dataset))
        
        train_loader, test_loader = create_data_loaders(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            batch_size=self.config.training.batch_size,
            shuffle_train=self.config.training.shuffle_train,
            shuffle_test=self.config.training.shuffle_test,
            num_workers=self.config.data.num_workers
        )
        
        return train_loader, test_loader
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train the model for one epoch.
        
        Returns:
            Tuple[float, float]: (average_loss, average_accuracy)
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, (batch_inputs, batch_targets) in enumerate(self.train_loader):
            # Move data to device
            batch_inputs = batch_inputs.to(self.device)
            batch_targets = batch_targets.to(self.device)
            
            # Reshape input for CNN (add channel dimension)
            batch_inputs = batch_inputs.view(batch_inputs.size(0), 1, -1)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(batch_inputs)
            loss = self.criterion(predictions, batch_targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(predictions, 1)
            correct = (predicted == batch_targets).sum().item()
            
            total_loss += loss.item()
            total_correct += correct
            total_samples += batch_targets.size(0)
        
        avg_loss = total_loss / len(self.train_loader)
        avg_accuracy = total_correct / total_samples
        
        return avg_loss, avg_accuracy
    
    def evaluate(self) -> Tuple[float, float]:
        """
        Evaluate the model on the test set.
        
        Returns:
            Tuple[float, float]: (average_loss, average_accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_inputs, batch_targets in self.test_loader:
                # Move data to device
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                # Reshape input for CNN
                batch_inputs = batch_inputs.view(batch_inputs.size(0), 1, -1)
                
                # Forward pass
                predictions = self.model(batch_inputs)
                loss = self.criterion(predictions, batch_targets)
                
                # Calculate accuracy
                _, predicted = torch.max(predictions, 1)
                correct = (predicted == batch_targets).sum().item()
                
                total_loss += loss.item()
                total_correct += correct
                total_samples += batch_targets.size(0)
        
        avg_loss = total_loss / len(self.test_loader)
        avg_accuracy = total_correct / total_samples
        
        return avg_loss, avg_accuracy
    
    def save_model(self, epoch: int, is_best: bool = False):
        """
        Save the model state.
        
        Args:
            epoch (int): Current epoch number
            is_best (bool): Whether this is the best model so far
        """
        os.makedirs(self.config.system.model_save_dir, exist_ok=True)
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.config.system.model_save_dir,
            f"model_epoch_{epoch}.pt"
        )
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'test_losses': self.test_losses,
            'test_accuracies': self.test_accuracies
        }, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(
                self.config.system.model_save_dir,
                "best_model.pt"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config.to_dict(),
                'test_accuracy': self.best_test_accuracy
            }, best_path)
    
    def train(self):
        """Run the complete training loop."""
        print(f"Starting training for {self.config.training.num_epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(self.config.training.num_epochs):
            epoch_start_time = time.time()
            
            # Train for one epoch
            train_loss, train_acc = self.train_epoch()
            
            # Evaluate on test set
            test_loss, test_acc = self.evaluate()
            
            # Update tracking
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.test_losses.append(test_loss)
            self.test_accuracies.append(test_acc)
            
            # Check if this is the best model
            is_best = test_acc > self.best_test_accuracy
            if is_best:
                self.best_test_accuracy = test_acc
            
            # Print progress
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch [{epoch+1}/{self.config.training.num_epochs}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, "
                  f"Time: {epoch_time:.2f}s")
            
            # Save model
            if (epoch + 1) % self.config.training.save_interval == 0:
                self.save_model(epoch + 1, is_best)
        
        # Save final model
        self.save_model(self.config.training.num_epochs, 
                       self.test_accuracies[-1] == self.best_test_accuracy)
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f} seconds")
        print(f"Best test accuracy: {self.best_test_accuracy:.4f}")


def main():
    """Main training function."""
    # Get configuration
    config = get_default_config()
    
    # Override config for quick testing if needed
    # config.training.num_epochs = 5
    
    # Create trainer and start training
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()