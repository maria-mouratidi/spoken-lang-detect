"""
Evaluation script for the trained language detection model.

This script loads a trained model and evaluates it on the test set,
generating detailed metrics and visualizations.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import LanguageClassifier
from data import create_datasets, create_data_loaders, LANGUAGE_NAMES
from utils.config import Config, get_default_config


class ModelEvaluator:
    """Evaluator class for the trained language classification model."""
    
    def __init__(self, model_path: str, config: Config = None):
        """
        Initialize the evaluator.
        
        Args:
            model_path (str): Path to the trained model
            config (Config): Configuration object (optional)
        """
        self.model_path = model_path
        
        # Load model and config
        self.checkpoint = torch.load(model_path, map_location='cpu')
        
        if config is None:
            # Try to load config from checkpoint
            if 'config' in self.checkpoint:
                self.config = Config()
                # Update with saved config
                for key, value in self.checkpoint['config'].items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
            else:
                self.config = get_default_config()
        else:
            self.config = config
        
        self.device = self.config.get_device()
        
        # Load model
        self.model = self._load_model()
        
        # Create data loaders
        self.train_loader, self.test_loader = self._create_data_loaders()
        
        print(f"Model loaded from: {model_path}")
        print(f"Test accuracy from checkpoint: {self.checkpoint.get('test_accuracy', 'N/A')}")
    
    def _load_model(self) -> LanguageClassifier:
        """Load the trained model."""
        model = LanguageClassifier(
            num_classes=self.config.model.num_classes,
            dropout_rate=self.config.model.dropout_rate
        )
        model.load_state_dict(self.checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model
    
    def _create_data_loaders(self):
        """Create data loaders."""
        train_dataset, test_dataset = create_datasets(
            data_dir=self.config.data.data_dir,
            normalize=self.config.data.normalize
        )
        
        train_loader, test_loader = create_data_loaders(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            batch_size=32,  # Larger batch for evaluation
            shuffle_train=False,
            shuffle_test=False,
            num_workers=self.config.data.num_workers
        )
        
        return train_loader, test_loader
    
    def predict(self, data_loader) -> tuple:
        """
        Generate predictions for a data loader.
        
        Args:
            data_loader: PyTorch data loader
            
        Returns:
            tuple: (predictions, true_labels, features)
        """
        all_predictions = []
        all_labels = []
        all_features = []
        
        with torch.no_grad():
            for batch_inputs, batch_targets in data_loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                # Reshape input
                batch_inputs = batch_inputs.view(batch_inputs.size(0), 1, -1)
                
                # Get predictions and features
                outputs = self.model(batch_inputs)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_targets.cpu().numpy())
                all_features.extend(outputs.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_labels), np.array(all_features)
    
    def evaluate_test_set(self):
        """Evaluate the model on the test set."""
        print("Evaluating on test set...")
        
        predictions, true_labels, features = self.predict(self.test_loader)
        
        # Calculate accuracy
        accuracy = np.mean(predictions == true_labels)
        print(f"Test Accuracy: {accuracy:.4f}")
        
        # Generate classification report
        print("\nClassification Report:")
        target_names = [LANGUAGE_NAMES[i] for i in range(self.config.model.num_classes)]
        print(classification_report(true_labels, predictions, target_names=target_names))
        
        return predictions, true_labels, features, accuracy
    
    def plot_confusion_matrix(self, predictions, true_labels, save_path=None):
        """
        Plot confusion matrix.
        
        Args:
            predictions: Model predictions
            true_labels: True labels
            save_path: Path to save the plot (optional)
        """
        cm = confusion_matrix(true_labels, predictions)
        
        plt.figure(figsize=(10, 8))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create labels
        labels = [LANGUAGE_NAMES[i] for i in range(self.config.model.num_classes)]
        
        # Plot
        sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix (Normalized)')
        plt.xlabel('Predicted Language')
        plt.ylabel('True Language')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm
    
    def plot_pca_analysis(self, features, labels, save_path=None):
        """
        Plot PCA analysis of the model's output space.
        
        Args:
            features: Model output features
            labels: True labels
            save_path: Path to save the plot (optional)
        """
        # Perform PCA
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)
        
        plt.figure(figsize=(12, 8))
        
        # Create scatter plot
        colors = plt.cm.Set3(np.linspace(0, 1, self.config.model.num_classes))
        
        for i in range(self.config.model.num_classes):
            mask = labels == i
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                       c=[colors[i]], label=LANGUAGE_NAMES[i], alpha=0.7)
        
        plt.xlabel(f'First Principal Component (Explained Variance: {pca.explained_variance_ratio_[0]:.3f})')
        plt.ylabel(f'Second Principal Component (Explained Variance: {pca.explained_variance_ratio_[1]:.3f})')
        plt.title('PCA Analysis of Model Output Space')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.3f}")
        
        return features_2d, pca
    
    def analyze_errors(self, predictions, true_labels):
        """
        Analyze prediction errors.
        
        Args:
            predictions: Model predictions
            true_labels: True labels
        """
        print("\nError Analysis:")
        
        # Find misclassified samples
        errors = predictions != true_labels
        error_count = np.sum(errors)
        
        print(f"Total errors: {error_count} out of {len(predictions)} ({error_count/len(predictions)*100:.2f}%)")
        
        if error_count > 0:
            print("\nMost common misclassifications:")
            for true_class in range(self.config.model.num_classes):
                true_mask = true_labels == true_class
                true_errors = errors & true_mask
                
                if np.sum(true_errors) > 0:
                    pred_classes = predictions[true_errors]
                    unique, counts = np.unique(pred_classes, return_counts=True)
                    
                    print(f"\n{LANGUAGE_NAMES[true_class]} misclassified as:")
                    for pred_class, count in zip(unique, counts):
                        print(f"  {LANGUAGE_NAMES[pred_class]}: {count} times")
    
    def plot_training_history(self, save_path=None):
        """
        Plot training history if available.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if 'train_losses' not in self.checkpoint:
            print("Training history not available in checkpoint.")
            return
        
        train_losses = self.checkpoint['train_losses']
        train_accs = self.checkpoint['train_accuracies']
        test_losses = self.checkpoint['test_losses']
        test_accs = self.checkpoint['test_accuracies']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        epochs = range(1, len(train_losses) + 1)
        ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
        ax1.plot(epochs, test_losses, 'r-', label='Test Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Test Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracies
        ax2.plot(epochs, train_accs, 'b-', label='Train Accuracy')
        ax2.plot(epochs, test_accs, 'r-', label='Test Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Test Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_full_evaluation(self, output_dir="outputs"):
        """
        Run complete evaluation and save results.
        
        Args:
            output_dir: Directory to save evaluation results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("Running full evaluation...")
        
        # Evaluate test set
        predictions, true_labels, features, accuracy = self.evaluate_test_set()
        
        # Plot confusion matrix
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        self.plot_confusion_matrix(predictions, true_labels, cm_path)
        
        # Plot PCA analysis
        pca_path = os.path.join(output_dir, "pca_analysis.png")
        self.plot_pca_analysis(features, true_labels, pca_path)
        
        # Analyze errors
        self.analyze_errors(predictions, true_labels)
        
        # Plot training history
        history_path = os.path.join(output_dir, "training_history.png")
        self.plot_training_history(history_path)
        
        print(f"\nEvaluation results saved to: {output_dir}")
        
        return {
            'accuracy': accuracy,
            'predictions': predictions,
            'true_labels': true_labels,
            'features': features
        }


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate trained language detection model")
    parser.add_argument("--model_path", type=str, default="saved_models/best_model.pt",
                       help="Path to the trained model")
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="Directory to save evaluation results")
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Model not found at: {args.model_path}")
        print("Available models:")
        model_dir = os.path.dirname(args.model_path)
        if os.path.exists(model_dir):
            for f in os.listdir(model_dir):
                if f.endswith('.pt'):
                    print(f"  {os.path.join(model_dir, f)}")
        return
    
    # Create evaluator and run evaluation
    evaluator = ModelEvaluator(args.model_path)
    evaluator.run_full_evaluation(args.output_dir)


if __name__ == "__main__":
    main()