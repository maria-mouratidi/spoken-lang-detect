"""
Visualization utilities for the spoken language detection project.

This module provides functions for creating plots and visualizations
for model analysis and data exploration.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
import torch


# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_waveforms(audio_data: np.ndarray, 
                   labels: np.ndarray, 
                   language_names: Dict[int, str],
                   num_samples: int = 5,
                   sampling_rate: int = 8000,
                   save_path: Optional[str] = None) -> None:
    """
    Plot waveforms of randomly selected audio samples.
    
    Args:
        audio_data: Array of audio waveforms
        labels: Array of language labels
        language_names: Mapping from label indices to language names
        num_samples: Number of samples to plot
        sampling_rate: Audio sampling rate
        save_path: Path to save the plot (optional)
    """
    # Randomly select samples
    indices = np.random.choice(len(audio_data), num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 2*num_samples))
    if num_samples == 1:
        axes = [axes]
    
    # Time axis (in seconds)
    time_axis = np.arange(audio_data.shape[1]) / sampling_rate
    
    for i, idx in enumerate(indices):
        waveform = audio_data[idx]
        label = labels[idx]
        language = language_names.get(label, f"Language {label}")
        
        axes[i].plot(time_axis, waveform)
        axes[i].set_title(f'Sample {idx}: {language}')
        axes[i].set_xlabel('Time (seconds)')
        axes[i].set_ylabel('Amplitude')
        axes[i].grid(True, alpha=0.3)
        
        # Set consistent y-axis limits
        max_amp = np.max(np.abs(audio_data))
        axes[i].set_ylim(-max_amp, max_amp)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_training_curves(train_losses: List[float],
                        train_accuracies: List[float],
                        test_losses: List[float],
                        test_accuracies: List[float],
                        save_path: Optional[str] = None) -> None:
    """
    Plot training and validation curves.
    
    Args:
        train_losses: Training loss values
        train_accuracies: Training accuracy values
        test_losses: Test loss values
        test_accuracies: Test accuracy values
        save_path: Path to save the plot (optional)
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, test_losses, 'r-', label='Test Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracies
    ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, test_accuracies, 'r-', label='Test Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(confusion_matrix: np.ndarray,
                         language_names: List[str],
                         normalize: bool = True,
                         save_path: Optional[str] = None) -> None:
    """
    Plot confusion matrix with better formatting.
    
    Args:
        confusion_matrix: Confusion matrix array
        language_names: List of language names
        normalize: Whether to normalize the matrix
        save_path: Path to save the plot (optional)
    """
    if normalize:
        cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        fmt = '.3f'
        title = 'Normalized Confusion Matrix'
    else:
        cm = confusion_matrix
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=language_names, yticklabels=language_names,
                cbar_kws={'label': 'Accuracy' if normalize else 'Count'})
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Language', fontsize=12)
    plt.ylabel('True Language', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_pca_analysis(features: np.ndarray,
                     labels: np.ndarray,
                     language_names: Dict[int, str],
                     save_path: Optional[str] = None) -> Tuple[np.ndarray, Any]:
    """
    Plot PCA analysis of model features.
    
    Args:
        features: Model output features
        labels: True labels
        language_names: Mapping from indices to language names
        save_path: Path to save the plot (optional)
        
    Returns:
        Tuple of (transformed_features, pca_object)
    """
    from sklearn.decomposition import PCA
    
    # Perform PCA
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)
    
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot with different colors for each language
    unique_labels = np.unique(labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        language_name = language_names.get(label, f"Language {label}")
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=[colors[i]], label=language_name, alpha=0.7, s=50)
    
    plt.xlabel(f'First Principal Component (Explained Variance: {pca.explained_variance_ratio_[0]:.3f})')
    plt.ylabel(f'Second Principal Component (Explained Variance: {pca.explained_variance_ratio_[1]:.3f})')
    plt.title('PCA Analysis of Model Output Space', fontsize=16, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.3f}")
    
    return features_2d, pca


def plot_model_architecture(model: torch.nn.Module,
                           input_shape: Tuple[int, ...] = (1, 1, 40000),
                           save_path: Optional[str] = None) -> None:
    """
    Visualize model architecture (requires torchview).
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        save_path: Path to save the plot (optional)
    """
    try:
        from torchview import draw_graph
        
        model_graph = draw_graph(
            model, 
            input_size=input_shape,
            expand_nested=True,
            graph_name='Language Classifier'
        )
        
        if save_path:
            model_graph.visual_graph.render(save_path, format='png')
            print(f"Model architecture saved to: {save_path}")
        else:
            # Display in notebook
            model_graph.visual_graph
            
    except ImportError:
        print("torchview not available. Install with: pip install torchview")
        print("Model summary:")
        print(model)


def plot_class_distribution(labels: np.ndarray,
                           language_names: Dict[int, str],
                           save_path: Optional[str] = None) -> None:
    """
    Plot distribution of classes in the dataset.
    
    Args:
        labels: Array of labels
        language_names: Mapping from indices to language names
        save_path: Path to save the plot (optional)
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # Create DataFrame for easier plotting
    df = pd.DataFrame({
        'Language': [language_names.get(label, f"Language {label}") for label in unique_labels],
        'Count': counts
    })
    
    plt.figure(figsize=(10, 6))
    
    # Create bar plot
    bars = plt.bar(df['Language'], df['Count'], color=sns.color_palette("husl", len(df)))
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01*max(counts),
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Distribution of Languages in Dataset', fontsize=16, fontweight='bold')
    plt.xlabel('Language', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Print statistics
    print(f"Total samples: {len(labels)}")
    print(f"Number of classes: {len(unique_labels)}")
    print(f"Average samples per class: {len(labels) / len(unique_labels):.1f}")


def create_training_report(model_info: Dict[str, Any],
                          training_history: Dict[str, List[float]],
                          test_accuracy: float,
                          confusion_matrix: np.ndarray,
                          language_names: List[str],
                          save_path: Optional[str] = None) -> str:
    """
    Create a comprehensive training report.
    
    Args:
        model_info: Dictionary with model information
        training_history: Training curves data
        test_accuracy: Final test accuracy
        confusion_matrix: Confusion matrix
        language_names: List of language names
        save_path: Path to save the report (optional)
        
    Returns:
        Report string
    """
    report = []
    report.append("=" * 60)
    report.append("SPOKEN LANGUAGE DETECTION - TRAINING REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Model information
    report.append("MODEL INFORMATION:")
    report.append("-" * 30)
    report.append(f"Total Parameters: {model_info.get('total_parameters', 'N/A'):,}")
    report.append(f"Trainable Parameters: {model_info.get('trainable_parameters', 'N/A'):,}")
    report.append(f"Number of Classes: {model_info.get('num_classes', 'N/A')}")
    report.append(f"Dropout Rate: {model_info.get('dropout_rate', 'N/A')}")
    report.append("")
    
    # Training results
    report.append("TRAINING RESULTS:")
    report.append("-" * 30)
    final_train_acc = training_history.get('train_accuracies', [])[-1] if training_history.get('train_accuracies') else 'N/A'
    final_train_loss = training_history.get('train_losses', [])[-1] if training_history.get('train_losses') else 'N/A'
    final_test_loss = training_history.get('test_losses', [])[-1] if training_history.get('test_losses') else 'N/A'
    
    report.append(f"Final Training Accuracy: {final_train_acc:.4f}" if isinstance(final_train_acc, float) else f"Final Training Accuracy: {final_train_acc}")
    report.append(f"Final Test Accuracy: {test_accuracy:.4f}")
    report.append(f"Final Training Loss: {final_train_loss:.4f}" if isinstance(final_train_loss, float) else f"Final Training Loss: {final_train_loss}")
    report.append(f"Final Test Loss: {final_test_loss:.4f}" if isinstance(final_test_loss, float) else f"Final Test Loss: {final_test_loss}")
    report.append("")
    
    # Per-class accuracy
    if confusion_matrix is not None:
        report.append("PER-CLASS ACCURACY:")
        report.append("-" * 30)
        class_accuracies = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
        for i, (lang, acc) in enumerate(zip(language_names, class_accuracies)):
            report.append(f"{lang:12}: {acc:.4f}")
        report.append("")
    
    # Most confused pairs
    if confusion_matrix is not None:
        report.append("MOST CONFUSED LANGUAGE PAIRS:")
        report.append("-" * 30)
        # Normalize confusion matrix
        cm_norm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        
        # Find top confusions (excluding diagonal)
        confusions = []
        for i in range(len(language_names)):
            for j in range(len(language_names)):
                if i != j:
                    confusions.append((cm_norm[i, j], language_names[i], language_names[j]))
        
        confusions.sort(reverse=True)
        for conf_rate, true_lang, pred_lang in confusions[:5]:
            report.append(f"{true_lang} → {pred_lang}: {conf_rate:.3f}")
        report.append("")
    
    report.append("=" * 60)
    
    report_text = "\n".join(report)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
        print(f"Training report saved to: {save_path}")
    
    return report_text


# Example usage and testing
if __name__ == "__main__":
    print("Visualization utilities loaded successfully!")
    
    # Test with dummy data
    dummy_features = np.random.randn(100, 6)
    dummy_labels = np.random.randint(0, 6, 100)
    language_dict = {0: "German", 1: "English", 2: "Spanish", 3: "French", 4: "Dutch", 5: "Portuguese"}
    
    print("Testing PCA visualization...")
    plot_pca_analysis(dummy_features, dummy_labels, language_dict)
    
    print("Testing class distribution...")
    plot_class_distribution(dummy_labels, language_dict)