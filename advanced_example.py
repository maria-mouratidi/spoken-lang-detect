#!/usr/bin/env python3
"""
Advanced example demonstrating more features of the spoken language detection model.

This script shows:
1. Batch processing of multiple samples
2. Confidence threshold analysis
3. Confusion analysis
4. Performance metrics
5. Language distribution in predictions
"""

import sys
import os
import numpy as np
import torch
from collections import Counter, defaultdict

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from data.dataset import LANGUAGE_NAMES, LANGUAGES
except ImportError:
    # Fallback definitions
    LANGUAGE_NAMES = {
        0: "German", 1: "English", 2: "Spanish", 
        3: "French", 4: "Dutch", 5: "Portuguese"
    }
    LANGUAGES = ["de", "en", "es", "fr", "nl", "pt"]

def load_model_and_data():
    """Load the model and test data."""
    print(" Loading model and data...")
    
    # Load model
    model_path = "saved_models/model.pt"
    model = torch.jit.load(model_path, map_location='cpu')
    model.eval()
    print(f"✓ Model loaded from {model_path}")
    
    # Load test data
    X_test = np.load("data/inputs_test_fp16.npy").astype(np.float32)
    y_test = np.load("data/targets_test_int8.npy")
    
    # Normalize data
    mean, std = np.mean(X_test), np.std(X_test)
    X_test = (X_test - mean) / (std + 1e-8)
    
    X_test = torch.from_numpy(X_test)
    y_test = torch.from_numpy(y_test)
    
    print(f"✓ Test data loaded: {len(X_test)} samples")
    return model, X_test, y_test

def batch_predict(model, X_test, batch_size=32):
    """Make predictions on the entire test set in batches."""
    print(f"🔮 Making predictions (batch size: {batch_size})...")
    
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch = X_test[i:i+batch_size]
            
            # Add channel dimension if needed
            if len(batch.shape) == 2:
                batch = batch.unsqueeze(1)
            
            # Forward pass
            outputs = model(batch)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.append(predictions)
            all_probabilities.append(probabilities)
    
    # Concatenate all results
    predictions = torch.cat(all_predictions)
    probabilities = torch.cat(all_probabilities)
    
    print(f"✓ Predictions completed for {len(predictions)} samples")
    return predictions, probabilities

def analyze_performance(predictions, probabilities, true_labels):
    """Analyze model performance in detail."""
    print("\n PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Overall accuracy
    correct = (predictions == true_labels).sum().item()
    total = len(predictions)
    accuracy = correct / total
    
    print(f"Overall Accuracy: {accuracy:.3f} ({correct}/{total})")
    
    # Per-class accuracy
    print("\nPer-Language Accuracy:")
    for lang_id in range(6):
        mask = true_labels == lang_id
        if mask.sum() > 0:
            lang_correct = ((predictions == true_labels) & mask).sum().item()
            lang_total = mask.sum().item()
            lang_accuracy = lang_correct / lang_total
            print(f"  {LANGUAGE_NAMES[lang_id]:12}: {lang_accuracy:.3f} ({lang_correct:3d}/{lang_total:3d})")
    
    # Confidence analysis
    confidences = torch.max(probabilities, dim=1)[0]
    print(f"\nConfidence Statistics:")
    print(f"  Mean confidence: {torch.mean(confidences):.3f}")
    print(f"  Median confidence: {torch.median(confidences):.3f}")
    print(f"  Min confidence: {torch.min(confidences):.3f}")
    print(f"  Max confidence: {torch.max(confidences):.3f}")
    
    # High confidence accuracy
    high_conf_mask = confidences > 0.8
    if high_conf_mask.sum() > 0:
        high_conf_correct = ((predictions == true_labels) & high_conf_mask).sum().item()
        high_conf_total = high_conf_mask.sum().item()
        high_conf_accuracy = high_conf_correct / high_conf_total
        print(f"  High confidence (>0.8) accuracy: {high_conf_accuracy:.3f} ({high_conf_correct}/{high_conf_total})")

def confusion_analysis(predictions, true_labels):
    """Analyze confusion patterns between languages."""
    print("\n CONFUSION ANALYSIS")
    print("="*60)
    
    # Create confusion matrix
    confusion_counts = defaultdict(int)
    
    for pred, true in zip(predictions, true_labels):
        pred_lang = LANGUAGE_NAMES[pred.item()]
        true_lang = LANGUAGE_NAMES[true.item()]
        
        if pred != true:
            confusion_counts[(true_lang, pred_lang)] += 1
    
    if confusion_counts:
        print("Most common confusions:")
        sorted_confusions = sorted(confusion_counts.items(), key=lambda x: x[1], reverse=True)
        
        for i, ((true_lang, pred_lang), count) in enumerate(sorted_confusions[:10]):
            print(f"  {i+1:2d}. {true_lang:12} → {pred_lang:12}: {count:3d} times")
    else:
        print("No confusions found (perfect accuracy)!")

def language_distribution_analysis(predictions, true_labels):
    """Analyze prediction distribution."""
    print("\n LANGUAGE DISTRIBUTION")
    print("="*60)
    
    # True distribution
    true_counts = Counter(true_labels.numpy())
    pred_counts = Counter(predictions.numpy())
    
    print("Distribution comparison:")
    print(f"{'Language':12} | {'True':>6} | {'Predicted':>9} | {'Difference':>10}")
    print("-" * 50)
    
    for lang_id in range(6):
        true_count = true_counts.get(lang_id, 0)
        pred_count = pred_counts.get(lang_id, 0)
        diff = pred_count - true_count
        diff_str = f"{diff:+d}" if diff != 0 else "0"
        
        print(f"{LANGUAGE_NAMES[lang_id]:12} | {true_count:6d} | {pred_count:9d} | {diff_str:>10}")

def show_sample_predictions(model, X_test, y_test, num_samples=3):
    """Show detailed predictions for a few samples."""
    print(f"\n SAMPLE PREDICTIONS (showing {num_samples} examples)")
    print("="*80)
    
    # Select random samples
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            sample = X_test[idx:idx+1]
            true_label = y_test[idx].item()
            
            # Add channel dimension
            if len(sample.shape) == 2:
                sample = sample.unsqueeze(1)
            
            # Predict
            output = model(sample)
            probabilities = torch.softmax(output, dim=1)[0]
            predicted_class = torch.argmax(output, dim=1)[0].item()
            
            # Show results
            is_correct = predicted_class == true_label
            status = "✓ CORRECT" if is_correct else "✗ WRONG"
            
            print(f"\nSample #{idx:4d} | {status}")
            print(f"True Language: {LANGUAGE_NAMES[true_label]}")
            print(f"Predicted:     {LANGUAGE_NAMES[predicted_class]} (confidence: {probabilities[predicted_class]:.3f})")
            
            # Show all probabilities
            print("All language probabilities:")
            for lang_id in range(6):
                prob = probabilities[lang_id].item()
                marker = "→" if lang_id == predicted_class else " "
                print(f"  {marker} {LANGUAGE_NAMES[lang_id]:12}: {prob:.3f}")

def main():
    """Main function."""
    print("Advanced Spoken Language Detection Analysis")
    print("="*60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Load model and data
    model, X_test, y_test = load_model_and_data()
    
    # Make predictions on all test data
    predictions, probabilities = batch_predict(model, X_test, batch_size=64)
    
    # Analyze performance
    analyze_performance(predictions, probabilities, y_test)
    
    # Analyze confusions
    confusion_analysis(predictions, y_test)
    
    # Analyze distributions
    language_distribution_analysis(predictions, y_test)
    
    # Show sample predictions
    show_sample_predictions(model, X_test, y_test, num_samples=3)
    
    print("\nAnalysis completed!")

if __name__ == "__main__":
    main()