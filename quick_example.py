#!/usr/bin/env python3
"""
Quick usage example - how to use the model for language detection.

This shows the simplest way to use the trained model for predictions.
"""

import torch
import numpy as np

def load_model():
    """Load the trained model."""
    model = torch.jit.load("saved_models/model.pt", map_location='cpu')
    model.eval()
    return model

def predict_language(model, audio_data):
    """
    Predict language for audio data.
    
    Args:
        model: Trained model
        audio_data: numpy array of shape (40000,) - 5 seconds at 8kHz
    
    Returns:
        dict: Prediction results
    """
    # Language mapping
    languages = {
        0: "German", 1: "English", 2: "Spanish", 
        3: "French", 4: "Dutch", 5: "Portuguese"
    }
    
    # Ensure correct shape and type
    if isinstance(audio_data, np.ndarray):
        audio_data = torch.from_numpy(audio_data.astype(np.float32))
    
    # Normalize (simple normalization - in practice, use training set statistics)
    audio_data = (audio_data - audio_data.mean()) / (audio_data.std() + 1e-8)
    
    # Add batch and channel dimensions: (1, 1, 40000)
    if len(audio_data.shape) == 1:
        audio_data = audio_data.unsqueeze(0).unsqueeze(0)
    elif len(audio_data.shape) == 2:
        audio_data = audio_data.unsqueeze(1)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(audio_data)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_class = torch.argmax(outputs, dim=1)[0].item()
    
    # Format results
    results = {
        'predicted_language': languages[predicted_class],
        'confidence': probabilities[predicted_class].item(),
        'all_probabilities': {
            languages[i]: probabilities[i].item() 
            for i in range(len(languages))
        }
    }
    
    return results

def main():
    """Demonstration of model usage."""
    print(" Quick Language Detection Example")
    print("="*40)
    
    # Load model
    print("Loading model...")
    model = load_model()
    print("✓ Model loaded")
    
    # Load a few test samples for demonstration
    print("Loading test samples...")
    test_data = np.load("data/inputs_test_fp16.npy")
    test_labels = np.load("data/targets_test_int8.npy")
    languages = ["German", "English", "Spanish", "French", "Dutch", "Portuguese"]
    
    # Test on 3 random samples
    np.random.seed(42)
    indices = np.random.choice(len(test_data), 3, replace=False)
    
    print(f"\nTesting on 3 random samples:")
    print("-" * 40)
    
    for i, idx in enumerate(indices):
        # Get sample
        audio_sample = test_data[idx]
        true_language = languages[test_labels[idx]]
        
        # Make prediction
        result = predict_language(model, audio_sample)
        
        # Show results
        is_correct = result['predicted_language'] == true_language
        status = "✓" if is_correct else "✗"
        
        print(f"\nSample {i+1} {status}")
        print(f"True language:      {true_language}")
        print(f"Predicted language: {result['predicted_language']}")
        print(f"Confidence:         {result['confidence']:.3f}")
        
        # Show top 3 predictions
        sorted_probs = sorted(result['all_probabilities'].items(), 
                            key=lambda x: x[1], reverse=True)
        print("Top 3 predictions:")
        for j, (lang, prob) in enumerate(sorted_probs[:3]):
            marker = "→" if j == 0 else " "
            print(f"  {marker} {lang}: {prob:.3f}")
    
    print(f"\n Example completed!")

if __name__ == "__main__":
    main()