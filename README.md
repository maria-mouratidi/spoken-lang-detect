# Spoken Language Detection

A deep learning project for classifying spoken languages from audio clips using PyTorch. This model can distinguish between German, English, Spanish, French, Dutch, and Portuguese spoken languages.

## Project Overview

This project implements a Convolutional Neural Network (CNN) to classify audio clips of spoken languages. The model processes 5-second audio clips sampled at 8kHz and predicts which of six European languages is being spoken.

### Supported Languages
- German (de)
- English (en) 
- Spanish (es)
- French (fr)
- Dutch (nl)
- Portuguese (pt)

## Architecture

The model uses a CNN architecture with:
- 4 convolutional layers with increasing filters (32 → 64 → 128 → 256)
- MaxPooling layers for dimensionality reduction
- Adaptive average pooling for handling variable input lengths
- Fully connected layers with dropout for classification
- Final output: 6 classes (one per language)

**Model Size:** ~100K parameters (efficient and lightweight)

## Performance

- **Test Accuracy:** ~50-60% on 6-class classification
- **Binary Classification:** 80%+ accuracy (English vs Spanish)
- **Training:** 60 epochs with Adam optimizer

## 🚀 Quick Start

### Prerequisites

```bash
python >= 3.8
torch >= 1.9.0
numpy
matplotlib
seaborn
pandas
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/spoken-lang-detect.git
cd spoken-lang-detect
```

2. Install dependencies:
```bash
pip install -r requirements.txt
``

### Usage

#### Training a new model:
```bash
python scripts/train.py
```

#### Running inference:
```bash
python scripts/predict.py --audio_file path/to/audio.wav
```

#### Evaluating the model:
```bash
python scripts/evaluate.py
```

## Project Structure

```
spoken-lang-detect/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   └── language_classifier.py    # CNN model architecture
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py               # Data loading and preprocessing
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py               # Configuration management
│   │   └── visualization.py        # Plotting and analysis tools
├── scripts/
│   ├── train.py                    # Training script
│   ├── evaluate.py                 # Evaluation script
│   └── predict.py                  # Inference script
├── data/                           # Dataset files
├── saved_models/                   # Trained model weights
├── tests/                         # Unit tests
├── requirements.txt
├── setup.py
└── README.md
```

## Technical Details

### Data Processing
- **Input Format:** 40,000 amplitude measurements per 5-second clip
- **Sampling Rate:** 8kHz
- **Normalization:** Z-score normalization applied to prevent overfitting
- **Data Augmentation:** Available for improved generalization

### Model Architecture Details
```python
Input: [batch_size, 1, 40000]
├── Conv1d(1→32, kernel=50, stride=5) + ReLU + MaxPool
├── Conv1d(32→64, kernel=3, stride=1) + ReLU + MaxPool  
├── Conv1d(64→128, kernel=3, stride=1) + ReLU + MaxPool
├── Conv1d(128→256, kernel=3, stride=1) + ReLU + MaxPool
├── AdaptiveAvgPool1d(1)
├── Flatten + Dropout(0.5)
├── Linear(256→128→64→6)
└── Output: [batch_size, 6]
```

### Training Configuration
- **Loss Function:** CrossEntropyLoss
- **Optimizer:** Adam (lr=0.001)
- **Batch Size:** 1 (due to memory constraints)
- **Epochs:** 60
- **Device:** CUDA if available, CPU otherwise

## Results & Analysis

### Confusion Matrix
The model shows good performance distinguishing between Germanic and Romance language families, with some expected confusion between linguistically similar languages.

### PCA Visualization
The model's output space demonstrates logical clustering of languages by linguistic families when visualized using Principal Component Analysis.

## Limitations & Bias

**Data Source:** The dataset was mined from YouTube videos using language-specific searches.

**Potential Biases:**
- Geographic/accent variations not well represented
- Speaker demographics may be skewed
- Audio quality variations from YouTube compression
- Topic/content bias from search methodology

**Use Case Limitations:**
- Model may perform poorly on accents not represented in training data
- Performance may degrade on non-native speakers
- Audio quality must be similar to training data (8kHz, clear speech)

## Authors

- Group 21 - Introduction to Deep Learning Course 2023

## 🙏 Acknowledgments

- Course instructors: Juan Sebastian Olier Jauregui, Dimitar Shterionov, and Cascha van Wanrooij