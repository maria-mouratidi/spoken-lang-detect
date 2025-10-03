"""
Basic tests for the spoken language detection project.

This module contains unit tests for the main components of the project.
"""

import unittest
import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.language_classifier import LanguageClassifier, create_model
from utils.config import Config, get_default_config


class TestLanguageClassifier(unittest.TestCase):
    """Test cases for the LanguageClassifier model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = LanguageClassifier()
        self.input_tensor = torch.randn(1, 1, 40000)  # Batch size 1, 1 channel, 40k samples
    
    def test_model_creation(self):
        """Test model can be created successfully."""
        self.assertIsInstance(self.model, LanguageClassifier)
        self.assertEqual(self.model.num_classes, 6)
        self.assertEqual(self.model.dropout_rate, 0.5)
    
    def test_forward_pass(self):
        """Test model forward pass produces correct output shape."""
        output = self.model(self.input_tensor)
        self.assertEqual(output.shape, (1, 6))  # Batch size 1, 6 classes
    
    def test_model_info(self):
        """Test model info method returns correct information."""
        info = self.model.get_model_info()
        self.assertIn('total_parameters', info)
        self.assertIn('trainable_parameters', info)
        self.assertIn('num_classes', info)
        self.assertEqual(info['num_classes'], 6)
    
    def test_create_model_factory(self):
        """Test the create_model factory function."""
        model = create_model(num_classes=2, dropout_rate=0.3)
        self.assertEqual(model.num_classes, 2)
        self.assertEqual(model.dropout_rate, 0.3)


class TestConfig(unittest.TestCase):
    """Test cases for the configuration system."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = get_default_config()
        self.assertIsInstance(config, Config)
        self.assertEqual(config.model.num_classes, 6)
        self.assertEqual(config.training.num_epochs, 60)
        self.assertEqual(config.data.sampling_rate, 8000)
    
    def test_config_device(self):
        """Test device selection in config."""
        config = get_default_config()
        device = config.get_device()
        self.assertIsInstance(device, torch.device)


class TestDataConstants(unittest.TestCase):
    """Test cases for data-related constants."""
    
    def test_language_constants(self):
        """Test language mapping constants."""
        try:
            from data.dataset import LANGUAGES, LANGUAGE_DICT, LANGUAGE_NAMES
            
            self.assertEqual(len(LANGUAGES), 6)
            self.assertEqual(len(LANGUAGE_DICT), 6)
            self.assertEqual(len(LANGUAGE_NAMES), 6)
            
            # Test mapping consistency
            for i, lang in enumerate(LANGUAGES):
                self.assertEqual(LANGUAGE_DICT[lang], i)
                self.assertIn(i, LANGUAGE_NAMES)
                
        except ImportError:
            self.skipTest("Data module not available")


if __name__ == '__main__':
    # Run tests
    unittest.main()