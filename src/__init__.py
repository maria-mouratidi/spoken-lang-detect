"""Main package initialization."""

__version__ = "1.0.0"
__author__ = "Group 21"
__description__ = "Spoken Language Detection using Deep Learning"

# Import main classes for easy access
from .models import LanguageClassifier, create_model
from .data import (
    LanguageDataset, 
    create_datasets, 
    create_data_loaders,
    LANGUAGES,
    LANGUAGE_NAMES
)
from .utils import Config, get_default_config

__all__ = [
    'LanguageClassifier',
    'create_model',
    'LanguageDataset',
    'create_datasets',
    'create_data_loaders',
    'Config',
    'get_default_config',
    'LANGUAGES',
    'LANGUAGE_NAMES'
]