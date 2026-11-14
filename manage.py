#!/usr/bin/env python
"""
Django management script for Alzheimer's Disease Prediction System
"""
import os
import sys
import warnings
from pathlib import Path

# Suppress deprecation warnings in development
warnings.filterwarnings('ignore', category=DeprecationWarning)

def main():
    """Run administrative tasks."""
    # Set default settings module
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'alzheimers_prediction.settings')
    
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    
    # Add custom management commands directory to path
    sys.path.append(str(Path(__file__).parent / 'alzheimers_prediction' / 'management'))
    
    # Execute command
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    # Set higher recursion limit for deep learning models
    sys.setrecursionlimit(10000)
    
    # Set environment variables for better performance
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')  # Suppress TensorFlow warnings
    os.environ.setdefault('PYTHONHASHSEED', '42')  # For reproducibility
    
    main()