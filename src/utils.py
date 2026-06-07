"""
Utility functions for Customer Support Ticket Classification.
Includes logging configuration, directory verification, and NLTK/SpaCy asset checking.
"""

import logging
from typing import Any

def setup_logger(name: str, log_file: str = "outputs/pipeline.log") -> logging.Logger:
    """
    Configures and returns a standard logger that writes to both console and file.
    
    Args:
        name: Name of the logger.
        log_file: Path to write log output.
        
    Returns:
        Configured logger instance.
    """
    pass

def check_nltk_assets() -> None:
    """
    Checks for required NLTK data assets (e.g., stopwords, punkt, wordnet, averaged_perceptron_tagger).
    Downloads them if they are missing.
    """
    pass

def check_spacy_model(model_name: str = "en_core_web_sm") -> None:
    """
    Checks if a SpaCy model is installed locally and downloads it if missing.
    
    Args:
        model_name: Name of the SpaCy model.
    """
    pass

def timer_decorator(func: Any) -> Any:
    """
    A decorator that logs the execution time of functions.
    Useful for tracking preprocessing and training bottleneck points.
    """
    pass
