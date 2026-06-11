"""
Utility functions for Customer Support Ticket Classification.
Includes logging configuration, directory verification, and NLTK/SpaCy asset checking.
"""

import os
import logging
import time
from functools import wraps
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
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger
        
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Ensure directory exists
    log_path = os.path.abspath(log_file)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    # File handler
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def check_nltk_assets() -> None:
    """
    Checks for required NLTK data assets (e.g., stopwords, punkt, wordnet).
    Downloads them if they are missing.
    """
    import nltk
    required_assets = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']
    for asset in required_assets:
        try:
            nltk.download(asset, quiet=True)
        except Exception as e:
            print(f"Error downloading NLTK asset {asset}: {e}")

def check_spacy_model(model_name: str = "en_core_web_sm") -> None:
    """
    Checks if a SpaCy model is installed locally and downloads it if missing.
    
    Args:
        model_name: Name of the SpaCy model.
    """
    import spacy
    if not spacy.util.is_package(model_name):
        print(f"Downloading SpaCy model: {model_name}...")
        spacy.cli.download(model_name)

def timer_decorator(func: Any) -> Any:
    """
    A decorator that logs the execution time of functions.
    Useful for tracking preprocessing and training bottleneck points.
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        logger = logging.getLogger("timer")
        if not logger.handlers:
            logger = setup_logger("timer")
        logger.info(f"Function '{func.__name__}' completed in {elapsed_time:.4f} seconds")
        return result
    return wrapper
