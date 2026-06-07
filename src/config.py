"""
Configuration module for Customer Support Ticket Classification.
Contains path definitions, random seeds, split ratios, and model hyperparameter grids.
"""

import os
from pathlib import Path

# Project Root Directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data Paths
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "customer_support_tickets_200k.csv"
PROCESSED_TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
PROCESSED_VAL_PATH = PROJECT_ROOT / "data" / "processed" / "val.csv"
PROCESSED_TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"

# Model Paths
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "best_model.joblib"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.joblib"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.joblib"

# Pipeline Configurations
TARGET_COL = "category"
TEXT_COL = "issue_description"
SPACY_MODEL = "en_core_web_sm"

# Pipeline Parameters
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15


# Text Representation Configuration
MAX_FEATURES = 10000
NGRAM_RANGE = (1, 2)
MIN_DF = 2
MAX_DF = 0.95

# Model Training Parameters
HYPERPARAMETER_GRIDS = {
    "naive_bayes": {
        "alpha": [0.01, 0.1, 1.0, 10.0]
    },
    "logistic_regression": {
        "C": [0.1, 1.0, 10.0],
        "penalty": ["l2"],
        "solver": ["lbfgs"],
        "max_iter": [1000]
    },
    "svm": {
        "C": [0.1, 1.0, 10.0],
        "kernel": ["linear", "rbf"]
    },
    "xgboost": {
        "n_estimators": [100, 200],
        "max_depth": [4, 6],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8]
    }
}
