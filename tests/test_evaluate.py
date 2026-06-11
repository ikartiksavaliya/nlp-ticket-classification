"""
Unit tests for the evaluation and error analysis module in src/evaluate.py.
"""

import os
import pytest
import numpy as np
import pandas as pd
from src.evaluate import (
    compute_metrics,
    generate_classification_report,
    plot_confusion_matrix,
    plot_roc_pr_curves,
    perform_error_analysis
)

def test_compute_metrics():
    """Test overall metric computation for accuracy, precision, recall, and f1-scores."""
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 1, 0, 1, 2]) # 2 mismatches: index 4 (1 vs 2) and index 5 (2 vs 1)
    
    metrics = compute_metrics(y_true, y_pred)
    
    assert "accuracy" in metrics
    assert "precision_macro" in metrics
    assert "recall_macro" in metrics
    assert "f1_macro" in metrics
    assert "f1_weighted" in metrics
    
    # Accuracy is 7/9
    assert pytest.approx(metrics["accuracy"], 0.01) == 7.0 / 9.0

def test_generate_classification_report():
    """Verify that the classification report is returned as a valid pandas DataFrame."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    target_names = ["class_0", "class_1"]
    
    report_df = generate_classification_report(y_true, y_pred, target_names)
    
    assert isinstance(report_df, pd.DataFrame)
    assert "precision" in report_df.columns
    assert "recall" in report_df.columns
    assert "f1-score" in report_df.columns
    assert "class_0" in report_df.index
    assert "class_1" in report_df.index

def test_plot_confusion_matrix(tmp_path):
    """Test generating and saving a confusion matrix image."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    target_names = ["class_0", "class_1"]
    save_path = tmp_path / "cm.png"
    
    # Should save without raising an error
    plot_confusion_matrix(y_true, y_pred, target_names, str(save_path))
    assert os.path.exists(save_path)

def test_plot_roc_pr_curves(tmp_path):
    """Test generating and saving multi-class ROC and PR curves."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_probs = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.2, 0.2, 0.6],
        [0.7, 0.2, 0.1],
        [0.1, 0.7, 0.2],
        [0.1, 0.1, 0.8]
    ])
    target_names = ["class_0", "class_1", "class_2"]
    save_path = tmp_path / "curves.png"
    
    plot_roc_pr_curves(y_true, y_probs, target_names, str(save_path))
    assert os.path.exists(save_path)

def test_perform_error_analysis():
    """Verify that error profiling extracts and correctly sorts misclassifications by prediction confidence."""
    df = pd.DataFrame({
        "ticket_id": [101, 102, 103, 104],
        "issue_description": ["issue one", "issue two", "issue three", "issue four"]
    })
    y_true = np.array([0, 1, 2, 0])
    y_pred = np.array([0, 2, 1, 0]) # Errors at indices 1 and 2
    # Mock probability matrix
    y_probs = np.array([
        [0.9, 0.05, 0.05], # True pred
        [0.1, 0.2, 0.7],   # Error (pred 2 with 0.7 prob, true 1 with 0.2 prob)
        [0.05, 0.8, 0.15], # Error (pred 1 with 0.8 prob, true 2 with 0.15 prob)
        [0.95, 0.02, 0.03] # True pred
    ])
    target_names = ["class_0", "class_1", "class_2"]
    
    error_df = perform_error_analysis(df, "issue_description", y_true, y_pred, y_probs, target_names)
    
    assert isinstance(error_df, pd.DataFrame)
    assert len(error_df) == 2
    # Columns check
    assert "text" in error_df.columns
    assert "true_category" in error_df.columns
    assert "predicted_category" in error_df.columns
    assert "prediction_confidence" in error_df.columns
    assert "true_category_probability" in error_df.columns
    
    # Index 2 error has higher confidence (0.8) than index 1 error (0.7), so it should be first
    assert error_df.iloc[0]["text"] == "issue three"
    assert error_df.iloc[0]["prediction_confidence"] == 0.8
    assert error_df.iloc[1]["text"] == "issue two"
    assert error_df.iloc[1]["prediction_confidence"] == 0.7
