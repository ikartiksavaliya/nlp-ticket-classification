"""
Model Evaluation and Error Analysis module for Customer Support Ticket Classification.
Calculates performance metrics (F1, precision, recall, accuracy), generates confusion matrices,
visualizes curves, and extracts misclassifications for error profiling.
"""

import os
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from src.utils import setup_logger

# Initialize Logger
logger = setup_logger("evaluate")

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes overall accuracy, precision, recall, and F1-score (macro and weighted).
    
    Args:
        y_true: Ground truth target labels.
        y_pred: Predicted target labels.
        
    Returns:
        Dictionary of metric names to values.
    """
    accuracy = float(accuracy_score(y_true, y_pred))
    precision_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    precision_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    recall_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    recall_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    
    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "precision_weighted": precision_weighted,
        "recall_macro": recall_macro,
        "recall_weighted": recall_weighted,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted
    }

def generate_classification_report(y_true: np.ndarray, y_pred: np.ndarray, target_names: List[str]) -> pd.DataFrame:
    """
    Generates a class-wise classification report as a pandas DataFrame.
    
    Args:
        y_true: Ground truth target labels.
        y_pred: Predicted target labels.
        target_names: Class labels.
        
    Returns:
        DataFrame containing precision, recall, and f1-score for each class.
    """
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
    return pd.DataFrame(report_dict).transpose()

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, target_names: List[str], save_path: str) -> None:
    """
    Generates and saves a confusion matrix heatmap.
    
    Args:
        y_true: Ground truth target labels.
        y_pred: Predicted target labels.
        target_names: Names of classes.
        save_path: Path to save the image (e.g. in reports/figures/).
    """
    cm = confusion_matrix(y_true, y_pred)
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        cbar=True
    )
    plt.title("Normalized Confusion Matrix", fontsize=14, fontweight='bold', pad=15)
    plt.ylabel("True Label", fontsize=12, labelpad=10)
    plt.xlabel("Predicted Label", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_pr_curves(y_true: np.ndarray, y_probs: np.ndarray, target_names: List[str], save_path: str) -> None:
    """
    Plots Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves
    for multi-class settings.
    
    Args:
        y_true: Ground truth target labels (integer labels).
        y_probs: Predicted class probabilities.
        target_names: Names of classes.
        save_path: Path to save the image.
    """
    n_classes = len(target_names)
    # One-hot encode y_true manually to support binary and multi-class correctly
    y_true_binarized = np.zeros((len(y_true), n_classes))
    y_true_binarized[np.arange(len(y_true)), y_true] = 1
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # ROC Curve
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, lw=2, label=f"{target_names[i]} (AUC = {roc_auc:.2f})")
        
    ax1.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontsize=11, labelpad=8)
    ax1.set_ylabel('True Positive Rate', fontsize=11, labelpad=8)
    ax1.set_title('Multi-class ROC Curves (One-vs-Rest)', fontsize=13, fontweight='bold', pad=12)
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Precision-Recall Curve
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_binarized[:, i], y_probs[:, i])
        ap = average_precision_score(y_true_binarized[:, i], y_probs[:, i])
        ax2.plot(recall, precision, lw=2, label=f"{target_names[i]} (AP = {ap:.2f})")
        
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall', fontsize=11, labelpad=8)
    ax2.set_ylabel('Precision', fontsize=11, labelpad=8)
    ax2.set_title('Multi-class Precision-Recall Curves (One-vs-Rest)', fontsize=13, fontweight='bold', pad=12)
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def perform_error_analysis(df: pd.DataFrame, text_col: str, y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray, target_names: List[str]) -> pd.DataFrame:
    """
    Extracts and profiles misclassified instances:
    - Lists ticket text, true label, predicted label, and model confidence (probability).
    - Returns instances sorted by highest confidence error (where model was most confidently wrong).
    - Crucial for identifying semantic overlap in support ticket categories.
    
    Args:
        df: The input dataframe containing original tickets.
        text_col: The raw text column name.
        y_true: Ground truth label indices.
        y_pred: Predicted label indices.
        y_probs: Prediction probability matrix.
        target_names: Mapping list of category names.
        
    Returns:
        DataFrame containing error log with text, predictions, targets, and confidences.
    """
    misclassified_mask = y_true != y_pred
    misclassified_indices = np.where(misclassified_mask)[0]
    
    error_texts = df[text_col].iloc[misclassified_indices].values
    error_true = [target_names[val] for val in y_true[misclassified_indices]]
    error_pred = [target_names[val] for val in y_pred[misclassified_indices]]
    
    # Confidence is the probability of the predicted category
    error_conf = y_probs[misclassified_indices, y_pred[misclassified_indices]]
    # True category probability is the probability assigned to the correct category
    error_true_prob = y_probs[misclassified_indices, y_true[misclassified_indices]]
    
    error_df = pd.DataFrame({
        "text": error_texts,
        "true_category": error_true,
        "predicted_category": error_pred,
        "prediction_confidence": error_conf,
        "true_category_probability": error_true_prob
    })
    
    error_df = error_df.sort_values(by="prediction_confidence", ascending=False).reset_index(drop=True)
    return error_df

def run_evaluation_pipeline() -> None:
    """
    Orchestrator for evaluating the best model on test and validation sets.
    - Loads pre-fitted artifacts (model, vectorizer, label encoder).
    - Transforms val and test sets.
    - Generates predictions and prediction probabilities.
    - Calculates and prints performance metrics.
    - Saves confusion matrices and ROC/PR curves to reports/figures/.
    - Outputs error analysis CSV to outputs/.
    """
    from src.config import (
        MODEL_PATH,
        VECTORIZER_PATH,
        LABEL_ENCODER_PATH,
        PROCESSED_TEST_PATH,
        TEXT_COL,
        TARGET_COL,
        PROJECT_ROOT
    )
    
    logger.info("--- Starting Model Evaluation and Error Profiling Pipeline ---")
    
    # Check if files exist
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH) or not os.path.exists(LABEL_ENCODER_PATH):
        logger.error("Model artifacts not found. Please run src/train.py first to train the model.")
        return
        
    logger.info("Loading model, vectorizer, and label encoder...")
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)
    
    target_names = list(le.classes_)
    
    # Load test data
    if not os.path.exists(PROCESSED_TEST_PATH):
        logger.error(f"Preprocessed test data not found at {PROCESSED_TEST_PATH}.")
        return
        
    logger.info("Loading test dataset...")
    test_df = pd.read_csv(PROCESSED_TEST_PATH)
    
    # Fillna on cleaned text column
    cleaned_col = f"cleaned_{TEXT_COL}"
    if cleaned_col not in test_df.columns:
        logger.error(f"Preprocessed text column '{cleaned_col}' not found in test.csv.")
        return
        
    test_df[cleaned_col] = test_df[cleaned_col].fillna("").astype(str)
    test_texts = test_df[cleaned_col].tolist()
    
    # Transform texts to numeric representation
    logger.info("Vectorizing test texts...")
    x_test = vectorizer.transform(test_texts).astype(np.float32)
    
    # Get true target labels encoded
    y_true = le.transform(test_df[TARGET_COL])
    
    # Generate predictions
    logger.info("Generating predictions and class probabilities...")
    y_pred = model.predict(x_test)
    y_probs = model.predict_proba(x_test)
    
    # Convert to numpy in case of cuML arrays
    if hasattr(y_pred, "get"):
        y_pred = y_pred.get()
    elif hasattr(y_pred, "to_numpy"):
        y_pred = y_pred.to_numpy()
        
    if hasattr(y_probs, "get"):
        y_probs = y_probs.get()
    elif hasattr(y_probs, "to_numpy"):
        y_probs = y_probs.to_numpy()
        
    # Cast predictions to int
    y_pred = y_pred.astype(np.int32)
    
    # Compute overall metrics
    logger.info("Computing metrics...")
    metrics = compute_metrics(y_true, y_pred)
    
    logger.info("=== Performance Metrics on Holdout Test Set ===")
    logger.info(f"Accuracy:          {metrics['accuracy']:.4f}")
    logger.info(f"Precision (Macro): {metrics['precision_macro']:.4f}")
    logger.info(f"Recall (Macro):    {metrics['recall_macro']:.4f}")
    logger.info(f"F1-score (Macro):  {metrics['f1_macro']:.4f}")
    logger.info(f"F1-score (Weighted): {metrics['f1_weighted']:.4f}")
    
    # Save classification report
    report_df = generate_classification_report(y_true, y_pred, target_names)
    reports_dir = PROJECT_ROOT / "reports"
    os.makedirs(reports_dir, exist_ok=True)
    report_df.to_csv(reports_dir / "classification_report.csv")
    logger.info(f"Classification report saved to {reports_dir / 'classification_report.csv'}")
    
    # Plot confusion matrix heatmap
    fig_dir = reports_dir / "figures"
    os.makedirs(fig_dir, exist_ok=True)
    cm_save_path = str(fig_dir / "confusion_matrix.png")
    logger.info("Plotting and saving confusion matrix...")
    plot_confusion_matrix(y_true, y_pred, target_names, cm_save_path)
    
    # Plot ROC / PR curves
    curves_save_path = str(fig_dir / "roc_pr_curves.png")
    logger.info("Plotting and saving ROC and Precision-Recall curves...")
    plot_roc_pr_curves(y_true, y_probs, target_names, curves_save_path)
    
    # Perform error analysis
    logger.info("Performing error analysis...")
    text_col_for_errors = TEXT_COL if TEXT_COL in test_df.columns else cleaned_col
    error_df = perform_error_analysis(test_df, text_col_for_errors, y_true, y_pred, y_probs, target_names)
    
    outputs_dir = PROJECT_ROOT / "outputs"
    os.makedirs(outputs_dir, exist_ok=True)
    error_log_path = outputs_dir / "misclassified_tickets.csv"
    error_df.to_csv(error_log_path, index=False)
    logger.info(f"Error analysis log saved to {error_log_path} ({len(error_df)} misclassified tickets found).")
    
    logger.info("--- Model Evaluation Completed Successfully ---")

if __name__ == "__main__":
    run_evaluation_pipeline()
