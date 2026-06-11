"""
Model Training module for Customer Support Ticket Classification.
Orchestrates data splitting, vectorization fitting, hyperparameter tuning,
model training, and model serialization using GPU acceleration where available.
"""

from typing import Dict, List, Tuple, Any
import os
import logging
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import joblib
from tqdm import tqdm

from src.utils import setup_logger, timer_decorator

# Initialize Logger
logger = setup_logger("train")

# Fallback-safe Imports for cuML (GPU) and XGBoost (GPU)
try:
    from cuml.naive_bayes import MultinomialNB as GPUMultinomialNB
    from cuml.linear_model import LogisticRegression as GPULogisticRegression
    from cuml.svm import SVC as GPUSVC
    HAS_CUML = True
    logger.info("RAPIDS cuML is available. GPU acceleration will be used for modeling.")
except ImportError:
    HAS_CUML = False
    logger.warning("RAPIDS cuML is not found. Modeling will fall back to CPU-based scikit-learn.")

try:
    import xgboost as xgb
    HAS_XGB = True
    logger.info("XGBoost is available.")
except ImportError:
    HAS_XGB = False
    logger.warning("XGBoost is not found.")

def create_model(model_type: str, params: Dict[str, Any], force_cpu: bool = False, probability: bool = False) -> Any:
    """
    Instantiates the appropriate estimator model on GPU (cuML/XGBoost) or CPU (scikit-learn).
    """
    use_gpu = HAS_CUML and not force_cpu
    
    if model_type == "naive_bayes":
        if use_gpu:
            return GPUMultinomialNB(**params)
        else:
            from sklearn.naive_bayes import MultinomialNB
            return MultinomialNB(**params)
            
    elif model_type == "logistic_regression":
        if use_gpu:
            return GPULogisticRegression(**params)
        else:
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(**params)
            
    elif model_type == "svm":
        if use_gpu:
            return GPUSVC(**params, probability=probability)
        else:
            from sklearn.svm import SVC
            return SVC(**params, probability=probability)
            
    elif model_type == "xgboost":
        from xgboost import XGBClassifier
        from src.config import RANDOM_STATE
        device_param = 'cuda' if (HAS_XGB and not force_cpu) else 'cpu'
        return XGBClassifier(**params, device=device_param, random_state=RANDOM_STATE, eval_metric='mlogloss')
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def train_and_fit_model(model_type: str, params: Dict[str, Any], x_train: csr_matrix, y_train: np.ndarray, probability: bool = True) -> Tuple[Any, bool]:
    """
    Fits the specified estimator on the training data, attempting to run on GPU first,
    and falling back to CPU if an error occurs.
    
    Returns:
        A tuple of (fitted_model, was_fallback_bool).
    """
    x_train_fit = x_train.astype(np.float32)
    # cuML NaiveBayes requires integer labels, whereas others accept float32
    if model_type == "naive_bayes":
        y_train_fit = y_train.astype(np.int32)
    else:
        y_train_fit = y_train.astype(np.float32)
        
    # Attempt GPU Training
    try:
        model = create_model(model_type, params, force_cpu=False, probability=probability)
        model.fit(x_train_fit, y_train_fit)
        return model, False
    except Exception as e:
        logger.warning(f"GPU training/fitting failed for {model_type} due to: {e}. Falling back to CPU...")
        # Fallback to CPU Training
        model = create_model(model_type, params, force_cpu=True, probability=probability)
        model.fit(x_train_fit, y_train_fit)
        return model, True

@timer_decorator
def load_and_split_data(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads raw CSV data and performs a clean Train/Validation/Test split
    using a fixed seed to guarantee reproducibility.
    Avoids data leakage by keeping validation/test partitions completely unseen.
    """
    from src.config import (
        PROCESSED_TRAIN_PATH,
        PROCESSED_VAL_PATH,
        PROCESSED_TEST_PATH,
        SAMPLE_SIZE,
        RANDOM_STATE,
        TEST_SIZE,
        VAL_SIZE,
        TEXT_COL,
        TARGET_COL
    )
    from src.preprocessing import preprocess_dataframe
    from sklearn.model_selection import train_test_split
    
    # Cache Check
    if os.path.exists(PROCESSED_TRAIN_PATH) and os.path.exists(PROCESSED_VAL_PATH) and os.path.exists(PROCESSED_TEST_PATH):
        logger.info("Loading preprocessed Train/Val/Test splits from cache...")
        train_df = pd.read_csv(PROCESSED_TRAIN_PATH)
        val_df = pd.read_csv(PROCESSED_VAL_PATH)
        test_df = pd.read_csv(PROCESSED_TEST_PATH)
        return train_df, val_df, test_df
        
    logger.info(f"Loading raw dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Drop rows where target or text column is null
    df = df.dropna(subset=[TARGET_COL, TEXT_COL]).copy()
    
    # Stratified Sampling to SAMPLE_SIZE
    if len(df) > SAMPLE_SIZE:
        logger.info(f"Sub-sampling dataset to {SAMPLE_SIZE} rows stratified by '{TARGET_COL}'...")
        _, df = train_test_split(
            df,
            test_size=SAMPLE_SIZE,
            random_state=RANDOM_STATE,
            stratify=df[TARGET_COL]
        )
        
    # Split into Train / Temp
    temp_size = TEST_SIZE + VAL_SIZE
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_size,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COL]
    )
    
    # Split Temp into Val / Test
    val_ratio = VAL_SIZE / temp_size
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_ratio),
        random_state=RANDOM_STATE,
        stratify=temp_df[TARGET_COL]
    )
    
    logger.info(f"Preprocessing splits (Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)})...")
    logger.info("Preprocessing train set:")
    train_df = preprocess_dataframe(train_df, TEXT_COL, TARGET_COL)
    
    logger.info("Preprocessing validation set:")
    val_df = preprocess_dataframe(val_df, TEXT_COL, TARGET_COL)
    
    logger.info("Preprocessing test set:")
    test_df = preprocess_dataframe(test_df, TEXT_COL, TARGET_COL)
    
    # Save cache
    os.makedirs(os.path.dirname(PROCESSED_TRAIN_PATH), exist_ok=True)
    train_df.to_csv(PROCESSED_TRAIN_PATH, index=False)
    val_df.to_csv(PROCESSED_VAL_PATH, index=False)
    test_df.to_csv(PROCESSED_TEST_PATH, index=False)
    logger.info(f"Preprocessed splits saved cache to {os.path.dirname(PROCESSED_TRAIN_PATH)}.")
    
    return train_df, val_df, test_df

def train_baseline_model(x_train: csr_matrix, y_train: np.ndarray, model_type: str = "naive_bayes") -> Any:
    """
    Trains a baseline model (e.g. Multinomial Naive Bayes) with default parameters.
    """
    logger.info(f"Training baseline model: {model_type}...")
    params = {"alpha": 1.0} if model_type == "naive_bayes" else {}
    model,runs_on_cpu = train_and_fit_model(model_type, params, x_train, y_train)
    return model

@timer_decorator
def run_grid_search(x_train: csr_matrix, y_train: np.ndarray, model_type: str) -> Tuple[Any, Dict[str, Any]]:
    """
    Performs K-fold cross-validation and hyperparameter optimization
    on the training set using custom tqdm grid search.
    """
    from src.config import HYPERPARAMETER_GRIDS, RANDOM_STATE
    from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_val_score
    
    grid = HYPERPARAMETER_GRIDS.get(model_type, {})
    param_list = list(ParameterGrid(grid))
    
    logger.info(f"Running custom Grid Search for '{model_type}' over {len(param_list)} combinations...")
    
    best_score = -1.0
    best_params = {}
    best_estimator = None
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    
    # Cast to types compatible with cuML
    x_train_fit = x_train.astype(np.float32)
    if model_type == "naive_bayes":
        y_train_fit = y_train.astype(np.int32)
    else:
        y_train_fit = y_train.astype(np.float32)
        
    pbar = tqdm(param_list, desc=f"GridSearch: {model_type}", leave=True)
    for params in pbar:
        try:
            model = create_model(model_type, params, force_cpu=False, probability=False)
            scores = cross_val_score(model, x_train_fit, y_train_fit, cv=cv, scoring='f1_macro')
            mean_score = float(scores.mean())
        except Exception as gpu_e:
            logger.warning(f"GPU Cross-Validation failed for {model_type} params {params} due to: {gpu_e}. Retrying on CPU...")
            try:
                model = create_model(model_type, params, force_cpu=True, probability=False)
                scores = cross_val_score(model, x_train_fit, y_train_fit, cv=cv, scoring='f1_macro')
                mean_score = float(scores.mean())
            except Exception as cpu_e:
                logger.error(f"CPU Fallback Cross-Validation also failed for {model_type} params {params}: {cpu_e}")
                continue
                
        pbar.set_postfix(f1_macro=f"{mean_score:.4f}")
        
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
            
    # Final Fit on entire training split using the best parameters
    if best_params:
        logger.info(f"Fitting final best {model_type} model on all training data with parameters {best_params}...")
        try:
            best_estimator = create_model(model_type, best_params, force_cpu=False, probability=True)
            best_estimator.fit(x_train_fit, y_train_fit)
        except Exception as e:
            logger.warning(f"GPU final fit failed: {e}. Re-fitting on CPU...")
            best_estimator = create_model(model_type, best_params, force_cpu=True, probability=True)
            best_estimator.fit(x_train_fit, y_train_fit)
            
    return best_estimator, best_params

def save_model_artifacts(model: Any, vectorizer: Any, label_encoder: Any) -> None:
    """
    Serializes and saves the trained model, vectorizer, and label encoder to disk.
    """
    from src.config import MODEL_DIR, MODEL_PATH, VECTORIZER_PATH, LABEL_ENCODER_PATH
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    logger.info(f"Saving best champion model to {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)
    
    logger.info(f"Saving vectorizer to {VECTORIZER_PATH}...")
    joblib.dump(vectorizer, VECTORIZER_PATH)
    
    logger.info(f"Saving label encoder to {LABEL_ENCODER_PATH}...")
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    
    logger.info("Model artifacts saved successfully.")

def train_pipeline() -> None:
    """
    Full pipeline orchestrator.
    """
    from src.config import RAW_DATA_PATH, TEXT_COL, TARGET_COL
    from src.feature_engineering import fit_tfidf_vectorizer, transform_texts
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import f1_score
    
    logger.info("--- Starting Model Development Pipeline ---")
    
    # 1. Load, Split and Preprocess (with Cache support)
    train_df, val_df, test_df = load_and_split_data(str(RAW_DATA_PATH))
    
    # Align and fill missing values
    train_df[f"cleaned_{TEXT_COL}"] = train_df[f"cleaned_{TEXT_COL}"].fillna("").astype(str)
    val_df[f"cleaned_{TEXT_COL}"] = val_df[f"cleaned_{TEXT_COL}"].fillna("").astype(str)
    
    train_texts = train_df[f"cleaned_{TEXT_COL}"].tolist()
    val_texts = val_df[f"cleaned_{TEXT_COL}"].tolist()
    
    # Fit LabelEncoder on training labels
    logger.info("Fitting LabelEncoder on target labels...")
    le = LabelEncoder()
    y_train = le.fit_transform(train_df[TARGET_COL])
    y_val = le.transform(val_df[TARGET_COL])
    
    # 2. Fit Vectorizer on Train *only*
    logger.info("Fitting TF-IDF Vectorizer...")
    vectorizer = fit_tfidf_vectorizer(train_texts)
    
    x_train = transform_texts(vectorizer, train_texts)
    x_val = transform_texts(vectorizer, val_texts)
    
    # 3. Train Baseline Model (Naive Bayes)
    baseline_model = train_baseline_model(x_train, y_train, "naive_bayes")
    val_preds = baseline_model.predict(x_val.astype(np.float32))
    
    # Cast predictions to numpy for scoring compatibility
    if hasattr(val_preds, "get"):
        val_preds = val_preds.get()
    elif hasattr(val_preds, "to_numpy"):
        val_preds = val_preds.to_numpy()
        
    baseline_score = float(f1_score(y_val, val_preds, average='macro'))
    logger.info(f"Baseline (Naive Bayes) Validation F1-Macro: {baseline_score:.4f}")
    
    # 4. Grid Search / Tune Candidate Models
    best_overall_score = baseline_score
    best_overall_model = baseline_model
    best_model_name = "naive_bayes"
    
    candidates = ["logistic_regression", "svm", "xgboost"]
    
    for candidate in candidates:
        model, best_params = run_grid_search(x_train, y_train, candidate)
        if model is not None:
            preds = model.predict(x_val.astype(np.float32))
            if hasattr(preds, "get"):
                preds = preds.get()
            elif hasattr(preds, "to_numpy"):
                preds = preds.to_numpy()
                
            score = float(f1_score(y_val, preds, average='macro'))
            logger.info(f"Candidate: {candidate} | Best Params: {best_params} | Validation F1-Macro: {score:.4f}")
            
            if score > best_overall_score:
                best_overall_score = score
                best_overall_model = model
                best_model_name = candidate
                
    logger.info(f"=== Champion Model Selected: {best_model_name} (Val F1-Macro: {best_overall_score:.4f}) ===")
    
    # 5. Save Model Artifacts
    save_model_artifacts(best_overall_model, vectorizer, le)
    logger.info("--- Model Development Pipeline Completed Successfully ---")

if __name__ == "__main__":
    train_pipeline()
