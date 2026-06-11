# 🎫 Customer Support Ticket Classification System

An end-to-end, production-grade Machine Learning and Natural Language Processing (NLP) pipeline designed to automatically classify customer support tickets into issue categories and departments.

This repository demonstrates rigorous preprocessing, feature engineering, multi-model optimization, data leakage prevention, OOV diagnostics, and deployment-ready architecture.

---

## 🚀 Key Features

- **19-Stage Preprocessing Pipeline (`src/preprocessing.py`)**: Features Unicode normalization, contraction expansions, email/URL/mention masking, emoji translation, character-level repetition collapsing, and POS-aware lemmatization via SpaCy.
- **Advanced Representation Analysis (`src/feature_engineering.py`)**: Implements Bag-of-Words, Binary vectors, TF-IDF, and Hashing Trick options, comparing sparsity and memory footprints.
- **Robust Model Selection (`src/train.py`)**: Leverages Stratified K-Fold Cross Validation and GridSearchCV across four estimators (Naive Bayes, Logistic Regression, SVMs, and XGBoost) with GPU acceleration (cuML) and automatic CPU fallbacks.
- **Strict Leakage Prevention**: Data splits are executed before vectorizer fitting or label encoding, ensuring zero validation/test set information leaks into the training pipeline.
- **Inference Diagnostics (`src/inference.py`)**: A production-grade inference wrapper containing real-time Out-Of-Vocabulary (OOV) tracking to monitor vocabulary drift.
- **Recruiter-Ready Web App (`app/app.py`)**: A Streamlit interface displaying confidence distributions, OOV word diagnostics, and step-by-step text transformation walkthroughs.

---

## 📂 Repository Structure

The project follows standard professional machine learning software engineering standards:

*   📂 **`data/`**: Cleanly partitioned data splits (`raw/` and cache-ready `processed/`).
*   📂 **`notebooks/`**: Numbered, interactive prototyping notebooks (Phase 1 to Phase 5).
*   📂 **`src/`**: Modular source code modules (config, preprocessing, features, training, evaluation, inference, utils).
*   📂 **`models/`**: Serialized champion model, vectorizer, and label encoder.
*   📂 **`reports/`**: Classification reports and performance charts.
*   📂 **`outputs/`**: Log files (`pipeline.log`) and structured misclassified ticket lists for error auditing.
*   📂 **`tests/`**: Unit and integration test suites containing 27 passing tests.
*   📂 **`app/`**: Streamlit interactive deployment dashboard.

---

## 📈 Performance & Results

The training pipeline automatically tunes models and selects the champion based on the macro F1-score:

| Model Type | Optimization Method | Metrics Monitored |
| :--- | :--- | :--- |
| **Multinomial Naive Bayes** | Baseline | Validation Macro F1 |
| **Logistic Regression** | GridSearchCV (3-Fold Stratified CV) | Hyperparameters (`C`, `penalty`) |
| **Support Vector Classifier (SVC)** | GridSearchCV (3-Fold Stratified CV) | Kernels (`linear`, `rbf`) |
| **XGBoost Classifier** | GridSearchCV (3-Fold Stratified CV) | Depth and Learning Rate |

Error analysis is logged to [outputs/misclassified_tickets.csv](file:///home/ikartiksavaliya/Desktop/Portfolio%20projects/nlp/outputs/misclassified_tickets.csv) sorted by model prediction confidence, allowing you to audit the most confident errors for target label ambiguity.

---

## 💼 Business Value Metrics

1. **Automated Zero-Touch Routing**: Support tickets classified with $>90\%$ confidence are routed directly to the corresponding department, eliminating manual human triaging.
2. **Reduced Average Handling Time (AHT)**: Predicting the category and displaying the parsed entities inside the agent dashboard saves support reps valuable triage time.
3. **Vocabulary Shift Alerting**: Real-time OOV ratio tracking flags when customers begin using new keywords (e.g., new bugs, billing issues), alerting the product team to emerging outages.

---

## ⚙️ Quick Start

### 1. Set Up Environment & Install Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Verify with Pytest
```bash
.venv/bin/pytest
```

### 3. Run the Training & Evaluation Pipeline
```bash
python src/train.py
python src/evaluate.py
```

### 4. Run the Streamlit Dashboard
```bash
streamlit run app/app.py
```

---

## 🗺️ Project Blueprint
For a step-by-step breakdown of concepts, folder details, and syllabus mappings, refer to [project_blueprint.md](file:///home/ikartiksavaliya/Desktop/Portfolio%20projects/nlp/project_blueprint.md).
