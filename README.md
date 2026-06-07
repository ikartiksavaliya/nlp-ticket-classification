# Customer Support Ticket Classification using Traditional NLP and Machine Learning

This repository hosts a portfolio-grade, end-to-end Machine Learning and Natural Language Processing project designed to automatically classify customer support tickets into issues and departments.

## 🚀 Project Overview

The core objective is to take raw, unstructured support tickets and categorize them with high precision. The system is designed using a modular engineering architecture, demonstrating robust text preprocessing, advanced text representation techniques, traditional machine learning models, systematic validation pipelines, and a production-grade inference engine.

## 📂 Repository Structure

The project follows a standard professional data science folder hierarchy:

*   📂 **`data/`**: Version-controlled partitions of the dataset (`raw/` and `processed/`).
*   📂 **`notebooks/`**: Phase-by-phase prototyping notebooks (exploratory analysis, cleaning, representations, modeling, evaluation).
*   📂 **`src/`**: Modularized, production-ready source code (config, preprocessing, features, training, evaluation, inference, utils).
*   📂 **`models/`**: Serialized model binaries, vectorizers, and encoder configurations.
*   📂 **`reports/`**: Generated performance figures (confusion matrices, ROC/PR curves).
*   📂 **`outputs/`**: Logs and lists of misclassified tickets for error auditing.
*   📂 **`tests/`**: Unit test suite targeting preprocessing, feature pipelines, and inference.
*   📂 **`app/`**: A Streamlit application showing the pipeline in action.

## 🗺️ Execution Blueprint

For a detailed walkthrough of the project planning, modular architecture, concept-to-code mapping, and delivery milestones, please refer to the master blueprint:
👉 **[project_blueprint.md](file:///home/ikartiksavaliya/Desktop/Portfolio%20projects/nlp/project_blueprint.md)**

## ⚙️ Setting Up the Environment

1.  **Clone this Repository**
2.  **Create a Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Verify Setup with pytest:**
    ```bash
    pytest tests/
    ```

## 📋 Syllabus Concept Coverage Map
Every key NLP topic—including text normalization, Unicode cleaning, entity masking, emoji translation, stemming vs lemmatization, Bag of Words, TF-IDF, the Hashing Trick, model cross-validation, OOV analysis, and data leakage prevention—is covered in this project. See the full breakdown in [project_blueprint.md](file:///home/ikartiksavaliya/Desktop/Portfolio%20projects/nlp/project_blueprint.md#5-nlp-concept-mapping-matrix).
