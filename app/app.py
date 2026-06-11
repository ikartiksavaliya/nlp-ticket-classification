"""
Streamlit Web Application for Customer Support Ticket Classification.
Integrates the TicketInferencePipeline to classify user-submitted ticket texts.
Provides real-time visualization of predictions, confidence levels, and OOV word tracking.
"""

import sys
from pathlib import Path
import os

# Guarantee that the project root is in python path for clean modular imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import streamlit as st
import pandas as pd
import numpy as np

from src.inference import TicketInferencePipeline
from src.config import MODEL_PATH, VECTORIZER_PATH, LABEL_ENCODER_PATH
from src.preprocessing import (
    normalize_unicode,
    handle_contractions,
    mask_entities,
    handle_emojis,
    remove_punctuation_and_numbers,
    normalize_whitespace,
    tokenize_words,
    remove_stopwords,
    handle_spelling_variations,
    apply_lemmatization
)

@st.cache_resource
def load_pipeline():
    """Cached loader for the model artifacts."""
    if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH) and os.path.exists(LABEL_ENCODER_PATH)):
        return None
    return TicketInferencePipeline(
        model_path=str(MODEL_PATH),
        vectorizer_path=str(VECTORIZER_PATH),
        label_encoder_path=str(LABEL_ENCODER_PATH)
    )

def step_by_step_preprocessing(text: str) -> list:
    """Performs step-by-step cleaning of text to show in the UI."""
    steps = []
    
    # 1. Unicode
    current = normalize_unicode(text)
    steps.append(("Unicode Normalization", current))
    
    # 2. Lowercase
    current = current.lower()
    steps.append(("Lowercasing", current))
    
    # 3. Expand Contractions
    current = handle_contractions(current)
    steps.append(("Expand Contractions", current))
    
    # 4. Mask Entities
    current = mask_entities(current)
    steps.append(("Mask Entities (URLs/Emails/Mentions)", current))
    
    # 5. Handle Emojis
    current, emojis = handle_emojis(current)
    emoji_suffix = f" (Extracted: {', '.join(emojis)})" if emojis else ""
    steps.append((f"Emoji Handling{emoji_suffix}", current))
    
    # 6. Remove Punctuation/Numbers
    current = remove_punctuation_and_numbers(current)
    steps.append(("Remove Punctuation & Numbers", current))
    
    # 7. Normalize Whitespace
    current = normalize_whitespace(current)
    steps.append(("Normalize Whitespace", current))
    
    # 8. Tokenize Words
    temp_text = current.replace("<URL>", "URLMASK").replace("<EMAIL>", "EMAILMASK").replace("<MENTION>", "MENTIONMASK")
    tokens = tokenize_words(temp_text)
    steps.append(("Word Tokenization", " | ".join(tokens)))
    
    # 9. Remove Stopwords
    tokens = remove_stopwords(tokens)
    steps.append(("Stopword Filter", " | ".join(tokens)))
    
    # 10. Spelling variations
    tokens = handle_spelling_variations(tokens)
    steps.append(("Spelling Repetitions Collapsed", " | ".join(tokens)))
    
    # 11. Lemmatize
    tokens = apply_lemmatization(tokens)
    steps.append(("SpaCy Lemmatization", " | ".join(tokens)))
    
    # 12. Reassemble
    cleaned_tokens = []
    for t in tokens:
        t_clean = t.strip()
        if not t_clean:
            continue
        if t_clean.upper() == "URLMASK":
            cleaned_tokens.append("<URL>")
        elif t_clean.upper() == "EMAILMASK":
            cleaned_tokens.append("<EMAIL>")
        elif t_clean.upper() == "MENTIONMASK":
            cleaned_tokens.append("<MENTION>")
        else:
            cleaned_tokens.append(t_clean)
    final = " ".join(cleaned_tokens)
    steps.append(("Final Reassembled Tokens", final))
    
    return steps

def main():
    st.set_page_config(
        page_title="Ticket Classification System",
        page_icon="🎫",
        layout="wide"
    )
    
    # Inject styling to make application premium and modern
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
        
        .main-title {
            background: linear-gradient(135deg, #4fd1c5 0%, #3182ce 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-family: 'Outfit', sans-serif;
            font-weight: 700;
            font-size: 2.8rem;
            margin-bottom: 0.2rem;
        }
        .sub-title {
            color: #a0aec0;
            font-family: 'Outfit', sans-serif;
            font-size: 1.1rem;
            margin-bottom: 1.8rem;
        }
        .card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }
        .metric-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
            margin-top: 1rem;
        }
        .metric-box {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
        }
        .metric-val {
            font-size: 2rem;
            font-weight: 700;
            color: #4fd1c5;
        }
        .metric-val-oov {
            font-size: 2rem;
            font-weight: 700;
            color: #e53e3e;
        }
        .metric-lbl {
            font-size: 0.85rem;
            color: #a0aec0;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 0.25rem;
        }
        .predicted-label {
            font-size: 1.6rem;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 0.5rem;
        }
        .oov-badge {
            background: rgba(229, 62, 62, 0.15);
            border: 1px solid rgba(229, 62, 62, 0.3);
            color: #feb2b2;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-size: 0.85rem;
            display: inline-block;
            margin: 0.2rem;
        }
        .bar-container {
            margin-bottom: 0.8rem;
        }
        .bar-lbl {
            display: flex;
            justify-content: space-between;
            font-size: 0.9rem;
            margin-bottom: 0.25rem;
        }
        .bar-bg {
            background: rgba(255, 255, 255, 0.07);
            border-radius: 4px;
            height: 8px;
            width: 100%;
        }
        .bar-fill {
            height: 100%;
            border-radius: 4px;
        }
        .step-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }
        .step-table th {
            text-align: left;
            padding: 8px;
            border-bottom: 2px solid rgba(255, 255, 255, 0.1);
            color: #a0aec0;
        }
        .step-table td {
            padding: 8px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown('<div class="main-title">🎫 Support Ticket Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Production NLP Classification & Diagnostic Dashboard</div>', unsafe_allow_html=True)
    
    # Load pipeline
    pipeline = load_pipeline()
    
    # 1. Sidebar Setup
    st.sidebar.header("⚙️ Configuration & Diagnostics")
    
    if pipeline is None:
        st.sidebar.error("⚠️ Model artifacts not found!")
        st.sidebar.warning("Please verify models/ directory or run training pipeline first.")
        st.error("Model artifacts not found in standard directories. Please run the training pipeline first.")
        return
        
    st.sidebar.success("✅ Model Pipeline Loaded Successfully")
    
    # Display vocabulary size and classes
    vocab_len = len(pipeline.vectorizer.vocabulary_)
    num_classes = len(pipeline.label_encoder.classes_)
    
    st.sidebar.markdown(
        f"""
        <div class="card">
            <b>Pipeline Metadata</b><br>
            • Classifier: <code>{type(pipeline.model).__name__}</code><br>
            • Classes Count: <code>{num_classes}</code><br>
            • Vocabulary Size: <code>{vocab_len:,}</code> features
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Load classification report if available
    report_path = project_root / "reports" / "classification_report.csv"
    if os.path.exists(report_path):
        st.sidebar.markdown("📈 <b>Model Performance (Test Set)</b>", unsafe_allow_html=True)
        try:
            report_df = pd.read_csv(report_path, index_col=0)
            # Filter class rows only (ignore averages)
            metrics_clean = report_df.loc[
                ~report_df.index.isin(['accuracy', 'macro avg', 'weighted avg'])
            ].copy()
            st.sidebar.dataframe(
                metrics_clean[['precision', 'recall', 'f1-score']].style.format("{:.2f}")
            )
        except Exception:
            pass
            
    # 2. Main Input Section
    col_left, col_right = st.columns([1.1, 0.9])
    
    with col_left:
        st.subheader("Input Ticket Text")
        ticket_text = st.text_area(
            "Paste the customer support ticket message below:",
            height=200,
            placeholder="Type ticket details here (e.g., My payment was charged twice but my account still says subscription expired...)"
        )
        
        classify_clicked = st.button("Classify Ticket", type="primary", use_container_width=True)
        
    with col_right:
        if classify_clicked:
            if not ticket_text.strip():
                st.warning("Please enter some ticket text to classify.")
            else:
                with st.spinner("Executing inference..."):
                    result = pipeline.predict_single(ticket_text)
                    
                # 3. Main Results Display
                st.subheader("Classification Summary")
                
                # Highlight class and confidence
                pred_class = result["predicted_category"]
                confidence = result["confidence"]
                
                # Select badge background gradient based on classification
                st.markdown(
                    f"""
                    <div class="card">
                        <div class="predicted-label">Category: {pred_class}</div>
                        <div class="metric-container">
                            <div class="metric-box">
                                <div class="metric-val">{confidence*100:.1f}%</div>
                                <div class="metric-lbl">Confidence Score</div>
                            </div>
                            <div class="metric-box">
                                <div class="metric-val-oov">{result["oov_analysis"]["oov_ratio"]*100:.1f}%</div>
                                <div class="metric-lbl">OOV Words Ratio</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
    # 4. Detailed analysis shown after prediction
    if classify_clicked and ticket_text.strip():
        # Display breakdown of probabilities
        tab1, tab2, tab3 = st.tabs(["📊 Class Probabilities", "🔍 Text Preprocessing Walkthrough", "🛑 OOV Word Diagnostic"])
        
        with tab1:
            st.subheader("Prediction Confidence Breakdowns")
            # Sort categories by probability
            sorted_probs = sorted(result["probabilities"].items(), key=lambda x: x[1], reverse=True)
            
            for label, val in sorted_probs:
                # Color code depending on probability strength
                if val > 0.5:
                    color = "linear-gradient(90deg, #4fd1c5, #319795)"
                elif val > 0.15:
                    color = "linear-gradient(90deg, #3182ce, #2b6cb0)"
                else:
                    color = "linear-gradient(90deg, #718096, #4a5568)"
                
                st.markdown(
                    f"""
                    <div class="bar-container">
                        <div class="bar-lbl">
                            <span><b>{label}</b></span>
                            <span>{val*100:.2f}%</span>
                        </div>
                        <div class="bar-bg">
                            <div class="bar-fill" style="width: {val*100}%; background: {color};"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
        with tab2:
            st.subheader("Step-by-step Transformation Pipeline")
            st.markdown("Inspect how the raw customer support ticket text was standardized and tokenized step by step:")
            
            steps = step_by_step_preprocessing(ticket_text)
            
            # Display step-by-step cleaning steps in a custom table
            table_rows = ""
            for idx, (step_name, step_val) in enumerate(steps):
                display_val = f"<code>{step_val}</code>" if step_val else "<i>[Empty]</i>"
                table_rows += f"<tr><td><b>{idx+1}. {step_name}</b></td><td>{display_val}</td></tr>"
                
            st.markdown(
                f"""
                <table class="step-table">
                    <thead>
                        <tr>
                            <th style="width: 30%;">Preprocessing Stage</th>
                            <th style="width: 70%;">Output Text State</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows}
                    </tbody>
                </table>
                """,
                unsafe_allow_html=True
            )
            
        with tab3:
            st.subheader("Out-Of-Vocabulary (OOV) Word Diagnostics")
            st.markdown(
                """
                OOV analysis identifies words that did not appear in the training corpus vocabulary.
                This is a vital metric to monitor for concept drift, vocabulary shift, or novel customer issues.
                """
            )
            
            oov_data = result["oov_analysis"]
            
            st.markdown(f"**OOV Tokens Percentage:** `{oov_data['oov_ratio']*100:.2f}%`")
            st.markdown(f"**Total OOV Tokens Count:** `{len(oov_data['oov_words'])}`")
            
            if oov_data["oov_words"]:
                st.markdown("**Identified Out-Of-Vocabulary Words:**")
                badges_html = "".join([f'<span class="oov-badge">{word}</span>' for word in oov_data["oov_words"]])
                st.markdown(badges_html, unsafe_allow_html=True)
            else:
                st.info("Excellent! All words in the ticket exist in the model's vocabulary. OOV Rate is 0.00%.")

if __name__ == "__main__":
    main()
