"""
Inference Pipeline module for Customer Support Ticket Classification.
Exposes a production-ready interface for classifying raw, incoming support tickets.
Handles loading of artifacts, preprocessing, text representation, and OOV checking.
"""

from typing import Dict, List, Tuple, Union, Any
import numpy as np
import joblib
from src.preprocessing import preprocess_text

class TicketInferencePipeline:
    """
    Encapsulates the end-to-end inference process:
    Raw Support Ticket text -> Preprocessing -> Vectorization -> Classification -> JSON Output.
    """
    def __init__(self, model_path: str, vectorizer_path: str, label_encoder_path: str):
        """
        Loads the serialized classifier, vectorizer, and label encoder.
        
        Args:
            model_path: Path to serialized model.
            vectorizer_path: Path to serialized vectorizer.
            label_encoder_path: Path to serialized label encoder.
        """
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.label_encoder = joblib.load(label_encoder_path)

    def _track_oov_words(self, preprocessed_text: str) -> Dict[str, Any]:
        """
        Analyzes the input text for Out-Of-Vocabulary (OOV) terms compared to
        the vectorizer's vocabulary.
        
        Args:
            preprocessed_text: Preprocessed text string.
            
        Returns:
            Dictionary containing percentage of OOV tokens and the list of OOV words.
        """
        tokens = preprocessed_text.split() if preprocessed_text else []
        if not tokens:
            return {"oov_ratio": 0.0, "oov_words": []}
            
        vocab = getattr(self.vectorizer, "vocabulary_", {})
        oov = [w for w in tokens if w not in vocab]
        
        oov_ratio = len(oov) / len(tokens)
        oov_words = sorted(list(set(oov)))
        
        return {
            "oov_ratio": float(oov_ratio),
            "oov_words": oov_words
        }

    def predict_single(self, text: str) -> Dict[str, Any]:
        """
        Predicts the category of a single support ticket.
        
        Args:
            text: Raw ticket text description.
            
        Returns:
            Dictionary containing:
            - "predicted_category": Label string.
            - "confidence": Float probability.
            - "probabilities": Dict mapping category labels to confidence percentages.
            - "oov_analysis": Result of _track_oov_words.
            
        Raises:
            ValueError: If input is invalid.
        """
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predicts categories for a list of support tickets.
        Handles large batch inputs efficiently.
        
        Args:
            texts: List of raw ticket text descriptions.
            
        Returns:
            List of dictionaries matching the structure of predict_single.
        """
        if not texts:
            return []
            
        cleaned_texts = []
        for t in texts:
            if t is None or not isinstance(t, str):
                cleaned_texts.append("")
            else:
                cleaned_texts.append(preprocess_text(t))
                
        oov_analyses = [self._track_oov_words(ct) for ct in cleaned_texts]
        
        # Transform using the vectorizer
        transformed = self.vectorizer.transform(cleaned_texts).astype(np.float32)
        
        # Predict class probabilities
        probabilities = self.model.predict_proba(transformed)
        if hasattr(probabilities, "get"):
            probabilities = probabilities.get()
            
        results = []
        classes = self.label_encoder.classes_
        for i, probs in enumerate(probabilities):
            predicted_idx = int(np.argmax(probs))
            predicted_category = str(classes[predicted_idx])
            confidence = float(probs[predicted_idx])
            prob_dict = {str(classes[j]): float(probs[j]) for j in range(len(classes))}
            
            results.append({
                "predicted_category": predicted_category,
                "confidence": confidence,
                "probabilities": prob_dict,
                "oov_analysis": oov_analyses[i]
            })
            
        return results

if __name__ == "__main__":
    import argparse
    from src.config import MODEL_PATH, VECTORIZER_PATH, LABEL_ENCODER_PATH
    import json
    
    parser = argparse.ArgumentParser(description="Run ticket classification inference.")
    parser.add_argument("text", type=str, help="Raw support ticket text to classify.")
    args = parser.parse_args()
    
    pipeline = TicketInferencePipeline(
        model_path=str(MODEL_PATH),
        vectorizer_path=str(VECTORIZER_PATH),
        label_encoder_path=str(LABEL_ENCODER_PATH)
    )
    result = pipeline.predict_single(args.text)
    print(json.dumps(result, indent=2))
