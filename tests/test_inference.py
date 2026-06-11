"""
Integration and Unit tests for the Inference Pipeline.
Tests the TicketInferencePipeline under standard, empty, and edge inputs in src/inference.py.
"""

import pytest
from src.inference import TicketInferencePipeline
from src.config import MODEL_PATH, VECTORIZER_PATH, LABEL_ENCODER_PATH

@pytest.fixture(scope="module")
def inference_pipeline():
    """Fixture to load the inference pipeline once for all tests."""
    return TicketInferencePipeline(
        model_path=str(MODEL_PATH),
        vectorizer_path=str(VECTORIZER_PATH),
        label_encoder_path=str(LABEL_ENCODER_PATH)
    )

def test_pipeline_single_prediction(inference_pipeline):
    """Verify that a single valid ticket description returns the expected output dictionary schema."""
    ticket = "I am having trouble accessing my billing dashboard and need a refund for my recent transaction."
    result = inference_pipeline.predict_single(ticket)
    
    # Check keys in the schema
    assert "predicted_category" in result
    assert "confidence" in result
    assert "probabilities" in result
    assert "oov_analysis" in result
    
    # Check data types and ranges
    assert isinstance(result["predicted_category"], str)
    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 1.0
    assert isinstance(result["probabilities"], dict)
    assert isinstance(result["oov_analysis"], dict)
    
    # Check oov analysis keys
    assert "oov_ratio" in result["oov_analysis"]
    assert "oov_words" in result["oov_analysis"]
    assert 0.0 <= result["oov_analysis"]["oov_ratio"] <= 1.0
    assert isinstance(result["oov_analysis"]["oov_words"], list)

def test_pipeline_batch_prediction(inference_pipeline):
    """Verify that batch predictions return lists of dictionaries with correct dimensions."""
    tickets = [
        "Please cancel my monthly premium plan subscription immediately.",
        "My app keeps crashing whenever I try to upload a profile picture."
    ]
    results = inference_pipeline.predict_batch(tickets)
    
    assert isinstance(results, list)
    assert len(results) == len(tickets)
    for res in results:
        assert "predicted_category" in res
        assert "confidence" in res
        assert "probabilities" in res
        assert "oov_analysis" in res

def test_pipeline_empty_input(inference_pipeline):
    """Verify that empty inputs, spaces-only, or null strings are handled gracefully without raising exceptions."""
    empty_cases = ["", "   ", None]
    results = inference_pipeline.predict_batch(empty_cases)
    
    assert len(results) == len(empty_cases)
    for res in results:
        assert "predicted_category" in res
        assert "confidence" in res
        assert "probabilities" in res
        assert "oov_analysis" in res
        # For empty input, OOV ratio should be 0.0
        assert res["oov_analysis"]["oov_ratio"] == 0.0
        assert len(res["oov_analysis"]["oov_words"]) == 0

def test_pipeline_all_oov_input(inference_pipeline):
    """Verify that a ticket with only out-of-vocabulary words returns a uniform distribution or default prediction gracefully."""
    oov_ticket = "xyzabc12345 qqqwwwpppeee zzzxxxyyy"
    result = inference_pipeline.predict_single(oov_ticket)
    
    assert "predicted_category" in result
    assert result["oov_analysis"]["oov_ratio"] == 1.0
    assert len(result["oov_analysis"]["oov_words"]) > 0
