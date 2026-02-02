"""
Tests for encoder module (with mocked models).
"""

import pytest
from unittest.mock import Mock, patch, MagicMock


def test_token_budget_batches():
    """Test token budget batching."""
    from protein_entropy.encoder import token_budget_batches
    
    sequences = ["A" * 100, "B" * 200, "C" * 150, "D" * 50]
    
    # With max_tokens=250, should get: [100+200=300 -> split], [100], [200], [150+50=200]
    # Actually: [100], [200], [150, 50]
    batches = token_budget_batches(sequences, max_tokens=250)
    
    assert len(batches) > 0
    
    # Check that no batch exceeds the token limit
    for batch in batches:
        total_tokens = sum(len(seq) for seq in batch)
        assert total_tokens <= 250 or len(batch) == 1  # Single sequence might exceed


def test_token_budget_batches_empty():
    """Test batching with empty list."""
    from protein_entropy.encoder import token_budget_batches
    
    batches = token_budget_batches([])
    assert batches == []


@pytest.mark.integration
@pytest.mark.skipif(
    not pytest.config.getoption("--run-integration", default=False),
    reason="Integration test - run with --run-integration"
)
def test_prostt5_encoder_integration():
    """
    Integration test for ProstT5 encoder.
    
    This test actually loads and runs the model.
    Skip by default unless RUN_INTEGRATION=1 or --run-integration flag.
    """
    from protein_entropy.encoder import ProstT5Encoder
    
    encoder = ProstT5Encoder(device="cpu")
    
    sequences = ["ACDEFGHIKLM"]
    encoded = encoder.encode(sequences)
    
    assert len(encoded) == 1
    assert len(encoded[0]) == len(sequences[0])
    assert all(c.islower() for c in encoded[0])


def test_prostt5_encoder_mock():
    """Test ProstT5 encoder with mocked model."""
    from protein_entropy.encoder import ProstT5Encoder
    
    with patch("protein_entropy.encoder.T5EncoderModel") as mock_model_class, \
         patch("protein_entropy.encoder.T5Tokenizer") as mock_tokenizer_class:
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_tokenizer.return_value = {
            "input_ids": MagicMock(),
            "attention_mask": MagicMock(),
        }
        
        # Mock model
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.eval.return_value = None
        mock_model.to.return_value = mock_model
        
        # Mock output
        import torch
        mock_output = Mock()
        mock_output.last_hidden_state = torch.randn(1, 10, 768)
        mock_model.return_value = mock_output
        
        # Test encoder
        encoder = ProstT5Encoder(device="cpu")
        
        # The mocked encoder should work
        assert encoder.model is not None
        assert encoder.tokenizer is not None


def test_modernprost_encoder_mock():
    """Test ModernProst encoder with mocked model."""
    from protein_entropy.encoder import ModernProstEncoder
    
    with patch("protein_entropy.encoder.AutoTokenizer") as mock_tokenizer_class, \
         patch("protein_entropy.encoder.AutoModelForMaskedLM") as mock_model_class:
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_tokenizer.return_value = {
            "input_ids": MagicMock(),
            "attention_mask": MagicMock(),
        }
        
        # Mock model
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.eval.return_value = None
        mock_model.to.return_value = mock_model
        
        # Test encoder
        encoder = ModernProstEncoder(device="cpu")
        
        # The mocked encoder should work
        assert encoder.model is not None
        assert encoder.tokenizer is not None
