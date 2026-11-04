"""
Tests for encoder module (with mocked models).
"""

import importlib.util
from unittest.mock import Mock, patch

import pytest


def test_token_budget_batches():
    """Test token budget batching."""
    from protein_entropy.encoder import token_budget_batches

    sequences = ["A" * 100, "B" * 200, "C" * 150, "D" * 50]

    # With max_tokens=250, batches should be: [100], [200], [150, 50]
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
def test_modernprost_encoder_integration():
    """
    Integration test for the ModernProst encoder.

    This test actually loads and runs the model.
    Skip by default unless RUN_INTEGRATION=1 or --run-integration flag.
    """
    from protein_entropy.encoder import THREE_DI_ALPHABET, ModernProstEncoder

    encoder = ModernProstEncoder(device="cpu")

    sequences = ["ACDEFGHIKLM", "MKTAYIAKQRQISFVKSHFSRQ"]
    encoded = encoder.encode(sequences)

    assert len(encoded) == len(sequences)
    # ModernProst predicts exactly one 3Di state per residue
    for seq, three_di in zip(sequences, encoded):
        assert len(three_di) == len(seq)
        assert set(three_di) <= set(THREE_DI_ALPHABET)


def _mock_batch(seq_lens):
    """Build a padded tokenizer output for sequences of the given lengths."""
    import torch
    from transformers import BatchEncoding

    width = max(seq_lens)
    mask = torch.zeros(len(seq_lens), width, dtype=torch.long)
    for i, n in enumerate(seq_lens):
        mask[i, :n] = 1
    return BatchEncoding({"input_ids": torch.ones_like(mask), "attention_mask": mask})


@pytest.mark.skipif(not importlib.util.find_spec("torch"), reason="PyTorch not installed")
def test_prostt5_encoder_mock():
    """ProstT5 output is decoded from generate(), uppercased and trimmed to input length."""
    import torch

    from protein_entropy.encoder import ProstT5Encoder

    sequences = ["ACDEF", "GHI"]

    with (
        patch("protein_entropy.downloader.is_model_cached", return_value=True),
        patch("transformers.AutoModelForSeq2SeqLM") as mock_model_class,
        patch("transformers.T5Tokenizer") as mock_tokenizer_class,
    ):
        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        # <AA2fold> + residues + </s>
        mock_tokenizer.return_value = _mock_batch([len(s) + 2 for s in sequences])
        mock_tokenizer.batch_decode.return_value = ["d p v v d d", "w k k"]

        mock_model = Mock()
        mock_model_class.from_pretrained.return_value.to.return_value = mock_model
        mock_model.float.return_value = mock_model
        mock_model.generate.return_value = torch.zeros(2, 7, dtype=torch.long)

        encoder = ProstT5Encoder(device="cpu")
        encoded = encoder.encode(sequences)

    assert encoded == ["DPVVD", "WKK"]
    # The generation budget leaves room for every residue plus </s>
    assert mock_model.generate.call_args.kwargs["max_new_tokens"] == 6
    assert mock_model.generate.call_args.kwargs["min_new_tokens"] == 3
    processed = mock_tokenizer.call_args.args[0]
    assert processed == ["<AA2fold> A C D E F", "<AA2fold> G H I"]
    # The trailing </s> of each sequence is masked out
    attention_mask = mock_model.generate.call_args.kwargs["attention_mask"]
    assert attention_mask[0, 6] == 0
    assert attention_mask[1, 4] == 0


@pytest.mark.skipif(not importlib.util.find_spec("torch"), reason="PyTorch not installed")
@pytest.mark.parametrize("multitask", [True, False])
def test_modernprost_encoder_mock(multitask):
    """ModernProst argmax class IDs map onto the 3Di alphabet, one per residue."""
    import torch

    from protein_entropy.encoder import THREE_DI_ALPHABET, ModernProstEncoder

    sequences = ["ACDE", "MU"]
    lens = [len(s) for s in sequences]

    # Class IDs 0..3 for the first protein, 19 and 5 for the second; padding is junk
    class_ids = torch.tensor([[0, 1, 2, 3], [19, 5, 7, 7]])
    logits_3di = torch.nn.functional.one_hot(class_ids, num_classes=20).float()

    with (
        patch("protein_entropy.downloader.is_model_cached", return_value=True),
        patch("transformers.AutoConfig") as mock_config_class,
        patch("transformers.AutoTokenizer") as mock_tokenizer_class,
        patch("transformers.AutoModel") as mock_model_class,
    ):
        mock_config_class.from_pretrained.return_value = Mock(spec=[])

        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_tokenizer.return_value = _mock_batch(lens)

        mock_model = Mock()
        mock_model_class.from_pretrained.return_value.to.return_value = mock_model
        output = Mock()
        if multitask:
            output.logits = {"3di": logits_3di, "12st": torch.zeros(2, 4, 12)}
        else:
            output.logits = logits_3di
        mock_model.return_value = output

        encoder = ModernProstEncoder(device="cpu")
        encoded = encoder.encode(sequences)

    assert encoded == [
        THREE_DI_ALPHABET[0:4],
        THREE_DI_ALPHABET[19] + THREE_DI_ALPHABET[5],
    ]
    # Rare residues are replaced by X and no special tokens are added
    assert mock_tokenizer.call_args.args[0] == ["ACDE", "MX"]
    assert mock_tokenizer.call_args.kwargs["add_special_tokens"] is False


@pytest.mark.skipif(not importlib.util.find_spec("torch"), reason="PyTorch not installed")
def test_modernprost_rejects_wrong_head_size():
    """A head that is not 20-way cannot be decoded as 3Di."""
    import torch

    from protein_entropy.encoder import ModernProstEncoder

    encoder = ModernProstEncoder.__new__(ModernProstEncoder)
    encoder.model_name = "test"
    encoder.device = "cpu"
    encoder.tokenizer = Mock(return_value=_mock_batch([3]))
    encoder.model = Mock(return_value=Mock(logits=torch.zeros(1, 3, 33)))

    with pytest.raises(RuntimeError, match="3Di logits"):
        encoder.encode(["ACD"])


def test_encode_sequences_unknown_model():
    """Unknown model names are rejected before any model is loaded."""
    from protein_entropy.encoder import encode_sequences

    with pytest.raises(ValueError, match="Unknown model type"):
        encode_sequences(["ACD"], model_type="nope", device="cpu")


def test_from_pretrained_retries_online_when_cache_incomplete():
    """A partial HF cache (cache-only load fails) falls back to a network load."""
    from protein_entropy.encoder import _from_pretrained

    loader = Mock()
    loader.from_pretrained.side_effect = [OSError("missing modeling file"), "loaded"]

    with patch("protein_entropy.downloader.is_model_cached", return_value=True):
        assert _from_pretrained(loader, "org/model", trust_remote_code=True) == "loaded"

    first, second = loader.from_pretrained.call_args_list
    assert first.kwargs["local_files_only"] is True
    assert "local_files_only" not in second.kwargs
    assert first.kwargs["trust_remote_code"] and second.kwargs["trust_remote_code"]


def test_from_pretrained_uses_configured_cache_dir(monkeypatch, tmp_path):
    """Models downloaded to PROTEIN_ENTROPY_CACHE are loaded from there."""
    from protein_entropy.encoder import _from_pretrained

    monkeypatch.setenv("PROTEIN_ENTROPY_CACHE", str(tmp_path))
    snapshot = tmp_path / "models--org--model" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")

    loader = Mock()
    _from_pretrained(loader, "org/model")
    assert loader.from_pretrained.call_args.kwargs == {
        "cache_dir": str(tmp_path),
        "local_files_only": True,
    }


def test_modernprost_refuses_unknown_remote_code():
    """trust_remote_code is only enabled for known ModernProst repos or local dirs."""
    from protein_entropy.encoder import _trust_remote_code

    assert _trust_remote_code("gbouras13/modernprost-50M")
    for repo in ("someone/untrusted-model", "Rostlab/ProstT5_fp16"):
        with pytest.raises(ValueError, match="Refusing to run remote code"):
            _trust_remote_code(repo)
