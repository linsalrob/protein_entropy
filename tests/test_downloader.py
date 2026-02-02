"""
Tests for downloader module.
"""

from pathlib import Path
from unittest.mock import patch

import pytest


def test_available_models():
    """Test that available models are defined."""
    from protein_entropy.downloader import AVAILABLE_MODELS

    assert "prostt5_fp16" in AVAILABLE_MODELS
    assert "modernprost" in AVAILABLE_MODELS
    assert "modernprost_base" in AVAILABLE_MODELS
    assert "modernprost_profiles" in AVAILABLE_MODELS
    assert AVAILABLE_MODELS["prostt5_fp16"] == "Rostlab/ProstT5_fp16"
    assert AVAILABLE_MODELS["modernprost"] == "gbouras13/modernprost-base"
    assert AVAILABLE_MODELS["modernprost_base"] == "gbouras13/modernprost-base"
    assert AVAILABLE_MODELS["modernprost_profiles"] == "gbouras13/modernprost-profiles"


def test_get_cache_dir():
    """Test cache directory detection."""
    from protein_entropy.downloader import get_cache_dir

    cache_dir = get_cache_dir()
    assert isinstance(cache_dir, Path)


def test_download_model_invalid():
    """Test downloading invalid model."""
    from protein_entropy.downloader import download_model

    with pytest.raises(ValueError, match="Unknown model"):
        download_model("nonexistent_model")


@patch("protein_entropy.downloader.snapshot_download")
def test_download_model_mock(mock_snapshot):
    """Test model download with mocked download."""
    from protein_entropy.downloader import download_model

    mock_snapshot.return_value = "/fake/path/to/model"

    result = download_model("prostt5_fp16", cache_dir="/tmp/test")

    assert result == "/fake/path/to/model"
    mock_snapshot.assert_called_once()

    # Check that the correct repo_id was used
    call_args = mock_snapshot.call_args
    assert call_args.kwargs["repo_id"] == "Rostlab/ProstT5_fp16"


def test_list_downloaded_models_empty(tmp_path):
    """Test listing models when none are downloaded."""
    from protein_entropy.downloader import list_downloaded_models

    models = list_downloaded_models(str(tmp_path))
    assert models == []


def test_is_model_cached_not_exists(tmp_path):
    """Test is_model_cached when model doesn't exist."""
    from protein_entropy.downloader import is_model_cached

    result = is_model_cached("Rostlab/ProstT5_fp16", str(tmp_path))
    assert result is False


def test_is_model_cached_exists(tmp_path):
    """Test is_model_cached when model exists."""
    from protein_entropy.downloader import is_model_cached

    # Create fake model cache structure
    model_cache = tmp_path / "models--Rostlab--ProstT5_fp16" / "snapshots" / "fake_snapshot"
    model_cache.mkdir(parents=True, exist_ok=True)
    
    # Create a dummy file to simulate model content
    (model_cache / "config.json").touch()

    result = is_model_cached("Rostlab/ProstT5_fp16", str(tmp_path))
    assert result is True
