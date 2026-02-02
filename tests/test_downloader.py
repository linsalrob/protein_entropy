"""
Tests for downloader module.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, Mock


def test_available_models():
    """Test that available models are defined."""
    from protein_entropy.downloader import AVAILABLE_MODELS
    
    assert "prostt5_fp16" in AVAILABLE_MODELS
    assert "modernprost" in AVAILABLE_MODELS
    assert AVAILABLE_MODELS["prostt5_fp16"] == "Rostlab/ProstT5_fp16"
    assert AVAILABLE_MODELS["modernprost"] == "gbouras13/modernprost-profiles"


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
