"""
Model and asset downloader.
"""

import logging
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

# Available models
AVAILABLE_MODELS = {
    "prostt5_fp16": "Rostlab/ProstT5_fp16",
    "modernprost": "gbouras13/modernprost-profiles",
}


def get_cache_dir() -> Path:
    """
    Get the cache directory for models.
    
    Returns:
        Path to cache directory
    """
    from pathlib import Path
    import os
    
    # Use HuggingFace cache by default, or user-specified location
    cache_dir = os.environ.get("PROTEIN_ENTROPY_CACHE")
    if cache_dir:
        return Path(cache_dir)
    
    # Default to HuggingFace cache
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    
    return Path.home() / ".cache" / "huggingface" / "hub"


def download_model(
    model_name: str,
    cache_dir: Optional[str] = None,
    force_download: bool = False,
) -> str:
    """
    Download a model from HuggingFace Hub.
    
    Args:
        model_name: Name of the model to download (prostt5_fp16 or modernprost)
        cache_dir: Optional cache directory path
        force_download: Force re-download even if cached
        
    Returns:
        Path to downloaded model directory
        
    Raises:
        ValueError: If model_name is not recognized
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {', '.join(AVAILABLE_MODELS.keys())}"
        )
    
    repo_id = AVAILABLE_MODELS[model_name]
    
    if cache_dir is None:
        cache_dir = str(get_cache_dir())
    
    logger.info(f"Downloading model {model_name} ({repo_id}) to {cache_dir}")
    
    try:
        model_path = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            force_download=force_download,
        )
        logger.info(f"Model downloaded successfully to: {model_path}")
        return model_path
    except Exception as e:
        logger.error(f"Failed to download model {model_name}: {e}")
        raise


def list_downloaded_models(cache_dir: Optional[str] = None) -> list[str]:
    """
    List models that have been downloaded.
    
    Args:
        cache_dir: Optional cache directory path
        
    Returns:
        List of downloaded model names
    """
    if cache_dir is None:
        cache_dir = str(get_cache_dir())
    
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return []
    
    downloaded = []
    for model_name, repo_id in AVAILABLE_MODELS.items():
        # Check if model directory exists in cache
        # HuggingFace cache uses "models--" prefix
        repo_cache_name = "models--" + repo_id.replace("/", "--")
        model_cache_path = cache_path / repo_cache_name
        
        if model_cache_path.exists():
            downloaded.append(model_name)
    
    return downloaded
