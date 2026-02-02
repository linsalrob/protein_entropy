"""
Device detection and GPU utilities.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_device() -> str:
    """
    Auto-detect the best available device.

    Tries in order:
    1. CUDA (NVIDIA GPUs)
    2. MPS (Apple Silicon)
    3. CPU (fallback)

    Returns:
        str: Device name ('cuda', 'mps', or 'cpu')
    """
    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            logger.info("Using Apple MPS device")
        else:
            device = "cpu"
            logger.info("Using CPU device")

        return device
    except ImportError:
        logger.warning("PyTorch not available, falling back to CPU")
        return "cpu"


def clear_gpu_memory() -> None:
    """
    Clear GPU memory cache if available.
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("GPU memory cache cleared")
    except (ImportError, Exception) as e:
        logger.debug(f"Could not clear GPU memory: {e}")


def get_gpu_memory_info() -> Optional[dict]:
    """
    Get GPU memory information if available.

    Returns:
        dict or None: Dictionary with 'allocated', 'reserved', and 'free' memory in bytes,
                     or None if GPU not available
    """
    try:
        import torch

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            total = torch.cuda.get_device_properties(0).total_memory
            free = total - allocated

            return {
                "allocated": allocated,
                "reserved": reserved,
                "free": free,
                "total": total,
            }
    except (ImportError, Exception) as e:
        logger.debug(f"Could not get GPU memory info: {e}")

    return None
