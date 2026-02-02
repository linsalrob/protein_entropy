"""
protein_entropy: Encode proteins using transformers and calculate their entropy.

This package provides tools to:
- Convert protein sequences to 3Di structural tokens using ProstT5
- Calculate Shannon entropy for both protein and 3Di sequences
- Optimize GPU memory usage for large-scale encoding
"""

import importlib.metadata
import logging

# Get version from package metadata
try:
    __version__ = importlib.metadata.version("protein_entropy")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0.dev0"

# Set up package-level logger
logger = logging.getLogger(__name__)

__all__ = ["__version__"]
