"""
Shannon entropy calculation for sequences.
"""

from collections import Counter
from typing import Union

import numpy as np

from .logging_config import get_logger

logger = get_logger(__name__)


def calculate_entropy(sequence: Union[str, list]) -> float:
    """
    Calculate Shannon entropy for a sequence.

    Shannon entropy: H = -sum(p_i * log2(p_i))
    where p_i is the probability of symbol i

    Args:
        sequence: String or list of symbols

    Returns:
        Shannon entropy value
    """
    if not sequence:
        return 0.0

    # Count occurrences of each symbol
    counts = Counter(sequence)
    total = len(sequence)

    # Calculate probabilities and entropy
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            probability = count / total
            entropy -= probability * np.log2(probability)

    return float(entropy)


def calculate_batch_entropy(sequences: list[str]) -> list[float]:
    """
    Calculate entropy for multiple sequences.

    Args:
        sequences: List of sequences

    Returns:
        List of entropy values
    """
    logger.debug(f"Calculating entropy for {len(sequences)} sequences")
    entropies = [calculate_entropy(seq) for seq in sequences]
    return entropies
