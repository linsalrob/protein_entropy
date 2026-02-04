"""
Tests for entropy calculation.
"""

import numpy as np


def test_calculate_entropy_empty():
    """Test entropy calculation for empty sequence."""
    from protein_entropy.entropy import calculate_entropy

    assert calculate_entropy("") == 0.0
    assert calculate_entropy([]) == 0.0


def test_calculate_entropy_uniform():
    """Test entropy for uniform distribution."""
    from protein_entropy.entropy import calculate_entropy

    # Single character repeated - entropy should be 0
    assert calculate_entropy("AAAA") == 0.0

    # Two characters equally distributed
    # Entropy = -0.5 * log2(0.5) - 0.5 * log2(0.5) = 1.0
    ent = calculate_entropy("AAABBB")
    assert np.isclose(ent, 1.0)


def test_calculate_entropy_protein():
    """Test entropy for protein sequence."""
    from protein_entropy.entropy import calculate_entropy

    # All 20 amino acids once
    seq = "ACDEFGHIKLMNPQRSTVWY"
    ent = calculate_entropy(seq)

    # Maximum entropy for 20 symbols = log2(20) ≈ 4.32
    assert ent > 4.0
    assert ent < 5.0


def test_calculate_batch_entropy():
    """Test batch entropy calculation."""
    from protein_entropy.entropy import calculate_batch_entropy

    sequences = ["AAAA", "ACGT", "AAABBB"]
    entropies = calculate_batch_entropy(sequences)

    assert len(entropies) == 3
    assert entropies[0] == 0.0  # All same
    assert entropies[1] > 0.0  # Mixed
    assert np.isclose(entropies[2], 1.0)  # Equal distribution
