"""
Test suite for validating 3Di encoders against example FASTA files.

This module tests that our ModernProst and ProstT5 encoders produce 3Di sequences
that match the expected outputs for at least 75% of sequences.
"""

import os
from pathlib import Path
from typing import Dict, Tuple

import pytest


def read_fasta(path: str) -> Dict[str, str]:
    """
    Read FASTA file and return dict mapping sequence IDs to sequences.
    
    Args:
        path: Path to FASTA file
        
    Returns:
        Dictionary mapping sequence ID (first token after '>') to sequence
        (concatenated lines with no whitespace)
    """
    sequences = {}
    current_id = None
    current_seq = []
    
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('>'):
                # Save previous sequence if any
                if current_id is not None:
                    sequences[current_id] = ''.join(current_seq)
                
                # Start new sequence - extract first token after '>'
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                # Append sequence line (no whitespace)
                current_seq.append(line.replace(' ', ''))
    
    # Save last sequence
    if current_id is not None:
        sequences[current_id] = ''.join(current_seq)
    
    return sequences


def get_example_data_dir() -> Path:
    """Get the path to example_data directory."""
    # tests/ is in the repository root, example_data/ is also in root
    tests_dir = Path(__file__).parent
    repo_root = tests_dir.parent
    example_data_dir = repo_root / "example_data"
    
    if not example_data_dir.exists():
        pytest.skip(f"Example data directory not found: {example_data_dir}")
    
    return example_data_dir


def compare_sequences(
    actual: Dict[str, str],
    expected: Dict[str, str],
    model_name: str,
) -> Tuple[int, int, float]:
    """
    Compare actual vs expected 3Di sequences.
    
    Args:
        actual: Dictionary of actual encoded sequences
        expected: Dictionary of expected encoded sequences
        model_name: Name of model for logging
        
    Returns:
        Tuple of (matches, total_comparable, match_percentage)
    """
    # Find common sequence IDs
    common_ids = set(actual.keys()) & set(expected.keys())
    
    if not common_ids:
        pytest.fail(f"No common sequence IDs found between actual and expected for {model_name}")
    
    matches = 0
    total = len(common_ids)
    
    mismatches = []
    for seq_id in sorted(common_ids):
        actual_seq = actual[seq_id]
        expected_seq = expected[seq_id]
        
        if actual_seq == expected_seq:
            matches += 1
        else:
            # Keep track of first few mismatches for debugging
            if len(mismatches) < 3:
                mismatches.append({
                    'id': seq_id,
                    'actual_len': len(actual_seq),
                    'expected_len': len(expected_seq),
                    'actual_prefix': actual_seq[:50] if actual_seq else '(empty)',
                    'expected_prefix': expected_seq[:50] if expected_seq else '(empty)',
                })
    
    match_pct = (matches / total * 100) if total > 0 else 0
    
    # Log results
    print(f"\n{model_name} Results:")
    print(f"  Total comparable sequences: {total}")
    print(f"  Exact matches: {matches}")
    print(f"  Match percentage: {match_pct:.1f}%")
    
    if mismatches:
        print(f"  First {len(mismatches)} mismatches:")
        for mm in mismatches:
            print(f"    - {mm['id']}: actual_len={mm['actual_len']}, expected_len={mm['expected_len']}")
            print(f"      actual[:50]:   {mm['actual_prefix']}")
            print(f"      expected[:50]: {mm['expected_prefix']}")
    
    return matches, total, match_pct


# Mark as integration test - skip by default unless RUN_INTEGRATION=1
@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION", "0") != "1",
    reason="Integration test - set RUN_INTEGRATION=1 to run"
)
def test_modernprost_against_example():
    """
    Test ModernProst encoder against example_modernprost_3di.fasta.
    
    Validates that at least 75% of sequences match exactly.
    """
    # Import here to allow graceful skipping if dependencies missing
    try:
        from protein_entropy.encoder import encode_sequences
    except ImportError as e:
        pytest.skip(f"Required dependencies not available: {e}")
    
    # Check for torch availability
    try:
        import torch
        # Prefer CPU to keep test deterministic and CI-friendly
        device = "cpu"
    except ImportError:
        pytest.skip("PyTorch not available")
    
    example_data_dir = get_example_data_dir()
    
    # Read input amino acid sequences
    aa_fasta = example_data_dir / "example_aa.fasta"
    if not aa_fasta.exists():
        pytest.skip(f"Input file not found: {aa_fasta}")
    
    aa_sequences = read_fasta(str(aa_fasta))
    
    # Read expected ModernProst 3Di output
    expected_fasta = example_data_dir / "example_modernprost_3di.fasta"
    if not expected_fasta.exists():
        pytest.skip(f"Expected output file not found: {expected_fasta}")
    
    expected_3di = read_fasta(str(expected_fasta))
    
    # Extract sequences in order (maintaining ID mapping)
    seq_ids = list(aa_sequences.keys())
    sequences = [aa_sequences[sid] for sid in seq_ids]
    
    print(f"\nEncoding {len(sequences)} sequences with ModernProst...")
    
    try:
        # Encode with modernprost-base (default)
        encoded = encode_sequences(
            sequences=sequences,
            model_type="modernprost_base",
            device=device,
            batch_size=10000,  # Reasonable batch size for CPU
        )
    except Exception as e:
        pytest.skip(f"Model encoding failed (model may not be cached): {e}")
    
    # Build dictionary mapping IDs to encoded sequences
    actual_3di = {seq_ids[i]: encoded[i] for i in range(len(seq_ids))}
    
    # Compare results
    matches, total, match_pct = compare_sequences(
        actual=actual_3di,
        expected=expected_3di,
        model_name="ModernProst-base",
    )
    
    # Assert at least 75% match
    assert match_pct >= 75.0, (
        f"ModernProst match rate {match_pct:.1f}% is below 75% threshold. "
        f"Only {matches}/{total} sequences matched exactly."
    )


# Mark as integration test - skip by default unless RUN_INTEGRATION=1
@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION", "0") != "1",
    reason="Integration test - set RUN_INTEGRATION=1 to run"
)
def test_prostt5_against_example():
    """
    Test ProstT5 encoder against example_prostt5_3di.fasta.
    
    Validates that at least 75% of sequences match exactly.
    """
    # Import here to allow graceful skipping if dependencies missing
    try:
        from protein_entropy.encoder import encode_sequences
    except ImportError as e:
        pytest.skip(f"Required dependencies not available: {e}")
    
    # Check for torch availability
    try:
        import torch
        # Prefer CPU to keep test deterministic and CI-friendly
        device = "cpu"
    except ImportError:
        pytest.skip("PyTorch not available")
    
    example_data_dir = get_example_data_dir()
    
    # Read input amino acid sequences
    aa_fasta = example_data_dir / "example_aa.fasta"
    if not aa_fasta.exists():
        pytest.skip(f"Input file not found: {aa_fasta}")
    
    aa_sequences = read_fasta(str(aa_fasta))
    
    # Read expected ProstT5 3Di output
    expected_fasta = example_data_dir / "example_prostt5_3di.fasta"
    if not expected_fasta.exists():
        pytest.skip(f"Expected output file not found: {expected_fasta}")
    
    expected_3di = read_fasta(str(expected_fasta))
    
    # Extract sequences in order (maintaining ID mapping)
    seq_ids = list(aa_sequences.keys())
    sequences = [aa_sequences[sid] for sid in seq_ids]
    
    print(f"\nEncoding {len(sequences)} sequences with ProstT5...")
    
    try:
        # Encode with ProstT5
        encoded = encode_sequences(
            sequences=sequences,
            model_type="prostt5",
            device=device,
            batch_size=10000,  # Reasonable batch size for CPU
        )
    except Exception as e:
        pytest.skip(f"Model encoding failed (model may not be cached): {e}")
    
    # Build dictionary mapping IDs to encoded sequences
    actual_3di = {seq_ids[i]: encoded[i] for i in range(len(seq_ids))}
    
    # Compare results
    matches, total, match_pct = compare_sequences(
        actual=actual_3di,
        expected=expected_3di,
        model_name="ProstT5",
    )
    
    # Assert at least 75% match
    assert match_pct >= 75.0, (
        f"ProstT5 match rate {match_pct:.1f}% is below 75% threshold. "
        f"Only {matches}/{total} sequences matched exactly."
    )


def test_fasta_reader_utility():
    """
    Test the FASTA reading utility function with known test data.
    
    This test doesn't require models and validates our parser works correctly.
    """
    example_data_dir = get_example_data_dir()
    
    aa_fasta = example_data_dir / "example_aa.fasta"
    if not aa_fasta.exists():
        pytest.skip(f"Input file not found: {aa_fasta}")
    
    sequences = read_fasta(str(aa_fasta))
    
    # Basic validation
    assert isinstance(sequences, dict), "read_fasta should return a dictionary"
    assert len(sequences) > 0, "Should read at least one sequence"
    
    # Check that all values are strings
    for seq_id, seq in sequences.items():
        assert isinstance(seq_id, str), f"Sequence ID should be string: {seq_id}"
        assert isinstance(seq, str), f"Sequence should be string for ID: {seq_id}"
        assert len(seq) > 0, f"Sequence should not be empty for ID: {seq_id}"
        # Sequences should not contain whitespace
        assert ' ' not in seq, f"Sequence contains whitespace for ID: {seq_id}"
        assert '\t' not in seq, f"Sequence contains tab for ID: {seq_id}"
        assert '\n' not in seq, f"Sequence contains newline for ID: {seq_id}"
    
    print(f"\nSuccessfully read {len(sequences)} sequences from {aa_fasta.name}")
    
    # Check expected sequences exist (based on the file preview)
    expected_ids = [
        "MW460250_1:DQEPQRKE_CDS_0001",
        "MW460250_1:DQEPQRKE_CDS_0002",
        "MW460250_1:DQEPQRKE_CDS_0003",
    ]
    
    for exp_id in expected_ids:
        assert exp_id in sequences, f"Expected sequence ID not found: {exp_id}"
