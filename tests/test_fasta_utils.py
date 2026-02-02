"""
Tests for FASTA utilities.
"""

import tempfile
from pathlib import Path

import pytest


def test_generate_random_protein():
    """Test random protein generation."""
    from protein_entropy.fasta_utils import generate_random_protein
    
    # Test basic generation
    seq = generate_random_protein(100)
    assert len(seq) == 100
    
    # All characters should be valid amino acids
    amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    assert all(aa in amino_acids for aa in seq)
    
    # Test reproducibility with seed
    seq1 = generate_random_protein(50, seed=42)
    seq2 = generate_random_protein(50, seed=42)
    assert seq1 == seq2


def test_read_fasta(tmp_path):
    """Test reading FASTA files."""
    from protein_entropy.fasta_utils import read_fasta
    
    # Create a test FASTA file
    fasta_file = tmp_path / "test.fasta"
    fasta_file.write_text(">seq1\nACDEFG\n>seq2\nhiklmn\n")
    
    sequences = list(read_fasta(str(fasta_file)))
    
    assert len(sequences) == 2
    assert sequences[0] == ("seq1", "ACDEFG")
    assert sequences[1] == ("seq2", "HIKLMN")  # Should be uppercase


def test_read_fasta_not_found():
    """Test reading non-existent FASTA file."""
    from protein_entropy.fasta_utils import read_fasta
    
    with pytest.raises(FileNotFoundError):
        list(read_fasta("/nonexistent/file.fasta"))


def test_write_fasta(tmp_path):
    """Test writing FASTA files."""
    from protein_entropy.fasta_utils import write_fasta, read_fasta
    
    output_file = tmp_path / "output.fasta"
    sequences = [
        ("seq1", "ACDEFG"),
        ("seq2", "HIKLMN"),
    ]
    
    write_fasta(str(output_file), sequences)
    
    # Read it back
    read_sequences = list(read_fasta(str(output_file)))
    assert len(read_sequences) == 2
    assert read_sequences[0][0] == "seq1"
    assert read_sequences[1][0] == "seq2"


def test_write_tsv(tmp_path):
    """Test writing TSV files."""
    from protein_entropy.fasta_utils import write_tsv
    
    output_file = tmp_path / "output.tsv"
    data = [
        ("seq1", 1.234567, 2.345678),
        ("seq2", 3.456789, 4.567890),
    ]
    
    write_tsv(str(output_file), data)
    
    # Read it back
    content = output_file.read_text()
    lines = content.strip().split("\n")
    
    assert len(lines) == 3  # Header + 2 data lines
    assert lines[0] == "sequence_id\taa_entropy\t3di_entropy"
    assert "seq1\t1.234567\t2.345678" in lines[1]
    assert "seq2\t3.456789\t4.567890" in lines[2]
