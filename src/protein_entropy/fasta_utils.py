"""
FASTA file I/O utilities.
"""

import logging
from pathlib import Path
from typing import Iterator, Tuple

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

logger = logging.getLogger(__name__)


def read_fasta(fasta_path: str) -> Iterator[Tuple[str, str]]:
    """
    Read sequences from a FASTA file.

    Args:
        fasta_path: Path to FASTA file

    Yields:
        Tuple of (sequence_id, sequence) where sequence is uppercase
    """
    path = Path(fasta_path)
    if not path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    logger.info(f"Reading FASTA file: {fasta_path}")
    count = 0

    for record in SeqIO.parse(str(path), "fasta"):
        sequence = str(record.seq).upper()
        count += 1
        yield record.id, sequence

    logger.info(f"Read {count} sequences from {fasta_path}")


def write_fasta(output_path: str, sequences: list[Tuple[str, str]]) -> None:
    """
    Write sequences to a FASTA file.

    Args:
        output_path: Path to output FASTA file
        sequences: List of (sequence_id, sequence) tuples
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for seq_id, sequence in sequences:
        record = SeqRecord(Seq(sequence), id=seq_id, description="")
        records.append(record)

    with open(path, "w") as f:
        SeqIO.write(records, f, "fasta")

    logger.info(f"Wrote {len(sequences)} sequences to {output_path}")


def write_tsv(output_path: str, data: list[Tuple[str, float, float]]) -> None:
    """
    Write entropy data to a TSV file.

    Args:
        output_path: Path to output TSV file
        data: List of (sequence_id, aa_entropy, 3di_entropy) tuples
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        f.write("sequence_id\taa_entropy\t3di_entropy\n")
        for seq_id, aa_ent, di_ent in data:
            f.write(f"{seq_id}\t{aa_ent:.6f}\t{di_ent:.6f}\n")

    logger.info(f"Wrote entropy data for {len(data)} sequences to {output_path}")


def generate_random_protein(length: int, seed: int = None) -> str:
    """
    Generate a random protein sequence.

    Args:
        length: Length of the protein sequence
        seed: Random seed for reproducibility

    Returns:
        Random protein sequence
    """
    import random

    # Standard 20 amino acids
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"

    if seed is not None:
        random.seed(seed)

    return "".join(random.choice(amino_acids) for _ in range(length))
