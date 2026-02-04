#!/usr/bin/env python3
"""
Basic example: Encode proteins and calculate entropy.

This example demonstrates:
1. Reading protein sequences from FASTA
2. Encoding to 3Di using ProstT5
3. Calculating Shannon entropy
4. Writing results
"""

from protein_entropy.encoder import encode_sequences
from protein_entropy.entropy import calculate_entropy
from protein_entropy.fasta_utils import read_fasta, write_fasta, write_tsv


def main():
    # Input file (use the test data)
    import os
    import sys

    # Try to locate the test data as a package resource, with a fallback to the
    # original source-tree path for development environments.
    try:
        from importlib import resources

        test_data_resource = (
            resources.files("protein_entropy") / "data" / "test_proteins.fasta"
        )
        with resources.as_file(test_data_resource) as resource_path:
            input_file = str(resource_path)
    except Exception:
        # Fallback to development path
        package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        input_file = os.path.join(
            package_dir, "src", "protein_entropy", "data", "test_proteins.fasta"
        )

    if not os.path.exists(input_file):
        print(f"Error: Test data not found at {input_file}")
        print("Please ensure protein_entropy is installed correctly.")
        sys.exit(1)
    
    print(f"Reading sequences from: {input_file}")
    
    # Read sequences
    sequences_data = list(read_fasta(input_file))
    seq_ids = [sid for sid, _ in sequences_data]
    sequences = [seq for _, seq in sequences_data]
    
    print(f"Loaded {len(sequences)} sequences")
    
    # Encode to 3Di
    print("\nEncoding to 3Di (this may take a moment)...")
    try:
        encoded = encode_sequences(
            sequences=sequences,
            model_type="prostt5",
            device="cpu",  # Use CPU for this example
            batch_size=1000,
        )
    except Exception as e:
        print(f"\nError during encoding: {e}")
        print("\nNote: This example requires PyTorch and transformers.")
        print("Install with: pip install torch transformers")
        sys.exit(1)
    
    # Calculate entropies
    print("\nCalculating entropies...")
    protein_entropies = [calculate_entropy(seq) for seq in sequences]
    three_di_entropies = [calculate_entropy(seq) for seq in encoded]
    
    # Display results
    print("\n" + "="*70)
    print("Results")
    print("="*70)
    print(f"{'ID':<15} {'AA Entropy':<15} {'3Di Entropy':<15} {'Reduction':<15}")
    print("-"*70)
    
    for seq_id, aa_ent, di_ent in zip(seq_ids, protein_entropies, three_di_entropies):
        reduction = (aa_ent - di_ent) / aa_ent * 100 if aa_ent > 0 else 0
        print(f"{seq_id:<15} {aa_ent:<15.4f} {di_ent:<15.4f} {reduction:<15.1f}%")
    
    print("="*70)
    
    # Write outputs
    output_3di = "example_output_3di.fasta"
    output_tsv = "example_output_entropy.tsv"
    
    print(f"\nWriting 3Di sequences to: {output_3di}")
    write_fasta(output_3di, list(zip(seq_ids, encoded)))
    
    print(f"Writing entropy data to: {output_tsv}")
    results = list(zip(seq_ids, protein_entropies, three_di_entropies))
    write_tsv(output_tsv, results)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
