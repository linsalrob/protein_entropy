# Examples

This directory contains example scripts demonstrating how to use protein_entropy.

## Available Examples

### basic_example.py

A simple example showing:
- Reading FASTA files
- Encoding proteins to 3Di
- Calculating Shannon entropy
- Writing results

Run with:
```bash
python basic_example.py
```

## Requirements

All examples require:
- protein_entropy installed
- PyTorch installed (`pip install torch`)
- transformers installed (included with protein_entropy)

## Running Examples

1. Install protein_entropy:
   ```bash
   pip install protein_entropy
   pip install torch
   ```

2. Download required models:
   ```bash
   protein_entropy download prostt5_fp16
   ```

3. Run an example:
   ```bash
   cd examples
   python basic_example.py
   ```

## Creating Your Own Examples

Feel free to use these examples as templates for your own projects. The basic pattern is:

1. Import required modules
2. Read input sequences
3. Encode to 3Di
4. Calculate entropy
5. Write results

See the [documentation](https://protein-entropy.readthedocs.io) for more details.
