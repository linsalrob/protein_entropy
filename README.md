# protein_entropy

[![CI](https://github.com/linsalrob/protein_entropy/workflows/CI/badge.svg)](https://github.com/linsalrob/protein_entropy/actions)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Encode proteins using transformers and calculate their entropy.

`protein_entropy` is a Python package that converts protein sequences into 3Di structural token sequences using state-of-the-art transformer models (ProstT5 and ModernProst), and calculates Shannon entropy for both original and encoded sequences.

## Features

- **3Di Encoding**: Convert protein sequences to 3Di structural tokens using:
  - ProstT5_fp16 (Rostlab)
  - ModernProst-profiles (gbouras13)
- **Entropy Calculation**: Compute Shannon entropy for protein and 3Di sequences
- **GPU Optimization**: Automatic device detection (CUDA/MPS/CPU) and batch size estimation
- **Flexible CLI**: Easy-to-use command-line interface with multiple subcommands
- **Production Ready**: Comprehensive testing, logging, and error handling

## Installation

### Basic Installation

```bash
pip install protein_entropy
```

### With PyTorch (for CUDA/GPU support)

```bash
# For CUDA 11.8
pip install protein_entropy
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install protein_entropy
pip install torch --index-url https://download.pytorch.org/whl/cu121

# For CPU only
pip install protein_entropy torch
```

### Development Installation

```bash
git clone https://github.com/linsalrob/protein_entropy.git
cd protein_entropy
pip install -e ".[dev]"
```

## Quick Start

### 1. Download Models

```bash
# Download ProstT5_fp16 model
protein_entropy download prostt5_fp16

# Download ModernProst model
protein_entropy download modernprost

# Download all models
protein_entropy download all

# List available and downloaded models
protein_entropy download prostt5_fp16 --list
```

### 2. Encode Proteins to 3Di

```bash
protein_entropy encode3di \
  -i proteins.fasta \
  -o encoded_3di.fasta \
  -m prostt5 \
  --batch-size 5000
```

### 3. Calculate Entropy

```bash
protein_entropy entropy \
  -p proteins.fasta \
  -t encoded_3di.fasta \
  -o entropy_results.tsv
```

### 4. Run Complete Pipeline

```bash
protein_entropy run \
  -i proteins.fasta \
  -o results \
  -m prostt5 \
  --batch-size 5000
```

This will generate:
- `results_3di.fasta`: 3Di encoded sequences
- `results_entropy.tsv`: Entropy values for each sequence

### 5. Estimate Optimal Batch Size

```bash
protein_entropy estimate \
  -m prostt5 \
  --start 5000 \
  --end 50000 \
  --step 5000 \
  --trials 3
```

## Usage

### Command Line Options

All commands support:
- `--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}`: Set logging level
- `--log-file FILE`: Write logs to file instead of stdout
- `--version`: Show version and exit

### Input Format

Protein sequences should be provided in FASTA format:

```
>sequence1
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK
>sequence2
ACDEFGHIKLMNPQRSTVWY
```

Sequences are automatically converted to uppercase before encoding.

### Output Format

**3Di FASTA output** (lowercase 3Di tokens):
```
>sequence1
mkaahdghaklghealghdalgheklahgdlaghdealghklaghdalghealk
>sequence2
acdefghiklmnpqrstvwy
```

**Entropy TSV output**:
```
sequence_id    aa_entropy    3di_entropy
sequence1      3.456789      2.345678
sequence2      4.321987      3.210987
```

## Python API

```python
from protein_entropy.encoder import encode_sequences
from protein_entropy.entropy import calculate_entropy
from protein_entropy.fasta_utils import read_fasta, write_fasta, write_tsv

# Read sequences
sequences_data = list(read_fasta("proteins.fasta"))
seq_ids, sequences = zip(*sequences_data)

# Encode to 3Di
encoded = encode_sequences(
    sequences=list(sequences),
    model_type="prostt5",
    device="cuda",
    batch_size=5000,
)

# Calculate entropies
protein_entropies = [calculate_entropy(seq) for seq in sequences]
three_di_entropies = [calculate_entropy(seq) for seq in encoded]

# Write outputs
write_fasta("output_3di.fasta", list(zip(seq_ids, encoded)))
write_tsv("entropy.tsv", list(zip(seq_ids, protein_entropies, three_di_entropies)))
```

## Models

### ProstT5_fp16

- **Repository**: Rostlab/ProstT5_fp16
- **Description**: T5-based model for predicting 3Di structural tokens from amino acid sequences
- **Reference**: [ProstT5 Paper](https://doi.org/10.1101/2023.07.23.550085)

### ModernProst-profiles

- **Repository**: gbouras13/modernprost-profiles
- **Description**: Modern implementation with profile-based predictions
- **Implementation**: Based on [phold](https://github.com/gbouras13/phold)

## GPU Support

The package automatically detects the best available device:
1. CUDA (NVIDIA GPUs)
2. MPS (Apple Silicon)
3. CPU (fallback)

For optimal performance on GPUs, use the `estimate` command to find the maximum batch size for your hardware.

## Testing

```bash
# Run basic tests (excludes integration tests)
pytest

# Run all tests including integration tests
pytest --run-integration

# Or set environment variable
RUN_INTEGRATION=1 pytest

# Run with coverage
pytest --cov=protein_entropy --cov-report=html
```

## Documentation

Full documentation is available at [Read the Docs](https://protein-entropy.readthedocs.io).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this software, please cite:

```bibtex
@software{protein_entropy,
  author = {Edwards, Rob},
  title = {protein_entropy: Encode proteins and calculate entropy},
  year = {2024},
  url = {https://github.com/linsalrob/protein_entropy}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- ProstT5 model by Rostlab
- ModernProst implementation by George Bouras
- 3Di structural alphabet from Foldseek

## Support

For questions and issues, please use the [GitHub issue tracker](https://github.com/linsalrob/protein_entropy/issues).
