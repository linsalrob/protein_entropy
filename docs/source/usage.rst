Usage Guide
===========

Command Line Interface
----------------------

The ``protein_entropy`` CLI provides several subcommands:

Download Command
~~~~~~~~~~~~~~~~

Download models from HuggingFace::

    protein_entropy download [prostt5_fp16|modernprost|all]

Options:

* ``--cache-dir DIR``: Specify cache directory
* ``--force``: Force re-download
* ``--list``: List available and downloaded models

Encode3Di Command
~~~~~~~~~~~~~~~~~

Encode protein sequences to 3Di::

    protein_entropy encode3di -i INPUT.fasta -o OUTPUT.fasta [OPTIONS]

Options:

* ``-m, --model``: Model to use (prostt5, prostt5_fp16, modernprost)
* ``--model-path PATH``: Custom model path
* ``--device``: Device (cuda, mps, cpu)
* ``--batch-size N``: Maximum tokens per batch

Entropy Command
~~~~~~~~~~~~~~~

Calculate entropy for sequences::

    protein_entropy entropy -p PROTEINS.fasta -t 3DI.fasta -o OUTPUT.tsv

Run Command
~~~~~~~~~~~

Run complete pipeline::

    protein_entropy run -i INPUT.fasta -o PREFIX [OPTIONS]

Generates:

* ``PREFIX_3di.fasta``: 3Di sequences
* ``PREFIX_entropy.tsv``: Entropy data

Estimate Command
~~~~~~~~~~~~~~~~

Estimate optimal GPU batch size::

    protein_entropy estimate -m MODEL [OPTIONS]

Options:

* ``--start N``: Starting length (default: 5000)
* ``--end N``: Ending length (default: 50000)
* ``--step N``: Step size (default: 5000)
* ``--trials N``: Trials per length (default: 3)

Global Options
~~~~~~~~~~~~~~

All commands support:

* ``--log-level LEVEL``: Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
* ``--log-file FILE``: Write logs to file
* ``--version``: Show version

Input Format
------------

FASTA format with protein sequences::

    >sequence_id_1
    MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK
    >sequence_id_2
    ACDEFGHIKLMNPQRSTVWY

Sequences are automatically converted to uppercase.

Output Formats
--------------

3Di FASTA Output
~~~~~~~~~~~~~~~~

Lowercase 3Di tokens::

    >sequence_id_1
    mkaahdghaklghealghdalgheklahgdlaghdealghklaghdalghealk
    >sequence_id_2
    acdefghiklmnpqrstvwy

Entropy TSV Output
~~~~~~~~~~~~~~~~~~

Tab-separated values::

    sequence_id    aa_entropy    3di_entropy
    seq1           3.456789      2.345678
    seq2           4.123456      3.234567

Python API
----------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    from protein_entropy.encoder import encode_sequences
    from protein_entropy.entropy import calculate_entropy
    from protein_entropy.fasta_utils import read_fasta

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

    # Calculate entropy
    for seq_id, protein_seq, encoded_seq in zip(seq_ids, sequences, encoded):
        protein_ent = calculate_entropy(protein_seq)
        encoded_ent = calculate_entropy(encoded_seq)
        print(f"{seq_id}: {protein_ent:.4f} -> {encoded_ent:.4f}")

Advanced Usage
~~~~~~~~~~~~~~

Custom device and model path:

.. code-block:: python

    from protein_entropy.device import get_device, get_gpu_memory_info
    from protein_entropy.encoder import ProstT5Encoder

    # Auto-detect device
    device = get_device()
    print(f"Using device: {device}")

    # Check GPU memory
    if device == "cuda":
        mem_info = get_gpu_memory_info()
        print(f"Free GPU memory: {mem_info['free'] / 1e9:.2f} GB")

    # Use custom encoder
    encoder = ProstT5Encoder(model_path="/path/to/model", device=device)
    encoded = encoder.encode(["ACDEFGHIKLM"])

GPU Optimization
----------------

Estimating Batch Size
~~~~~~~~~~~~~~~~~~~~~

Use the estimate command to find optimal batch size::

    protein_entropy estimate -m prostt5 --start 5000 --end 50000

The tool will:

1. Generate random sequences of increasing length
2. Attempt to encode them
3. Detect OutOfMemoryError to find the limit
4. Recommend a safe batch size (80% of max)

Manual Optimization
~~~~~~~~~~~~~~~~~~~

Start with a conservative batch size and increase::

    # Start small
    protein_entropy run -i large.fasta -o output --batch-size 5000

    # Increase if no OOM errors
    protein_entropy run -i large.fasta -o output --batch-size 10000

Models
------

ProstT5_fp16
~~~~~~~~~~~~

* **Repository**: Rostlab/ProstT5_fp16
* **Type**: T5-based encoder
* **Input**: Amino acid sequences
* **Output**: 3Di structural tokens
* **Recommended for**: General use, good balance of speed and accuracy

ModernProst-profiles
~~~~~~~~~~~~~~~~~~~~

* **Repository**: gbouras13/modernprost-profiles
* **Type**: Masked language model
* **Implementation**: Based on phold
* **Recommended for**: When profile information is important

Troubleshooting
---------------

Out of Memory Errors
~~~~~~~~~~~~~~~~~~~~

If you encounter CUDA OOM errors:

1. Reduce batch size: ``--batch-size 2500``
2. Use estimate command to find optimal size
3. Process sequences in smaller files
4. Use CPU if GPU memory is insufficient

Model Download Issues
~~~~~~~~~~~~~~~~~~~~~

If model download fails:

1. Check internet connection
2. Verify HuggingFace Hub access
3. Try with ``--force`` flag
4. Set custom cache directory: ``--cache-dir /path/to/cache``

Import Errors
~~~~~~~~~~~~~

If you get PyTorch import errors:

1. Install PyTorch separately: ``pip install torch``
2. For GPU: Use appropriate CUDA version
3. For AMD GPUs: Install ROCm version separately
