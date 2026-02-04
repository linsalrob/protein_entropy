Examples
========

This page contains practical examples for using protein_entropy.

Basic Examples
--------------

Example 1: Simple Encoding
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Encode a small set of proteins:

.. code-block:: python

    from protein_entropy.encoder import encode_sequences
    from protein_entropy.fasta_utils import read_fasta, write_fasta

    # Read input
    sequences_data = list(read_fasta("proteins.fasta"))
    seq_ids, sequences = zip(*sequences_data)

    # Encode
    encoded = encode_sequences(
        sequences=list(sequences),
        model_type="prostt5",
    )

    # Write output
    write_fasta("output_3di.fasta", list(zip(seq_ids, encoded)))

Example 2: Calculate Entropy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calculate and compare entropies:

.. code-block:: python

    from protein_entropy.entropy import calculate_entropy
    from protein_entropy.fasta_utils import read_fasta

    # Read sequences
    proteins = list(read_fasta("proteins.fasta"))
    three_di = list(read_fasta("3di.fasta"))

    # Calculate and compare
    for (pid, prot), (did, di) in zip(proteins, three_di):
        prot_ent = calculate_entropy(prot)
        di_ent = calculate_entropy(di)
        reduction = (prot_ent - di_ent) / prot_ent * 100
        print(f"{pid}: {prot_ent:.3f} -> {di_ent:.3f} ({reduction:.1f}% reduction)")

Example 3: Batch Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Process large files efficiently:

.. code-block:: python

    from protein_entropy.encoder import token_budget_batches, ProstT5Encoder
    from protein_entropy.fasta_utils import read_fasta

    # Read all sequences
    sequences_data = list(read_fasta("large_proteins.fasta"))
    seq_ids, sequences = zip(*sequences_data)

    # Create encoder
    encoder = ProstT5Encoder(device="cuda")

    # Batch sequences
    batches = token_budget_batches(list(sequences), max_tokens=10000)

    # Process batches
    all_encoded = []
    for i, batch in enumerate(batches):
        print(f"Processing batch {i+1}/{len(batches)}")
        encoded_batch = encoder.encode(batch)
        all_encoded.extend(encoded_batch)

Advanced Examples
-----------------

Example 4: Custom Device Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from protein_entropy.device import get_device, get_gpu_memory_info
    from protein_entropy.encoder import encode_sequences

    # Check device
    device = get_device()
    print(f"Using: {device}")

    if device == "cuda":
        mem_info = get_gpu_memory_info()
        free_gb = mem_info['free'] / 1e9
        print(f"Free GPU memory: {free_gb:.2f} GB")
        
        # Adjust batch size based on available memory
        if free_gb > 10:
            batch_size = 20000
        elif free_gb > 5:
            batch_size = 10000
        else:
            batch_size = 5000
    else:
        batch_size = 1000  # Smaller for CPU

    # Encode with optimal batch size
    encoded = encode_sequences(
        sequences=sequences,
        model_type="prostt5",
        device=device,
        batch_size=batch_size,
    )

Example 5: Compare Models
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from protein_entropy.encoder import encode_sequences
    from protein_entropy.entropy import calculate_entropy

    sequences = ["ACDEFGHIKLMNPQRSTVWY"]

    # Encode with both models
    prostt5_encoded = encode_sequences(sequences, model_type="prostt5")
    modern_encoded = encode_sequences(sequences, model_type="modernprost")

    # Compare
    print("Original:", sequences[0])
    print("ProstT5:", prostt5_encoded[0])
    print("ModernProst:", modern_encoded[0])
    
    # Compare entropies
    print("\nEntropy comparison:")
    print(f"Original: {calculate_entropy(sequences[0]):.3f}")
    print(f"ProstT5: {calculate_entropy(prostt5_encoded[0]):.3f}")
    print(f"ModernProst: {calculate_entropy(modern_encoded[0]):.3f}")

Example 6: Error Handling
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from protein_entropy.encoder import encode_sequences
    from protein_entropy.device import clear_gpu_memory
    import logging

    logging.basicConfig(level=logging.INFO)

    sequences = ["ACDEF" * 10000]  # Very long sequence

    try:
        encoded = encode_sequences(
            sequences=sequences,
            model_type="prostt5",
            device="cuda",
            batch_size=50000,
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("GPU OOM! Trying with smaller batch...")
            clear_gpu_memory()
            
            # Retry with smaller batch
            encoded = encode_sequences(
                sequences=sequences,
                model_type="prostt5",
                device="cuda",
                batch_size=10000,
            )
        else:
            raise

Command Line Examples
---------------------

Example 7: Pipeline Script
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a shell script for batch processing::

    #!/bin/bash
    
    # Process multiple files
    for file in data/*.fasta; do
        basename=$(basename "$file" .fasta)
        echo "Processing $basename..."
        
        protein_entropy run \
            -i "$file" \
            -o "results/$basename" \
            -m prostt5 \
            --batch-size 10000 \
            --log-file "logs/$basename.log"
    done
    
    echo "All files processed!"

Example 8: SLURM Job Script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Submit to SLURM cluster::

    #!/bin/bash
    #SBATCH --job-name=protein_entropy
    #SBATCH --output=logs/job_%j.out
    #SBATCH --error=logs/job_%j.err
    #SBATCH --time=24:00:00
    #SBATCH --gpus=1
    #SBATCH --mem=32G
    
    # Activate environment
    source activate protein_entropy_env
    
    # Estimate optimal batch size
    protein_entropy estimate \
        -m prostt5 \
        --device cuda \
        > batch_size_estimate.txt
    
    # Extract recommended batch size (80% of max)
    BATCH_SIZE=$(grep "Recommended" batch_size_estimate.txt | awk '{print $4}')
    
    # Run pipeline
    protein_entropy run \
        -i ${INPUT_FASTA} \
        -o ${OUTPUT_PREFIX} \
        -m prostt5 \
        --batch-size ${BATCH_SIZE} \
        --device cuda \
        --log-level INFO

See Also
--------

* :doc:`usage` for detailed command line usage
* :doc:`api` for complete API documentation
