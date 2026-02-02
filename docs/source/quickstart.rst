Quick Start
===========

This guide will help you get started with protein_entropy quickly.

1. Download Models
------------------

First, download the required models::

    protein_entropy download prostt5_fp16

2. Encode Proteins
------------------

Encode your protein sequences to 3Di::

    protein_entropy encode3di -i proteins.fasta -o output_3di.fasta

3. Calculate Entropy
--------------------

Calculate Shannon entropy for both protein and 3Di sequences::

    protein_entropy entropy -p proteins.fasta -t output_3di.fasta -o entropy.tsv

4. Run Complete Pipeline
-------------------------

Or run everything in one command::

    protein_entropy run -i proteins.fasta -o results

This will generate:

* ``results_3di.fasta``: 3Di encoded sequences
* ``results_entropy.tsv``: Entropy values

5. Optimize for Your GPU
-------------------------

Find the optimal batch size for your GPU::

    protein_entropy estimate -m prostt5

Example Output
--------------

Entropy TSV file::

    sequence_id    aa_entropy    3di_entropy
    seq1           3.456789      2.345678
    seq2           4.123456      3.234567

Next Steps
----------

* Read the :doc:`usage` guide for detailed information
* Explore the :doc:`api` documentation for Python integration
* Check out :doc:`examples` for more use cases
