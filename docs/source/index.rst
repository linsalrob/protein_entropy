protein_entropy Documentation
==============================

Welcome to protein_entropy's documentation!

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   usage
   api
   examples

Overview
--------

``protein_entropy`` is a Python package that converts protein sequences into 3Di structural token sequences using state-of-the-art transformer models (ProstT5 and ModernProst), and calculates Shannon entropy for both original and encoded sequences.

Features
--------

* 3Di Encoding using ProstT5_fp16 and ModernProst-profiles
* Shannon entropy calculation for protein and 3Di sequences
* Automatic GPU detection (CUDA/MPS/CPU)
* GPU memory estimation for optimal batch sizing
* Comprehensive CLI with multiple subcommands
* Full Python API for programmatic access

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
