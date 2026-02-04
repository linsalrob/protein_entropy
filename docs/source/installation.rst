Installation
============

Requirements
------------

* Python >= 3.12
* PyTorch (optional for GPU support, required for model inference)

Basic Installation
------------------

Install from PyPI::

    pip install protein_entropy

With GPU Support
----------------

For CUDA 11.8::

    pip install protein_entropy
    pip install torch --index-url https://download.pytorch.org/whl/cu118

For CUDA 12.1::

    pip install protein_entropy
    pip install torch --index-url https://download.pytorch.org/whl/cu121

For CPU only::

    pip install protein_entropy torch

Development Installation
------------------------

Clone the repository and install in development mode::

    git clone https://github.com/linsalrob/protein_entropy.git
    cd protein_entropy
    pip install -e ".[dev]"

Verifying Installation
----------------------

Check the version::

    protein_entropy --version

Run tests::

    pytest
