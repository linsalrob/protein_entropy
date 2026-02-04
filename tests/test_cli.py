"""
Tests for CLI.
"""

from unittest.mock import patch

import pytest


def test_version():
    """Test version flag."""
    from protein_entropy.cli import create_parser

    parser = create_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--version"])

    # Version flag should exit with 0
    assert exc_info.value.code == 0


def test_no_command():
    """Test running with no command shows help."""
    from protein_entropy.cli import main

    with patch("sys.argv", ["protein_entropy"]):
        result = main()
        assert result == 1


def test_log_level_option():
    """Test log level option."""
    from protein_entropy.cli import create_parser

    parser = create_parser()
    # Log level arguments must come after the subcommand
    args = parser.parse_args(["download", "prostt5_fp16", "--log-level", "DEBUG"])

    assert args.log_level == "DEBUG"


def test_download_command_parsing():
    """Test download command parsing."""
    from protein_entropy.cli import create_parser

    parser = create_parser()

    # Test basic download
    args = parser.parse_args(["download", "prostt5_fp16"])
    assert args.command == "download"
    assert args.model == "prostt5_fp16"

    # Test with options
    args = parser.parse_args(["download", "modernprost", "--force", "--cache-dir", "/tmp"])
    assert args.model == "modernprost"
    assert args.force is True
    assert args.cache_dir == "/tmp"


def test_encode3di_command_parsing():
    """Test encode3di command parsing."""
    from protein_entropy.cli import create_parser

    parser = create_parser()
    args = parser.parse_args(
        [
            "encode3di",
            "-i",
            "input.fasta",
            "-o",
            "output.fasta",
            "-m",
            "prostt5",
            "--batch-size",
            "10000",
        ]
    )

    assert args.command == "encode3di"
    assert args.input == "input.fasta"
    assert args.output == "output.fasta"
    assert args.model == "prostt5"
    assert args.batch_size == 10000


def test_entropy_command_parsing():
    """Test entropy command parsing."""
    from protein_entropy.cli import create_parser

    parser = create_parser()
    args = parser.parse_args(
        [
            "entropy",
            "-p",
            "proteins.fasta",
            "-t",
            "3di.fasta",
            "-o",
            "entropy.tsv",
        ]
    )

    assert args.command == "entropy"
    assert args.protein == "proteins.fasta"
    assert args.three_di == "3di.fasta"
    assert args.output == "entropy.tsv"


def test_run_command_parsing():
    """Test run command parsing."""
    from protein_entropy.cli import create_parser

    parser = create_parser()
    args = parser.parse_args(
        [
            "run",
            "-i",
            "input.fasta",
            "-o",
            "output",
            "-m",
            "modernprost",
        ]
    )

    assert args.command == "run"
    assert args.input == "input.fasta"
    assert args.output_prefix == "output"
    assert args.model == "modernprost"


def test_estimate_command_parsing():
    """Test estimate command parsing."""
    from protein_entropy.cli import create_parser

    parser = create_parser()
    args = parser.parse_args(
        [
            "estimate",
            "-m",
            "prostt5",
            "--start",
            "1000",
            "--end",
            "10000",
            "--step",
            "1000",
            "--trials",
            "5",
        ]
    )

    assert args.command == "estimate"
    assert args.model == "prostt5"
    assert args.start == 1000
    assert args.end == 10000
    assert args.step == 1000
    assert args.trials == 5
