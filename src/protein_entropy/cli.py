"""
Command-line interface for protein_entropy.
"""

import argparse
import logging
import sys
from typing import Optional

from . import __version__


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Configure logging.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path (if None, logs to stdout)
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure root logger
    handlers = []

    if log_file:
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        handlers.append(file_handler)
    else:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        handlers.append(console_handler)

    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True,
        format="%(asctime)s %(levelname)s %(name)s\n%(message)s"
    )


def cmd_download(args) -> int:
    """Download models and assets."""
    from .downloader import AVAILABLE_MODELS, download_model, list_downloaded_models

    logger = logging.getLogger(__name__)

    if args.list:
        print("Available models:")
        for name, repo in AVAILABLE_MODELS.items():
            print(f"  {name}: {repo}")

        print("\nDownloaded models:")
        downloaded = list_downloaded_models(args.cache_dir)
        if downloaded:
            for name in downloaded:
                print(f"  {name}")
        else:
            print("  (none)")
        return 0

    if args.model == "all":
        models_to_download = list(AVAILABLE_MODELS.keys())
    else:
        # Validate model name before downloading
        if args.model not in AVAILABLE_MODELS:
            logger.error(
                f"Unknown model: {args.model}. "
                f"Available models: {', '.join(AVAILABLE_MODELS.keys())}"
            )
            return 1
        models_to_download = [args.model]

    for model_name in models_to_download:
        try:
            download_model(
                model_name=model_name,
                cache_dir=args.cache_dir,
                force_download=args.force,
            )
            logger.info(f"Successfully downloaded {model_name}")
        except Exception as e:
            logger.error(f"Failed to download {model_name}: {e}")
            return 1

    return 0


def cmd_encode3di(args) -> int:
    """Encode proteins to 3Di."""
    from .encoder import encode_sequences
    from .fasta_utils import read_fasta, write_fasta

    logger = logging.getLogger(__name__)

    logger.info(f"Reading input from: {args.input}")

    # Read sequences with error handling
    try:
        sequences_data = list(read_fasta(args.input))
    except FileNotFoundError:
        logger.error(f"Input file not found: {args.input}")
        return 1
    except PermissionError:
        logger.error(f"Permission denied reading file: {args.input}")
        return 1
    except Exception as e:
        logger.error(f"Error reading input file: {e}")
        return 1

    seq_ids = [sid for sid, _ in sequences_data]
    sequences = [seq for _, seq in sequences_data]

    logger.info(f"Loaded {len(sequences)} sequences")

    # Encode
    logger.info(f"Encoding with model: {args.model}")
    try:
        encoded = encode_sequences(
            sequences=sequences,
            model_type=args.model,
            model_path=args.model_path,
            device=args.device,
            batch_size=args.batch_size,
        )
    except Exception as e:
        logger.exception(f"Encoding failed", exc_info=True)
        raise

    # Write output
    output_data = list(zip(seq_ids, encoded))
    write_fasta(args.output, output_data)

    logger.info(f"Wrote {len(encoded)} sequences to: {args.output}")
    return 0


def cmd_entropy(args) -> int:
    """Calculate entropy for sequences."""
    from .entropy import calculate_batch_entropy
    from .fasta_utils import read_fasta, write_tsv

    logger = logging.getLogger(__name__)

    logger.info(f"Reading protein sequences from: {args.protein}")
    try:
        protein_data = list(read_fasta(args.protein))
    except FileNotFoundError:
        logger.error(f"Protein file not found: {args.protein}")
        return 1
    except PermissionError:
        logger.error(f"Permission denied reading file: {args.protein}")
        return 1
    except Exception as e:
        logger.error(f"Error reading protein file: {e}")
        return 1

    seq_ids = [sid for sid, _ in protein_data]
    protein_seqs = [seq for _, seq in protein_data]

    logger.info(f"Reading 3Di sequences from: {args.three_di}")
    try:
        three_di_data = list(read_fasta(args.three_di))
    except FileNotFoundError:
        logger.error(f"3Di file not found: {args.three_di}")
        return 1
    except PermissionError:
        logger.error(f"Permission denied reading file: {args.three_di}")
        return 1
    except Exception as e:
        logger.error(f"Error reading 3Di file: {e}")
        return 1

    three_di_ids = [sid for sid, _ in three_di_data]
    three_di_seqs = [seq for _, seq in three_di_data]

    # Verify lengths match
    if len(protein_seqs) != len(three_di_seqs):
        logger.error(
            f"Mismatch: {len(protein_seqs)} protein sequences "
            f"but {len(three_di_seqs)} 3Di sequences"
        )
        return 1

    # Check if sequence IDs match
    if seq_ids != three_di_ids:
        logger.warning(
            "Sequence IDs do not match between protein and 3Di files. "
            "Pairing sequences by position."
        )
        # Log first few mismatches for debugging
        for i, (pid, did) in enumerate(zip(seq_ids[:5], three_di_ids[:5])):
            if pid != did:
                logger.warning(f"Position {i}: protein ID '{pid}' != 3Di ID '{did}'")

    logger.info(f"Calculating entropy for {len(protein_seqs)} sequence pairs")

    # Calculate entropies
    protein_entropies = calculate_batch_entropy(protein_seqs)
    three_di_entropies = calculate_batch_entropy(three_di_seqs)

    # Combine results
    results = list(zip(seq_ids, protein_entropies, three_di_entropies))

    # Write output
    write_tsv(args.output, results)

    logger.info(f"Wrote entropy data to: {args.output}")
    return 0


def cmd_run(args) -> int:
    """Run the complete pipeline."""
    from .encoder import encode_sequences
    from .entropy import calculate_batch_entropy
    from .fasta_utils import read_fasta, write_fasta, write_tsv

    logger = logging.getLogger(__name__)

    logger.info("Running complete pipeline")

    # Step 1: Read input
    logger.info(f"Reading input from: {args.input}")
    sequences_data = list(read_fasta(args.input))
    seq_ids = [sid for sid, _ in sequences_data]
    sequences = [seq for _, seq in sequences_data]

    logger.info(f"Loaded {len(sequences)} sequences")

    # Step 2: Encode to 3Di
    logger.info(f"Encoding with model: {args.model}")
    try:
        encoded = encode_sequences(
            sequences=sequences,
            model_type=args.model,
            model_path=args.model_path,
            device=args.device,
            batch_size=args.batch_size,
        )
    except Exception as e:
        logger.exception(f"Encoding failed", exc_info=True)
        raise

    # Step 3: Calculate entropies
    logger.info("Calculating entropies")
    protein_entropies = calculate_batch_entropy(sequences)
    three_di_entropies = calculate_batch_entropy(encoded)

    # Step 4: Write outputs
    # Write 3Di sequences
    three_di_output = args.output_prefix + "_3di.fasta"
    output_data = list(zip(seq_ids, encoded))
    write_fasta(three_di_output, output_data)
    logger.info(f"Wrote 3Di sequences to: {three_di_output}")

    # Write entropy data
    entropy_output = args.output_prefix + "_entropy.tsv"
    results = list(zip(seq_ids, protein_entropies, three_di_entropies))
    write_tsv(entropy_output, results)
    logger.info(f"Wrote entropy data to: {entropy_output}")

    logger.info("Pipeline complete")
    return 0


def cmd_estimate(args) -> int:
    """Estimate optimal GPU batch size."""
    from .gpu_estimator import estimate_max_sequence_length, print_estimation_report

    max_length, results = estimate_max_sequence_length(
        model_type=args.model,
        start_length=args.start,
        end_length=args.end,
        step=args.step,
        num_trials=args.trials,
        model_path=args.model_path,
        device=args.device,
    )

    print_estimation_report(max_length, results)

    return 0


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="protein_entropy",
        description="Encode proteins using transformers and calculate their entropy",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    # Create a parent parser for shared arguments that can be used after subcommands
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level (default: INFO)",
    )

    parent_parser.add_argument(
        "--log-file",
        type=str,
        help="Write logs to file instead of stdout",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Download command
    parser_download = subparsers.add_parser(
        "download",
        help="Download models and assets",
        parents=[parent_parser],
    )
    parser_download.add_argument(
        "model",
        choices=["prostt5_fp16", "modernprost", "modernprost_base", "modernprost_profiles", "all"],
        help="Model to download",
    )
    parser_download.add_argument(
        "--cache-dir",
        type=str,
        help="Cache directory for models",
    )
    parser_download.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if cached",
    )
    parser_download.add_argument(
        "--list",
        action="store_true",
        help="List available and downloaded models",
    )
    parser_download.set_defaults(func=cmd_download)

    # Encode3di command
    parser_encode = subparsers.add_parser(
        "encode3di",
        help="Encode proteins to 3Di",
        parents=[parent_parser],
    )
    parser_encode.add_argument(
        "-i",
        "--input",
        required=True,
        type=str,
        help="Input FASTA file with protein sequences",
    )
    parser_encode.add_argument(
        "-o",
        "--output",
        required=True,
        type=str,
        help="Output FASTA file with 3Di sequences",
    )
    parser_encode.add_argument(
        "-m",
        "--model",
        default="prostt5",
        choices=[
            "prostt5",
            "prostt5_fp16",
            "modernprost",
            "modernprost_base",
            "modernprost_profiles",
        ],
        help="Model to use for encoding (default: prostt5)",
    )
    parser_encode.add_argument(
        "--model-path",
        type=str,
        help="Path to model directory (optional)",
    )
    parser_encode.add_argument(
        "--device",
        type=str,
        choices=["cuda", "mps", "cpu"],
        help="Device to use (auto-detected if not specified)",
    )
    parser_encode.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Maximum tokens per batch (default: 5000)",
    )
    parser_encode.set_defaults(func=cmd_encode3di)

    # Entropy command
    parser_entropy = subparsers.add_parser(
        "entropy",
        help="Calculate entropy for sequences",
        parents=[parent_parser],
    )
    parser_entropy.add_argument(
        "-p",
        "--protein",
        required=True,
        type=str,
        help="Input FASTA file with protein sequences",
    )
    parser_entropy.add_argument(
        "-t",
        "--three-di",
        required=True,
        type=str,
        help="Input FASTA file with 3Di sequences",
    )
    parser_entropy.add_argument(
        "-o",
        "--output",
        required=True,
        type=str,
        help="Output TSV file with entropy data",
    )
    parser_entropy.set_defaults(func=cmd_entropy)

    # Run command (end-to-end pipeline)
    parser_run = subparsers.add_parser(
        "run",
        help="Run complete pipeline (encode + entropy)",
        parents=[parent_parser],
    )
    parser_run.add_argument(
        "-i",
        "--input",
        required=True,
        type=str,
        help="Input FASTA file with protein sequences",
    )
    parser_run.add_argument(
        "-o",
        "--output-prefix",
        required=True,
        type=str,
        help="Output prefix for generated files",
    )
    parser_run.add_argument(
        "-m",
        "--model",
        default="prostt5",
        choices=[
            "prostt5",
            "prostt5_fp16",
            "modernprost",
            "modernprost_base",
            "modernprost_profiles",
        ],
        help="Model to use for encoding (default: prostt5)",
    )
    parser_run.add_argument(
        "--model-path",
        type=str,
        help="Path to model directory (optional)",
    )
    parser_run.add_argument(
        "--device",
        type=str,
        choices=["cuda", "mps", "cpu"],
        help="Device to use (auto-detected if not specified)",
    )
    parser_run.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Maximum tokens per batch (default: 5000)",
    )
    parser_run.set_defaults(func=cmd_run)

    # Estimate command
    parser_estimate = subparsers.add_parser(
        "estimate",
        help="Estimate optimal GPU batch size",
        parents=[parent_parser],
    )
    parser_estimate.add_argument(
        "-m",
        "--model",
        default="prostt5",
        choices=[
            "prostt5",
            "prostt5_fp16",
            "modernprost",
            "modernprost_base",
            "modernprost_profiles",
        ],
        help="Model to test (default: prostt5)",
    )
    parser_estimate.add_argument(
        "--model-path",
        type=str,
        help="Path to model directory (optional)",
    )
    parser_estimate.add_argument(
        "--device",
        type=str,
        choices=["cuda", "mps", "cpu"],
        help="Device to use (auto-detected if not specified)",
    )
    parser_estimate.add_argument(
        "--start",
        type=int,
        default=5000,
        help="Starting sequence length (default: 5000)",
    )
    parser_estimate.add_argument(
        "--end",
        type=int,
        default=50000,
        help="Ending sequence length (default: 50000)",
    )
    parser_estimate.add_argument(
        "--step",
        type=int,
        default=5000,
        help="Step size for length increase (default: 5000)",
    )
    parser_estimate.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of trials per length (default: 3)",
    )
    parser_estimate.set_defaults(func=cmd_estimate)

    return parser


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Check if command was provided
    if not args.command:
        parser.print_help()
        return 1

    # Setup logging
    # Note: log_level is available either from global args or subcommand args
    log_level = getattr(args, "log_level", "INFO")
    log_file = getattr(args, "log_file", None)
    setup_logging(log_level, log_file)

    # Run command
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        logging.getLogger(__name__).error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
