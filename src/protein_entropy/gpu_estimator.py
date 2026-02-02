"""
GPU memory estimation tool for optimal batch sizing.
"""

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def estimate_max_sequence_length(
    model_type: str = "prostt5",
    start_length: int = 5000,
    end_length: int = 50000,
    step: int = 5000,
    num_trials: int = 3,
    model_path: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[int, list]:
    """
    Estimate maximum sequence length that can be encoded without OOM error.

    This function generates random protein sequences of increasing length and
    attempts to encode them. When it encounters a CUDA OutOfMemoryError,
    it knows the limit has been reached.

    Args:
        model_type: Model to test ('prostt5' or 'modernprost')
        start_length: Starting sequence length in amino acids
        end_length: Ending sequence length in amino acids
        step: Step size for increasing length
        num_trials: Number of trials per length
        model_path: Optional model path
        device: Device to use (auto-detected if None)

    Returns:
        Tuple of (max_length, results_list) where results_list contains
        dictionaries with length and success status
    """
    from .device import clear_gpu_memory, get_device
    from .encoder import encode_sequences
    from .fasta_utils import generate_random_protein

    if device is None:
        device = get_device()

    if device == "cpu":
        logger.warning("Running on CPU - memory estimation may not be meaningful")

    logger.info(f"Estimating max sequence length for {model_type} on {device}")
    logger.info(f"Testing lengths from {start_length} to {end_length} in steps of {step}")

    results = []
    max_successful_length = 0

    current_length = start_length

    while current_length <= end_length:
        logger.info(f"Testing length: {current_length}")

        success_count = 0

        for trial in range(num_trials):
            # Clear GPU memory before each trial
            clear_gpu_memory()

            # Generate random protein sequence
            sequence = generate_random_protein(current_length, seed=trial)

            try:
                # Try to encode the sequence
                encoded = encode_sequences(
                    sequences=[sequence],
                    model_type=model_type,
                    model_path=model_path,
                    device=device,
                    batch_size=current_length,
                )

                if encoded:
                    success_count += 1
                    logger.debug(f"Trial {trial + 1}/{num_trials}: Success")
                else:
                    logger.debug(f"Trial {trial + 1}/{num_trials}: Failed (empty result)")

            except Exception as e:
                # Check if it's an OOM error
                error_msg = str(e).lower()
                if "out of memory" in error_msg or "oom" in error_msg:
                    logger.info(
                        f"Trial {trial + 1}/{num_trials}: Out of memory at length {current_length}"
                    )
                    break
                else:
                    logger.warning(f"Trial {trial + 1}/{num_trials}: Unexpected error: {e}")
                    # Continue with other trials

        # Record results
        result = {
            "length": current_length,
            "successful_trials": success_count,
            "total_trials": num_trials,
            "success_rate": success_count / num_trials,
        }
        results.append(result)

        # If we got OOM errors, we've found the limit
        if success_count == 0:
            logger.info(f"Found memory limit at length {current_length}")
            break

        if success_count == num_trials:
            max_successful_length = current_length

        current_length += step

    logger.info(f"Maximum successful length: {max_successful_length}")
    logger.info("Estimation complete")

    return max_successful_length, results


def print_estimation_report(max_length: int, results: list) -> None:
    """
    Print a formatted report of the estimation results.

    Args:
        max_length: Maximum successful length
        results: List of result dictionaries
    """
    print("\n" + "=" * 60)
    print("GPU Memory Estimation Report")
    print("=" * 60)
    print(f"\nMaximum Successful Length: {max_length:,} amino acids")
    print(f"\nRecommended batch size: {max_length * 0.8:.0f} amino acids")
    print("\nDetailed Results:")
    print("-" * 60)
    print(f"{'Length':<15} {'Successful':<15} {'Total':<15} {'Success Rate':<15}")
    print("-" * 60)

    for result in results:
        print(
            f"{result['length']:<15,} "
            f"{result['successful_trials']:<15} "
            f"{result['total_trials']:<15} "
            f"{result['success_rate']:<15.1%}"
        )

    print("=" * 60)
    print("\nNote: Use the recommended batch size (80% of max) for production runs")
    print("to account for model overhead and variations in sequence length.")
    print("=" * 60 + "\n")
