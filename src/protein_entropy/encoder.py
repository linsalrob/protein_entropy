"""
3Di encoding using ModernProst and ProstT5 models.

The decoding logic follows the working implementation in genome_entropy
(https://github.com/linsalrob/genome_entropy, src/genome_entropy/encode3di),
which is itself adapted from phold
(https://github.com/gbouras13/phold/blob/main/src/phold/features/predict_3Di.py).
"""

import inspect
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, List, Optional

from .logging_config import get_logger

logger = get_logger(__name__)

# ModernProst classifier heads emit class indices into these alphabets.
THREE_DI_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
THREE_DI_ALPHABET_SIZE = 20


def token_budget_batches(
    sequences: List[str],
    max_tokens: int = 5000,
) -> List[List[str]]:
    """
    Batch sequences based on token budget.

    Groups sequences into batches where the total number of tokens
    (amino acids) doesn't exceed max_tokens.

    Args:
        sequences: List of protein sequences
        max_tokens: Maximum tokens per batch

    Returns:
        List of sequence batches
    """
    batches = []
    current_batch = []
    current_tokens = 0

    for seq in sequences:
        seq_len = len(seq)

        # If adding this sequence would exceed budget, start new batch
        if current_tokens + seq_len > max_tokens and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0

        current_batch.append(seq)
        current_tokens += seq_len

    # Add remaining sequences
    if current_batch:
        batches.append(current_batch)

    logger.debug(f"Created {len(batches)} batches with max {max_tokens} tokens")
    return batches


def _replace_rare_residues(seq: str) -> str:
    """Replace rare/ambiguous amino acids by X (3Di has no equivalent)."""
    return re.sub(r"[UZOB]", "X", seq.upper())


def _from_pretrained(loader: Any, model_name: str, **kwargs: Any) -> Any:
    """
    Call ``loader.from_pretrained``, using only local files when the model is cached.

    Both attempts use the cache directory that ``download_model`` and
    ``is_model_cached`` use, so models downloaded to PROTEIN_ENTROPY_CACHE are found.
    A cache can be partial (e.g. weights present but the remote-code modules
    missing), so a failed cache-only load is retried with network access.
    """
    from .downloader import get_cache_dir, is_model_cached

    cache_dir = str(get_cache_dir())
    if is_model_cached(model_name, cache_dir):
        try:
            return loader.from_pretrained(
                model_name, cache_dir=cache_dir, local_files_only=True, **kwargs
            )
        except OSError as e:
            logger.warning(f"Cached copy of {model_name} is incomplete ({e}); retrying online")
    return loader.from_pretrained(model_name, cache_dir=cache_dir, **kwargs)


def _trust_remote_code(model_name: str) -> bool:
    """
    Only run Hub remote code for the ModernProst repositories we know about.

    A local directory is trusted because the user chose to put it there; any other
    Hub repository ID is refused rather than silently executing its code.
    """
    from .downloader import AVAILABLE_MODELS

    modernprost_repos = {
        repo for name, repo in AVAILABLE_MODELS.items() if name.startswith("modernprost")
    }
    if model_name in modernprost_repos or Path(model_name).is_dir():
        return True
    raise ValueError(
        f"Refusing to run remote code from {model_name!r}: ModernProst models need "
        "trust_remote_code. Use one of the known ModernProst models "
        f"({', '.join(sorted(modernprost_repos))}) or download the "
        "repository and pass its local directory as --model-path."
    )


class ProstT5Encoder:
    """
    Encoder using the ProstT5 sequence-to-sequence model.

    ProstT5 translates amino acids to 3Di with its decoder (``generate``), so the
    full seq2seq model is required; the encoder alone has no 3Di head.
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        """
        Initialize ProstT5 encoder.

        Args:
            model_path: HF model ID or local path (default: Rostlab/ProstT5_fp16)
            device: Device to use ('cuda', 'mps', or 'cpu')
        """
        self.device = device
        self.model_name = model_path if model_path else "Rostlab/ProstT5_fp16"
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the model and tokenizer."""
        try:
            from transformers import AutoModelForSeq2SeqLM, T5Tokenizer
        except ImportError as e:
            logger.error("PyTorch or transformers not installed")
            raise RuntimeError(
                "PyTorch is required for encoding. Install with: pip install torch transformers"
            ) from e

        logger.info(f"Loading ProstT5 model from {self.model_name}")

        self.tokenizer = _from_pretrained(T5Tokenizer, self.model_name, do_lower_case=False)
        self.model = _from_pretrained(AutoModelForSeq2SeqLM, self.model_name).to(self.device)

        # Only GPUs support half precision; use full precision on CPU/MPS
        self.model = self.model.half() if self.device.startswith("cuda") else self.model.float()
        self.model.eval()

        logger.info(f"Model loaded successfully on {self.device}")

    def encode(self, sequences: List[str]) -> List[str]:
        """
        Encode protein sequences to 3Di.

        Args:
            sequences: List of protein sequences

        Returns:
            List of 3Di encodings (uppercase), one per input sequence
        """
        import torch

        if not sequences:
            return []

        logger.debug(f"Encoding {len(sequences)} sequences with ProstT5")

        # Space-separate residues and prepend the AA->3Di task prefix
        cleaned = [_replace_rare_residues(seq) for seq in sequences]
        processed = ["<AA2fold> " + " ".join(seq) for seq in cleaned]

        min_len = min(len(s) for s in cleaned)
        max_len = max(len(s) for s in cleaned)

        ids = self.tokenizer(
            processed,
            add_special_tokens=True,
            padding="longest",
            return_tensors="pt",
        ).to(self.device)

        # Mask the trailing </s> token so it is not treated as a residue.
        # Position = 1 (<AA2fold>) + len(seq).
        for idx, seq in enumerate(cleaned):
            mask_position = len(seq) + 1
            if mask_position < ids.attention_mask.shape[1]:
                ids.attention_mask[idx, mask_position] = 0

        # Generation settings recommended on the ProstT5 model card for AA->3Di
        gen_kwargs = {
            "do_sample": True,
            "num_beams": 3,
            "top_p": 0.95,
            "temperature": 1.2,
            "top_k": 6,
            "repetition_penalty": 1.2,
        }

        with torch.no_grad():
            translations = self.model.generate(
                ids.input_ids,
                attention_mask=ids.attention_mask,
                # max_length would count the decoder-start token, leaving room for
                # only len - 1 residues; budget the new tokens (residues + </s>).
                max_new_tokens=max_len + 1,
                min_new_tokens=min_len,
                early_stopping=True,
                num_return_sequences=1,
                **gen_kwargs,
            )

        decoded = self.tokenizer.batch_decode(translations, skip_special_tokens=True)

        # ProstT5 emits lowercase 3Di; normalise to the uppercase Foldseek alphabet
        # and trim to the input length.
        three_di_sequences = [
            "".join(ts.split()).upper()[: len(seq)] for ts, seq in zip(decoded, cleaned)
        ]

        short = [
            i for i, (ts, seq) in enumerate(zip(three_di_sequences, cleaned)) if len(ts) < len(seq)
        ]
        if short:
            logger.warning(
                f"ProstT5 generated fewer 3Di tokens than residues for {len(short)} of "
                f"{len(cleaned)} sequences in this batch (batch indices {short[:10]}); "
                "their 3Di entropy covers only the generated tokens"
            )

        logger.debug(f"Encoded {len(three_di_sequences)} sequences")
        return three_di_sequences


class ModernProstEncoder:
    """
    Encoder using ModernProst models (gbouras13/modernprost-*).

    ModernProst is a residue-level classifier: each residue position gets logits
    over the 20 3Di states, which are decoded by argmax into THREE_DI_ALPHABET.
    The multitask models also emit a 12-state head, which is ignored here.
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        """
        Initialize ModernProst encoder.

        Args:
            model_path: HF model ID or local path (default: gbouras13/modernprost-50M)
            device: Device to use
        """
        self.device = device
        self.model_name = model_path if model_path else "gbouras13/modernprost-50M"
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the model and tokenizer."""
        try:
            from transformers import AutoConfig, AutoModel, AutoTokenizer
        except ImportError as e:
            logger.error("PyTorch or transformers not installed")
            raise RuntimeError(
                "PyTorch is required for encoding. Install with: pip install torch transformers"
            ) from e

        logger.info(f"Loading ModernProst model from {self.model_name}")
        trust = _trust_remote_code(self.model_name)
        try:
            config = _from_pretrained(AutoConfig, self.model_name, trust_remote_code=trust)

            # The multitask remote code imports its configuration module without a
            # relative prefix, so its cache directory must be importable.
            module_dir = str(Path(inspect.getfile(config.__class__)).parent)
            if module_dir not in sys.path:
                sys.path.append(module_dir)

            # ModernBert's compiled MLP is not needed for inference
            if hasattr(config, "reference_compile"):
                config.reference_compile = False

            self.tokenizer = _from_pretrained(
                AutoTokenizer, self.model_name, trust_remote_code=trust
            )
            self.model = _from_pretrained(
                AutoModel, self.model_name, config=config, trust_remote_code=trust
            ).to(self.device)
        except Exception as e:
            logger.error(f"Failed to load ModernProst model {self.model_name}: {e}")
            raise RuntimeError(f"Failed to load ModernProst model {self.model_name}: {e}") from e

        # Half precision only on CUDA; CPU and MPS may not support it properly
        if self.device.startswith("cuda"):
            self.model = self.model.half()
        self.model.eval()

        logger.info(f"ModernProst model loaded on {self.device}")

    def _three_di_logits(self, logits: Any) -> Any:
        """Select the 3Di head from the model output."""
        if isinstance(logits, Mapping):
            if "3di" not in logits:
                raise RuntimeError(
                    f"{self.model_name} output has no '3di' head (heads: {sorted(logits)})"
                )
            return logits["3di"]
        return logits

    def encode(self, sequences: List[str]) -> List[str]:
        """
        Encode protein sequences to 3Di using ModernProst.

        Args:
            sequences: List of protein sequences

        Returns:
            List of 3Di encodings (uppercase), one per input sequence
        """
        import torch

        if not sequences:
            return []

        logger.debug(f"Encoding {len(sequences)} sequences with ModernProst")

        cleaned = [_replace_rare_residues(seq) for seq in sequences]

        # ModernProst tokenises raw residues: no spaces, no special tokens, no prefix
        inputs = self.tokenizer(
            cleaned,
            padding="longest",
            truncation=False,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = self._three_di_logits(outputs.logits)
        attention_mask = inputs["attention_mask"]

        if logits.ndim != 3 or logits.shape[-1] != THREE_DI_ALPHABET_SIZE:
            raise RuntimeError(
                f"{self.model_name} 3Di logits must have shape B x L x "
                f"{THREE_DI_ALPHABET_SIZE}; got {tuple(logits.shape)}"
            )
        if logits.shape[:2] != attention_mask.shape:
            raise RuntimeError(
                f"{self.model_name} logits shape {tuple(logits.shape[:2])} does not "
                f"match attention mask {tuple(attention_mask.shape)}"
            )

        predictions = logits.argmax(dim=-1).cpu()
        masks = attention_mask.cpu().bool()

        three_di_sequences = []
        for row, mask, seq in zip(predictions, masks, cleaned):
            class_ids = row[mask].tolist()
            if len(class_ids) != len(seq):
                raise RuntimeError(
                    f"{self.model_name} tokenizer produced {len(class_ids)} residue "
                    f"positions for a protein of length {len(seq)}"
                )
            three_di_sequences.append("".join(THREE_DI_ALPHABET[i] for i in class_ids))

        logger.debug(f"Encoded {len(three_di_sequences)} sequences")
        return three_di_sequences


def encode_sequences(
    sequences: List[str],
    model_type: str = "modernprost",
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    batch_size: int = 5000,
) -> List[str]:
    """
    Encode protein sequences to 3Di.

    Args:
        sequences: List of protein sequences
        model_type: A model name from downloader.AVAILABLE_MODELS
        model_path: Optional HF model ID or local path, overriding model_type's default
        device: Device to use (auto-detected if None)
        batch_size: Maximum tokens per batch

    Returns:
        List of 3Di encoded sequences
    """
    from .device import get_device
    from .downloader import AVAILABLE_MODELS

    if device is None:
        device = get_device()

    key = model_type.lower().replace("-", "_")
    if key not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available models: {', '.join(AVAILABLE_MODELS)}"
        )
    model_name = model_path if model_path else AVAILABLE_MODELS[key]

    if key.startswith("prostt5"):
        encoder = ProstT5Encoder(model_path=model_name, device=device)
    else:
        encoder = ModernProstEncoder(model_path=model_name, device=device)

    # Batch sequences
    batches = token_budget_batches(sequences, max_tokens=batch_size)

    # Encode batches
    all_encoded = []
    for i, batch in enumerate(batches):
        batchlen = sum(len(s) for s in batch)
        logger.info(f"Encoding batch {i+1}/{len(batches)} [length: {batchlen}]")
        all_encoded.extend(encoder.encode(batch))

    return all_encoded
