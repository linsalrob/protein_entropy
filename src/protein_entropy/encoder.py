"""
3Di encoding using ProstT5 and modernprost models.
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


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


class ProstT5Encoder:
    """
    Encoder using ProstT5_fp16 model.
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        """
        Initialize ProstT5 encoder.

        Args:
            model_path: Path to model directory (if None, will load from HF cache)
            device: Device to use ('cuda', 'mps', or 'cpu')
        """
        self.device = device
        self.model = None
        self.tokenizer = None
        self._load_model(model_path)

    def _load_model(self, model_path: Optional[str]) -> None:
        """Load the model and tokenizer."""
        try:
            import torch  # noqa: F401
            from transformers import T5EncoderModel, T5Tokenizer

            model_name = model_path if model_path else "Rostlab/ProstT5_fp16"

            logger.info(f"Loading ProstT5 model from {model_name}")

            self.tokenizer = T5Tokenizer.from_pretrained(
                model_name,
                do_lower_case=False,
            )

            self.model = T5EncoderModel.from_pretrained(model_name)

            # Move model to device
            if self.device != "cpu":
                self.model = self.model.to(self.device)

            self.model.eval()

            logger.info(f"Model loaded successfully on {self.device}")

        except ImportError as e:
            logger.error("PyTorch or transformers not installed")
            raise RuntimeError(
                "PyTorch is required for encoding. Install with: pip install torch transformers"
            ) from e

    def encode(self, sequences: List[str]) -> List[str]:
        """
        Encode protein sequences to 3Di.

        Args:
            sequences: List of protein sequences (uppercase)

        Returns:
            List of 3Di encodings (lowercase)
        """
        import torch

        if not sequences:
            return []

        logger.debug(f"Encoding {len(sequences)} sequences with ProstT5")

        # Add space between amino acids as required by ProstT5
        spaced_sequences = [" ".join(list(seq)) for seq in sequences]

        # Tokenize
        inputs = self.tokenizer(
            spaced_sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        # Move to device
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get 3Di tokens from embeddings
        # For ProstT5, the decoder generates 3Di tokens
        # This is a simplified version - actual implementation may vary
        embeddings = outputs.last_hidden_state
        attention_mask = inputs.get("attention_mask")

        # Convert embeddings to 3Di tokens
        # ProstT5 outputs structure tokens directly
        three_di_sequences = []

        for i, seq in enumerate(sequences):
            # Determine the valid token length for this sequence
            if attention_mask is not None:
                valid_len = int(attention_mask[i].sum().item())
            else:
                # Fallback to sequence length if no attention mask is available
                valid_len = len(seq)

            # Get the embedding for this sequence using the valid token length
            seq_embedding = embeddings[i, :valid_len]

            # Convert to 3Di alphabet (simplified)
            # In practice, ProstT5 has a specific decoding mechanism
            # For now, we'll generate placeholder 3Di tokens
            # The actual 3Di alphabet is: acdefghiklmnpqrstvwy
            three_di = self._decode_3di(seq_embedding, seq_embedding.size(0))
            three_di_sequences.append(three_di.lower())

        logger.debug(f"Encoded {len(three_di_sequences)} sequences")
        return three_di_sequences

    def _decode_3di(self, embedding, seq_len: int) -> str:
        """
        Decode embeddings to 3Di tokens.

        WARNING: This is a placeholder implementation.
        ProstT5 is an encoder-only model (T5EncoderModel) and does not
        include a decoder or 3Di classification head. This method uses
        a simplified approach that does not produce accurate 3Di predictions.

        For production use, consider using ModernProstEncoder which has
        a proper prediction head, or implement a classification layer
        trained to map ProstT5 embeddings to 3Di tokens.
        """
        import torch

        # 3Di alphabet (20 states corresponding to structural conformations)
        three_di_alphabet = "acdefghiklmnpqrstvwy"

        # Simple approach: use argmax over alphabet dimension
        # This is a placeholder and will not produce accurate results
        indices = torch.argmax(embedding, dim=-1) % 20
        three_di = "".join([three_di_alphabet[idx] for idx in indices[:seq_len]])

        return three_di


class ModernProstEncoder:
    """
    Encoder using modernprost-base model (George's implementation).
    Can also use modernprost-profiles as an alternate option.

    This is adapted from:
    https://github.com/gbouras13/phold/blob/42e345c49f7768b2d79ddfc625e6dafa558aff75/src/phold/features/predict_3Di.py#L793-L916
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        """
        Initialize ModernProst encoder.

        Args:
            model_path: Path to model directory
            device: Device to use
        """
        self.device = device
        self.model = None
        self.tokenizer = None
        self._load_model(model_path)

    def _load_model(self, model_path: Optional[str]) -> None:
        """Load the model and tokenizer."""
        try:
            import torch  # noqa: F401
            from transformers import AutoConfig, AutoModel, AutoTokenizer

            model_name = model_path if model_path else "gbouras13/modernprost-base"

            logger.info(f"Loading ModernProst model from {model_name}")

            # Load config first with trust_remote_code to handle custom configurations
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            
            if hasattr(config, "reference_compile"):
                config.reference_compile = False
                logger.debug("Set reference_compile=False in model config")

            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                model_name, config=config, trust_remote_code=True
            )

            # Move to device
            if self.device != "cpu":
                self.model = self.model.to(self.device)

            self.model.eval()

            logger.info(f"ModernProst model loaded on {self.device}")

        except ImportError as e:
            logger.error("PyTorch or transformers not installed")
            raise RuntimeError(
                "PyTorch is required for encoding. Install with: pip install torch transformers"
            ) from e

    def encode(self, sequences: List[str]) -> List[str]:
        """
        Encode protein sequences to 3Di using ModernProst.

        This implements the logic from George's phold code (lines 793-916).

        Args:
            sequences: List of protein sequences

        Returns:
            List of 3Di encodings (lowercase)
        """
        import torch

        if not sequences:
            return []

        logger.debug(f"Encoding {len(sequences)} sequences with ModernProst")

        three_di_sequences = []

        # Process sequences in batches for efficiency
        # Add spaces between amino acids for all sequences
        spaced_sequences = [" ".join(list(seq)) for seq in sequences]

        # Tokenize all sequences at once with padding
        inputs = self.tokenizer(
            spaced_sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        # Move to device
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get predictions for all sequences
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Get predicted token IDs
        predicted_token_ids = torch.argmax(logits, dim=-1)

        # Process each sequence's predictions
        for i in range(len(sequences)):
            # Get the sequence tokens for this sequence
            seq_tokens = predicted_token_ids[i]

            # Convert token IDs to 3Di alphabet
            # The _tokens_to_3di method will filter out special tokens
            three_di = self._tokens_to_3di(seq_tokens)
            three_di_sequences.append(three_di.lower())

        logger.debug(f"Encoded {len(three_di_sequences)} sequences")
        return three_di_sequences

    def _tokens_to_3di(self, token_ids) -> str:
        """
        Convert token IDs to 3Di alphabet using the tokenizer vocabulary.

        This uses the tokenizer's ID-to-token mapping so that the predicted
        token IDs are correctly mapped to 3Di symbols.
        """
        # Ensure we have a plain Python list of IDs
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()

        # Convert IDs to tokens using the tokenizer vocabulary
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)

        # Collect non-special 3Di tokens
        special_tokens = set(getattr(self.tokenizer, "all_special_tokens", []))
        three_di_tokens = []
        for tok in tokens:
            # Skip special tokens (e.g., <pad>, <s>, </s>, etc.)
            if tok in special_tokens:
                continue
            # Strip common subword prefixes (e.g., sentencepiece "▁")
            if tok.startswith("▁"):
                tok = tok[1:]
            # Skip empty tokens after stripping
            if not tok:
                continue
            three_di_tokens.append(tok)

        # Join tokens to form the final 3Di sequence
        return "".join(three_di_tokens)


def encode_sequences(
    sequences: List[str],
    model_type: str = "prostt5",
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    batch_size: int = 5000,
) -> List[str]:
    """
    Encode protein sequences to 3Di.

    Args:
        sequences: List of protein sequences
        model_type: Model to use ('prostt5', 'prostt5_fp16', 'modernprost', 'modernprost_base', or 'modernprost_profiles')
        model_path: Optional path to model
        device: Device to use (auto-detected if None)
        batch_size: Maximum tokens per batch

    Returns:
        List of 3Di encoded sequences
    """
    from .device import get_device

    if device is None:
        device = get_device()

    # Select encoder
    if model_type.lower() in ["prostt5", "prostt5_fp16"]:
        encoder = ProstT5Encoder(model_path=model_path, device=device)
    elif model_type.lower() in [
        "modernprost",
        "modernprost-profiles",
        "modernprost_profiles",
        "modernprost-base",
        "modernprost_base",
    ]:
        encoder = ModernProstEncoder(model_path=model_path, device=device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Batch sequences
    batches = token_budget_batches(sequences, max_tokens=batch_size)

    # Encode batches
    all_encoded = []
    for i, batch in enumerate(batches):
        logger.info(f"Encoding batch {i+1}/{len(batches)}")
        encoded_batch = encoder.encode(batch)
        all_encoded.extend(encoded_batch)

    return all_encoded
