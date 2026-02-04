"""
3Di encoding using ProstT5 and modernprost models.
"""

from typing import List, Optional

from .logging_config import get_logger

logger = get_logger(__name__)


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

            # Check if model is cached locally
            from .downloader import is_model_cached

            is_local = is_model_cached(model_name)
            logger.debug(f"Model cached locally: {is_local}")

            # Set loading parameters based on cache status
            load_kwargs = {}
            if is_local:
                load_kwargs["local_files_only"] = True
                logger.debug("Using local_files_only=True")

            self.tokenizer = T5Tokenizer.from_pretrained(
                model_name,
                do_lower_case=False,
                **load_kwargs,
            )

            self.model = T5EncoderModel.from_pretrained(model_name, **load_kwargs)

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

            # Check if model is cached locally
            from .downloader import is_model_cached

            is_local = is_model_cached(model_name)
            logger.debug(f"Model cached locally: {is_local}")

            # Set loading parameters based on cache status
            load_kwargs = {"trust_remote_code": True}
            if is_local:
                load_kwargs["local_files_only"] = True
                logger.debug("Using local_files_only=True")

            # Load config first with trust_remote_code to handle custom configurations
            config = AutoConfig.from_pretrained(model_name, **load_kwargs)

            if hasattr(config, "reference_compile"):
                config.reference_compile = False
                logger.debug("Set reference_compile=False in model config")

            self.tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)
            self.model = AutoModel.from_pretrained(model_name, config=config, **load_kwargs)

            # Move to device
            if self.device != "cpu":
                self.model = self.model.to(self.device)

            # ModernProst models use half precision only on CUDA
            # CPU and MPS may not support half precision properly
            if self.device.startswith("cuda"):
                self.model = self.model.half()
            
            self.model.eval()

            logger.info(f"ModernProst model loaded on {self.device}")

        except ImportError as e:
            logger.error("PyTorch or transformers not installed")
            raise RuntimeError(
                "PyTorch is required for encoding. Install with: pip install torch transformers"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load ModernProst model: {e}")
            raise RuntimeError(
                f"Failed to load ModernProst model from {model_path or 'gbouras13/modernprost-base'}: {e}"
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

        # Validate that model and tokenizer are loaded
        if self.tokenizer is None:
            raise RuntimeError(
                "Tokenizer not initialized. Model loading may have failed. "
                "Check logs for details or try re-downloading the model."
            )
        if self.model is None:
            raise RuntimeError(
                "Model not initialized. Model loading may have failed. "
                "Check logs for details or try re-downloading the model."
            )
        if self.device is None:
            raise RuntimeError(
                "Device not initialized. "
                "Check logs for details or try re-initialising the GPU."
            )
        

        logger.debug(f"Encoding {len(sequences)} sequences with ModernProst")

        three_di_sequences = []

        # Validate input sequences before processing
        # Filter out empty, None, or invalid sequences with detailed logging
        valid_sequences = []
        invalid_indices = []

        for i, seq in enumerate(sequences):
            seq = seq.replace("U", "X").replace("Z", "X").replace("O", "X")
            if seq is None:
                logger.warning(f"Sequence at index {i} is None, skipping")
                invalid_indices.append(i)
            elif not isinstance(seq, str):
                logger.warning(f"Sequence at index {i} has invalid type {type(seq)}, skipping")
                invalid_indices.append(i)
            elif len(seq) == 0:
                logger.warning(f"Sequence at index {i} is empty, skipping")
                invalid_indices.append(i)
            elif seq.isspace():
                logger.warning(f"Sequence at index {i} contains only whitespace, skipping")
                invalid_indices.append(i)
            else:
                valid_sequences.append(seq)

        if invalid_indices:
            logger.warning(
                f"Filtered out {len(invalid_indices)} invalid sequences "
                f"at indices: {invalid_indices[:10]}"
                + (f" and {len(invalid_indices) - 10} more..." if len(invalid_indices) > 10 else "")
            )

        if not valid_sequences:
            logger.error("All sequences were invalid or empty after filtering")
            return []

        # Process sequences in batches for efficiency
        # Add spaces between amino acids for all sequences
        spaced_sequences = []
        for seq in valid_sequences:
            spaced = " ".join(list(seq))
            # Double-check that spacing didn't create problematic output
            if not spaced or spaced.isspace():
                logger.error(f"Spaced sequence became empty/whitespace: original='{seq[:50]}...'")
                spaced_sequences.append("A")  # Use fallback single amino acid
            else:
                spaced_sequences.append(spaced)

        logger.debug(f"Processing {len(spaced_sequences)} valid sequences")
        logger.debug(f"Sample spaced sequences (first 3):\n{spaced_sequences[:3]}")
        
        # Tokenize all sequences at once with padding
        try:
            logger.debug("Tokenizing sequences")
            inputs = self.tokenizer(
                spaced_sequences,
                padding="longest",
                truncation=False,
                return_tensors="pt",
                add_special_tokens=True,  # Include BOS/EOS tokens as expected by the model
            )


            # Validate tokenizer outputs
            logger.debug(f"Tokenizer output keys: {inputs.keys()}")
            for key, value in inputs.items():
                if value is None:
                    raise ValueError(f"Tokenizer returned None for key '{key}'")
                logger.debug(f"  {key}: shape={getattr(value, 'shape', 'N/A')}, type={type(value)}")

        except Exception as e:
            logger.error(f"Tokenization failed 1: {e}")
            logger.error(f"Input sequences count: {len(spaced_sequences)}")
            logger.error(f"Sample sequences: {spaced_sequences[:3]}")
            raise RuntimeError(
                f"Tokenization failed with error: {e}. "
                "This may indicate invalid input sequences or tokenizer configuration issues."
            ) from e

        # Move to device
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get predictions for all sequences
        with torch.no_grad():
            outputs = self.model(**inputs)

            # Handle different model output structures
            # Some models return ModelOutput with .logits attribute
            # Others return tuples or tensors directly
            if hasattr(outputs, "logits") and outputs.logits is not None:
                logits = outputs.logits
                logger.debug("Using outputs.logits")
            elif isinstance(outputs, tuple) and len(outputs) > 0:
                logits = outputs[0]
                logger.debug("Using outputs[0] from tuple")
            elif isinstance(outputs, torch.Tensor):
                logits = outputs
                logger.debug("Using direct tensor output")
            else:
                # Provide detailed error information
                output_attrs = [attr for attr in dir(outputs) if not attr.startswith("_")]
                logger.error(f"Unexpected model output type: {type(outputs)}")
                logger.error(f"Output attributes: {output_attrs}")

                # Check if outputs has other common attributes
                for attr_name in ["last_hidden_state", "hidden_states", "pooler_output"]:
                    if hasattr(outputs, attr_name):
                        attr_val = getattr(outputs, attr_name)
                        logger.error(
                            f"  {attr_name}: {type(attr_val)}, shape: {attr_val.shape if hasattr(attr_val, 'shape') else 'N/A'}"
                        )

                raise ValueError(
                    f"Model output does not contain logits or recognizable structure. "
                    f"Output type: {type(outputs)}, "
                    f"Available attributes: {output_attrs}"
                )

        # Get predicted token IDs
        predicted_token_ids = torch.argmax(logits, dim=-1)
        
        logger.debug(f"Logits shape: {logits.shape}")
        logger.debug(f"Predicted token IDs shape: {predicted_token_ids.shape}")
        logger.debug(f"Sample predicted IDs (first sequence, first 20): {predicted_token_ids[0][:20].tolist() if len(predicted_token_ids[0]) > 0 else []}")
        
        # Debug: Check what tokens these IDs map to
        if len(predicted_token_ids) > 0 and len(predicted_token_ids[0]) > 0:
            sample_ids = predicted_token_ids[0][:min(20, len(predicted_token_ids[0]))].tolist()
            sample_tokens = self.tokenizer.convert_ids_to_tokens(sample_ids)
            logger.debug(f"Sample tokens from IDs: {sample_tokens}")
            logger.debug(f"Special tokens: {self.tokenizer.all_special_tokens}")

        # Process each valid sequence's predictions
        valid_results = []
        for i in range(len(valid_sequences)):
            # Get the sequence tokens for this sequence
            seq_tokens = predicted_token_ids[i]

            # Convert token IDs to 3Di alphabet
            # The _tokens_to_3di method will filter out special tokens
            three_di = self._tokens_to_3di(seq_tokens)
            # Note: Keep uppercase as expected by Foldseek and standard 3Di format
            valid_results.append(three_di)

        # Reconstruct results with empty strings for invalid sequences
        result_idx = 0
        for i in range(len(sequences)):
            if i in invalid_indices:
                three_di_sequences.append("")  # Empty string for invalid sequences
            else:
                three_di_sequences.append(valid_results[result_idx])
                result_idx += 1

        logger.debug(f"Encoded {len(three_di_sequences)} sequences ({len(valid_results)} valid)")
        return three_di_sequences

    def _tokens_to_3di(self, token_ids) -> str:
        """
        Convert predicted token IDs to 3Di alphabet characters.

        ModernProst predicts 3Di structural states. The output logits correspond
        to the 3Di alphabet, NOT the amino acid alphabet used for input.
        
        3Di alphabet (20 states): A C D E F G H I K L M N P Q R S T V W (Y is rare)
        Standard order used by Foldseek/3Di tools.
        
        The model outputs token IDs that need to be mapped to this alphabet.
        """
        import torch
        
        # Ensure we have a plain Python list of IDs
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()

        # 3Di alphabet in standard order (as used by Foldseek)
        # This is the target vocabulary the model predicts
        THREE_DI_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
        
        # The tokenizer vocabulary is for INPUT (amino acids)
        # But we need to check what the OUTPUT vocabulary is
        # Let's try using the tokenizer first, then fall back to direct mapping
        
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        
        logger.debug(f"_tokens_to_3di: Processing {len(tokens)} tokens")
        logger.debug(f"_tokens_to_3di: First 10 raw tokens: {tokens[:10]}")
        logger.debug(f"_tokens_to_3di: First 10 token IDs: {token_ids[:10]}")

        # Check if tokenizer vocabulary has 3Di characters
        # by examining what we got
        sample_tokens_upper = [t.upper() if isinstance(t, str) else str(t) for t in tokens[:10]]
        logger.debug(f"_tokens_to_3di: Sample tokens (uppercase): {sample_tokens_upper}")
        
        # Filter and clean tokens
        special_tokens = set(getattr(self.tokenizer, "all_special_tokens", []))
        three_di_chars = []
        
        for idx, tok in enumerate(tokens):
            # Skip special tokens (e.g., <pad>, <s>, </s>, <unk>, etc.)
            if tok in special_tokens:
                if idx < 5:  # Log first few special tokens
                    logger.debug(f"_tokens_to_3di: Skipping special token at {idx}: {tok}")
                continue
                
            # Strip any special prefixes (e.g., sentencepiece "▁" or Ġ)
            cleaned_tok = tok
            if isinstance(tok, str):
                if tok.startswith("▁"):
                    cleaned_tok = tok[1:]
                elif tok.startswith("Ġ"):
                    cleaned_tok = tok[1:]
            else:
                cleaned_tok = str(tok)
            
            # Skip empty tokens after cleaning
            if not cleaned_tok:
                continue
            
            # Convert to uppercase (3Di alphabet is uppercase)
            cleaned_tok_upper = cleaned_tok.upper()
            
            if idx < 10:  # Log first few conversions
                logger.debug(f"_tokens_to_3di: Token {idx}: '{tok}' -> cleaned: '{cleaned_tok}' -> upper: '{cleaned_tok_upper}'")
            
            # Add to result
            if len(cleaned_tok_upper) == 1:
                three_di_chars.append(cleaned_tok_upper)
            else:
                # Multi-character tokens - add each character
                three_di_chars.extend(list(cleaned_tok_upper))

        result = "".join(three_di_chars)
        logger.debug(f"_tokens_to_3di: Result length: {len(result)}, first 50 chars: {result[:50]}")
        return result


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
        batchlen = sum(len(s) for s in batch)
        logger.info(f"Encoding batch {i+1}/{len(batches)} [length: {batchlen}]")
        encoded_batch = encoder.encode(batch)
        all_encoded.extend(encoded_batch)

    return all_encoded
