"""Whisper tokenizer implementation for audio transcription.

This module provides tokenization functionality for the Whisper model,
including special tokens, language tokens, and text encoding/decoding.
"""

import json
import re
from typing import List, Tuple, Dict, Optional
from functools import lru_cache

@lru_cache()
def bytes_to_unicode():
    """
    Returns mapping from byte values to unicode strings.
    This is needed for byte-level BPE as used in GPT-2.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

@lru_cache()
def unicode_to_bytes():
    """
    Returns reverse mapping from unicode strings to byte values.
    """
    return {v: k for k, v in bytes_to_unicode().items()}

# Special token IDs
EOT_TOKEN = 50257
SOT_TOKEN = 50258
TRANSCRIBE_TOKEN = 50359
TRANSLATE_TOKEN = 50358
NO_TIMESTAMPS_TOKEN = 50363
SOT_PREV_TOKEN = 50361
SOT_LM_TOKEN = 50360
TIMESTAMP_BEGIN = 50364

# Language codes to token ID mapping
LANGUAGE_TO_TOKEN = {
    "en": 50259, "zh": 50260, "de": 50261, "es": 50262, "ru": 50263,
    "ko": 50264, "fr": 50265, "ja": 50266, "pt": 50267, "tr": 50268,
    "pl": 50269, "ca": 50270, "nl": 50271, "ar": 50272, "sv": 50273,
    "it": 50274, "id": 50275, "hi": 50276, "fi": 50277, "vi": 50278,
    "he": 50279, "uk": 50280, "el": 50281, "ms": 50282, "cs": 50283,
    "ro": 50284, "da": 50285, "hu": 50286, "ta": 50287, "no": 50288,
    "th": 50289, "ur": 50290, "hr": 50291, "bg": 50292, "lt": 50293,
    "la": 50294, "mi": 50295, "ml": 50296, "cy": 50297, "sk": 50298,
    "te": 50299, "fa": 50300, "lv": 50301, "bn": 50302, "sr": 50303,
    "az": 50304, "sl": 50305, "kn": 50306, "et": 50307, "mk": 50308,
    "br": 50309, "eu": 50310, "is": 50311, "hy": 50312, "ne": 50313,
    "mn": 50314, "bs": 50315, "kk": 50316, "sq": 50317, "sw": 50318,
    "gl": 50319, "mr": 50320, "pa": 50321, "si": 50322, "km": 50323,
    "sn": 50324, "yo": 50325, "so": 50326, "af": 50327, "oc": 50328,
    "ka": 50329, "be": 50330, "tg": 50331, "sd": 50332, "gu": 50333,
    "am": 50334, "yi": 50335, "lo": 50336, "uz": 50337, "fo": 50338,
    "ht": 50339, "ps": 50340, "tk": 50341, "nn": 50342, "mt": 50343,
    "sa": 50344, "lb": 50345, "my": 50346, "bo": 50347, "tl": 50348,
    "mg": 50349, "as": 50350, "tt": 50351, "haw": 50352, "ln": 50353,
    "ha": 50354, "ba": 50355, "jw": 50356, "su": 50357
}


class WhisperTokenizer:
    """Whisper tokenizer for encoding and decoding text."""

    def __init__(self, vocab_file: str = None, multilingual: bool = True,
                 task: str = "transcribe", language: str = "en"):
        """Initialize the tokenizer.

        Args:
            vocab_file: Path to vocabulary JSON file
            multilingual: Whether this is a multilingual model
            task: Task type ('transcribe' or 'translate')
            language: Language code (e.g., 'en', 'ar')
        """
        self.multilingual = multilingual
        self.task = task
        self.language = language
        self.vocab_to_id: Dict[str, int] = {}
        self.id_to_vocab: Dict[int, str] = {}
        self.language_tokens: Dict[str, int] = {}
        self.non_speech_tokens_cache: Optional[List[int]] = None

        if vocab_file:
            self.load_vocab_from_file(vocab_file)
        else:
            self.initialize_builtin_vocab()

        self.initialize_special_tokens()
        self.initialize_language_tokens()

    def load_vocab_from_file(self, vocab_file: str) -> bool:
        """Load vocabulary from JSON file.

        Args:
            vocab_file: Path to vocabulary file

        Returns:
            True if successful, False otherwise
        """
        try:
            with open(vocab_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse JSON
            data = json.loads(content)

            # Check if it's HuggingFace tokenizers format
            if 'model' in data and 'vocab' in data['model']:
                vocab = data['model']['vocab']
                print(f"Loading HuggingFace tokenizer format with {len(vocab)} tokens")

                for token, token_id in vocab.items():
                    self.vocab_to_id[token] = token_id
                    self.id_to_vocab[token_id] = token

            # Or if it's a simple array of tokens
            elif isinstance(data, list):
                for token_id, token in enumerate(data):
                    self.vocab_to_id[token] = token_id
                    self.id_to_vocab[token_id] = token
            else:
                print("Unknown tokenizer format")
                return False

            print(f"Loaded {len(self.vocab_to_id)} tokens from vocabulary file")
            return True

        except Exception as e:
            print(f"Error loading vocabulary: {e}")
            return False

    def initialize_builtin_vocab(self):
        """Initialize with basic built-in vocabulary."""
        # Add basic ASCII characters
        for i in range(256):
            token = chr(i)
            self.vocab_to_id[token] = i
            self.id_to_vocab[i] = token

        # Add common English words
        common_tokens = [
            " the", " and", " to", " of", " a", " in", " is", " it", " you", " that",
            " he", " was", " for", " on", " are", " as", " with", " his", " they",
            " I", " at", " be", " this", " have", " from"
        ]

        token_id = 256
        for token in common_tokens:
            self.vocab_to_id[token] = token_id
            self.id_to_vocab[token_id] = token
            token_id += 1

    def initialize_special_tokens(self):
        """Initialize special tokens."""
        special_tokens = {
            "<|endoftext|>": EOT_TOKEN,
            "<|startoftranscript|>": SOT_TOKEN,
            "<|transcribe|>": TRANSCRIBE_TOKEN,
            "<|translate|>": TRANSLATE_TOKEN,
            "<|notimestamps|>": NO_TIMESTAMPS_TOKEN,
            "<|startofprev|>": SOT_PREV_TOKEN,
            "<|startoflm|>": SOT_LM_TOKEN
        }

        for token, token_id in special_tokens.items():
            self.vocab_to_id[token] = token_id
            self.id_to_vocab[token_id] = token

        # Add timestamp tokens
        for i in range(1500):
            token_id = TIMESTAMP_BEGIN + i
            seconds = i * 0.02
            token_str = f"<|{seconds:.2f}|>"
            self.vocab_to_id[token_str] = token_id
            self.id_to_vocab[token_id] = token_str

    def initialize_language_tokens(self):
        """Initialize language tokens."""
        for lang_code, token_id in LANGUAGE_TO_TOKEN.items():
            token_str = f"<|{lang_code}|>"
            self.vocab_to_id[token_str] = token_id
            self.id_to_vocab[token_id] = token_str
            self.language_tokens[lang_code] = token_id

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encode text to token IDs.

        Args:
            text: Text to encode
            add_special_tokens: Whether to add special tokens

        Returns:
            List of token IDs
        """
        if not text:
            return []

        normalized = self.normalize_text(text)
        tokens = self.tokenize_text(normalized)

        token_ids = []
        for token in tokens:
            if token in self.vocab_to_id:
                token_ids.append(self.vocab_to_id[token])
            else:
                # Split unknown tokens into characters
                for c in token:
                    if c in self.vocab_to_id:
                        token_ids.append(self.vocab_to_id[c])

        return token_ids

    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text.

        Args:
            tokens: List of token IDs
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text
        """
        raw_bpe = ""

        for token_id in tokens:
            if token_id in self.id_to_vocab:
                token = self.id_to_vocab[token_id]

                # Skip special tokens if requested
                if skip_special_tokens:
                    if (len(token) >= 4 and token[0:2] == "<|" and
                        token[-2:] == "|>"):
                        continue

                raw_bpe += token

        # Decode BPE to proper text
        result = self.decode_bpe(raw_bpe)
        return result

    def decode_bpe(self, raw_bpe: str) -> str:
        """Decode BPE tokens to text.

        Args:
            raw_bpe: Raw BPE string

        Returns:
            Decoded text
        """
        # Get the unicode-to-bytes mapping
        byte_decoder = unicode_to_bytes()

        # Convert unicode characters back to bytes
        byte_list = []
        for char in raw_bpe:
            if char in byte_decoder:
                byte_list.append(byte_decoder[char])
            else:
                # If character not in mapping, use its byte value directly
                byte_list.append(ord(char))

        # Convert bytes to string
        try:
            text = bytearray(byte_list).decode('utf-8', errors='replace')
        except Exception:
            text = raw_bpe

        # Replace BPE space token with regular space
        text = text.replace('\u0120', ' ')

        return text

    def token_to_id(self, token: str) -> int:
        """Convert token to ID.

        Args:
            token: Token string

        Returns:
            Token ID or -1 if not found
        """
        return self.vocab_to_id.get(token, -1)

    def id_to_token(self, token_id: int) -> str:
        """Convert ID to token.

        Args:
            token_id: Token ID

        Returns:
            Token string or empty string if not found
        """
        return self.id_to_vocab.get(token_id, "")

    def get_language_token(self, language_code: str) -> int:
        """Get language token ID.

        Args:
            language_code: Language code (e.g., 'en', 'ar')

        Returns:
            Language token ID or -1 if not found
        """
        return self.language_tokens.get(language_code, -1)

    def get_sot_sequence(self, language_code: str = "", task: str = "transcribe") -> List[int]:
        """Get start-of-transcription sequence.

        Args:
            language_code: Language code
            task: Task type ('transcribe' or 'translate')

        Returns:
            List of token IDs for SOT sequence
        """
        sequence = [SOT_TOKEN]

        if self.multilingual and language_code:
            lang_token = self.get_language_token(language_code)
            if lang_token != -1:
                sequence.append(lang_token)

        if task == "transcribe":
            sequence.append(TRANSCRIBE_TOKEN)
        elif task == "translate":
            sequence.append(TRANSLATE_TOKEN)

        return sequence

    def get_non_speech_tokens(self) -> List[int]:
        """Get non-speech token IDs.

        Returns:
            List of non-speech token IDs
        """
        if self.non_speech_tokens_cache is None:
            tokens = set()

            # Punctuation and symbols
            symbols = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
            for c in symbols:
                token_id = self.token_to_id(c)
                if token_id != -1:
                    tokens.add(token_id)

                spaced_id = self.token_to_id(" " + c)
                if spaced_id != -1:
                    tokens.add(spaced_id)

            self.non_speech_tokens_cache = list(tokens)

        return self.non_speech_tokens_cache

    def is_timestamp_token(self, token_id: int) -> bool:
        """Check if token is a timestamp.

        Args:
            token_id: Token ID

        Returns:
            True if timestamp token
        """
        return TIMESTAMP_BEGIN <= token_id < TIMESTAMP_BEGIN + 1500

    def timestamp_to_seconds(self, token_id: int) -> float:
        """Convert timestamp token to seconds.

        Args:
            token_id: Timestamp token ID

        Returns:
            Time in seconds or -1.0 if not a timestamp
        """
        if not self.is_timestamp_token(token_id):
            return -1.0
        return (token_id - TIMESTAMP_BEGIN) * 0.02

    def seconds_to_timestamp(self, seconds: float) -> int:
        """Convert seconds to timestamp token.

        Args:
            seconds: Time in seconds

        Returns:
            Timestamp token ID
        """
        offset = int(seconds / 0.02)
        return TIMESTAMP_BEGIN + offset

    def split_to_word_tokens(self, tokens: List[int]) -> Tuple[List[str], List[List[int]]]:
        """Split tokens into words.

        Args:
            tokens: List of token IDs

        Returns:
            Tuple of (words, word_token_lists)
        """
        words = []
        word_tokens = []

        current_word_tokens = []
        current_word = ""

        for token_id in tokens:
            if token_id >= EOT_TOKEN:
                # Special token - finish current word
                if current_word_tokens:
                    words.append(current_word)
                    word_tokens.append(current_word_tokens)
                    current_word = ""
                    current_word_tokens = []
                continue

            token_str = self.id_to_token(token_id)
            if not token_str:
                continue

            current_word_tokens.append(token_id)
            current_word += token_str

            # Check if word is complete
            if token_str and (token_str[-1] == ' ' or token_str[-1] in ".,!?;:"):
                words.append(current_word)
                word_tokens.append(current_word_tokens)
                current_word = ""
                current_word_tokens = []

        # Add final word
        if current_word_tokens:
            words.append(current_word)
            word_tokens.append(current_word_tokens)

        return words, word_tokens

    def normalize_text(self, text: str) -> str:
        """Normalize text.

        Args:
            text: Input text

        Returns:
            Normalized text
        """
        normalized = text.lower()
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = normalized.strip()
        return normalized

    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into subwords.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        # Simple whitespace tokenization
        tokens = []
        words = text.split()

        for i, word in enumerate(words):
            if i > 0:
                tokens.append(" " + word)
            else:
                tokens.append(word)

        return tokens

    # Accessor methods for special tokens
    def get_sot_token(self) -> int:
        return SOT_TOKEN

    def get_eot_token(self) -> int:
        return EOT_TOKEN

    def get_transcribe_token(self) -> int:
        return TRANSCRIBE_TOKEN

    def get_translate_token(self) -> int:
        return TRANSLATE_TOKEN

    def get_no_timestamps_token(self) -> int:
        return NO_TIMESTAMPS_TOKEN

    def get_sot_prev_token(self) -> int:
        return SOT_PREV_TOKEN

    def get_sot_lm_token(self) -> int:
        return SOT_LM_TOKEN

    def get_timestamp_begin(self) -> int:
        return TIMESTAMP_BEGIN

    def is_multilingual(self) -> bool:
        return self.multilingual

    # Properties for compatibility with Tokenizer class
    @property
    def non_speech_tokens(self) -> List[int]:
        """Property to access non-speech tokens (for compatibility)."""
        return self.get_non_speech_tokens()

    @property
    def transcribe(self) -> int:
        """Property to access transcribe token ID (for compatibility)."""
        return TRANSCRIBE_TOKEN

    @property
    def translate(self) -> int:
        """Property to access translate token ID (for compatibility)."""
        return TRANSLATE_TOKEN

    @property
    def sot(self) -> int:
        """Property to access SOT token ID (for compatibility)."""
        return SOT_TOKEN

    @property
    def sot_lm(self) -> int:
        """Property to access SOT_LM token ID (for compatibility)."""
        return SOT_LM_TOKEN

    @property
    def sot_prev(self) -> int:
        """Property to access SOT_PREV token ID (for compatibility)."""
        return SOT_PREV_TOKEN

    @property
    def eot(self) -> int:
        """Property to access EOT token ID (for compatibility)."""
        return EOT_TOKEN

    @property
    def no_timestamps(self) -> int:
        """Property to access no_timestamps token ID (for compatibility)."""
        return NO_TIMESTAMPS_TOKEN

    @property
    def timestamp_begin(self) -> int:
        """Property to access timestamp_begin token ID (for compatibility)."""
        return TIMESTAMP_BEGIN

    @property
    def sot_sequence(self) -> List[int]:
        """Property to access SOT sequence (for compatibility)."""
        return self.get_sot_sequence(self.language, self.task)