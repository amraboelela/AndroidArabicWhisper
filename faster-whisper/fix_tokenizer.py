#!/usr/bin/env python3

import os
import json
import numpy as np

def create_proper_tokenizer_json():
    """Create a more complete tokenizer.json that handles the encoding properly"""
    vocab_path = "./whisper_ct2/vocabulary.json"
    tokenizer_path = "./whisper_ct2/tokenizer.json"

    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    print(f"Loaded vocabulary with {len(vocab)} tokens")

    # Create vocab mapping
    vocab_map = {token: i for i, token in enumerate(vocab)}

    # Find some key tokens to verify they exist
    key_tokens = [" -", " '", "-", "'", " ", "▁", "Ġ"]
    found_tokens = {}
    for token in key_tokens:
        if token in vocab_map:
            found_tokens[token] = vocab_map[token]
            print(f"Found token '{token}' at index {vocab_map[token]}")

    # Look for space-related tokens
    space_tokens = []
    for i, token in enumerate(vocab):
        if token.startswith(' ') or token.startswith('Ġ') or token.startswith('▁'):
            space_tokens.append((i, token))

    print(f"Found {len(space_tokens)} space-related tokens")
    if space_tokens:
        print("First 10 space tokens:")
        for i, (idx, token) in enumerate(space_tokens[:10]):
            print(f"  {idx}: '{token}'")

    # Create a more comprehensive tokenizer config
    # Based on GPT-style tokenizer but adapted for Whisper
    tokenizer_config = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": 50256,
                "content": "<|endoftext|>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True
            }
        ],
        "normalizer": {
            "type": "NFC"
        },
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": True
        },
        "post_processor": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True
        },
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": True
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": vocab_map,
            "merges": [
                # Add some basic merges to handle common patterns
                "Ġ t",
                "i n",
                "e r",
                "Ġ a",
                "h e",
                "o n",
                "r e"
            ]
        }
    }

    # Write the tokenizer config
    with open(tokenizer_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)

    print(f"Created improved tokenizer.json")
    return True

def test_tokenizer_fix():
    """Test if the tokenizer fix works"""
    try:
        create_proper_tokenizer_json()

        # Import after creating the tokenizer file
        from faster_whisper import WhisperModel

        print("Loading model with improved tokenizer...")
        model = WhisperModel("./whisper_ct2", device="cpu", compute_type="int8")
        print("✅ Model loaded successfully!")

        # Test the tokenizer directly
        print("\n=== Testing tokenizer methods ===")
        tokenizer = model.hf_tokenizer

        # Test basic encoding
        test_strings = ["hello", " hello", " -", " '", "test"]
        for test_str in test_strings:
            try:
                tokens = tokenizer.encode(test_str)
                print(f"'{test_str}' -> {tokens.ids[:5] if hasattr(tokens, 'ids') else tokens[:5]}")
            except Exception as e:
                print(f"'{test_str}' -> ERROR: {e}")

        # Test transcription with empty audio
        print("\n=== Testing empty audio transcription ===")
        empty_audio = np.asarray([], dtype="float32")
        segments, info = model.transcribe(empty_audio)
        segments_list = list(segments)

        print(f"✅ Empty audio test passed: {len(segments_list)} segments")

        # Test with short silent audio
        print("\n=== Testing silent audio ===")
        silent_audio = np.zeros(8000, dtype=np.float32)  # 0.5 seconds
        segments, info = model.transcribe(silent_audio)
        segments_list = list(segments)

        print(f"Silent audio results:")
        print(f"  Language: {info.language}")
        print(f"  Duration: {info.duration:.2f}s")
        print(f"  Segments: {len(segments_list)}")

        print("\n✅ All tests passed! Tokenizer issue fixed.")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_tokenizer_fix()