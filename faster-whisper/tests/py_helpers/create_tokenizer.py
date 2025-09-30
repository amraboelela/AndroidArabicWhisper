#!/usr/bin/env python3
"""
Python helper script to create tokenizer.json from vocabulary.json
Created by Amr Aboelela
"""

import os
import json

def create_tokenizer_json(model_path):
    """Create tokenizer.json from vocabulary.json"""
    vocab_path = os.path.join(model_path, "vocabulary.json")
    tokenizer_path = os.path.join(model_path, "tokenizer.json")

    if os.path.exists(tokenizer_path):
        print("tokenizer.json already exists")
        return True

    if not os.path.exists(vocab_path):
        print("vocabulary.json not found")
        return False

    try:
        # Load vocabulary
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)

        # Create a minimal tokenizer config
        tokenizer_config = {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": [],
            "normalizer": None,
            "pre_tokenizer": None,
            "post_processor": None,
            "decoder": None,
            "model": {
                "type": "BPE",
                "dropout": None,
                "unk_token": None,
                "continuing_subword_prefix": None,
                "end_of_word_suffix": None,
                "fuse_unk": False,
                "vocab": {token: i for i, token in enumerate(vocab)},
                "merges": []
            }
        }

        with open(tokenizer_path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)

        print(f"Created basic tokenizer.json with {len(vocab)} tokens")
        return True

    except Exception as e:
        print(f"Error creating tokenizer.json: {e}")
        return False

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python3 create_tokenizer.py <model_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    success = create_tokenizer_json(model_path)
    sys.exit(0 if success else 1)