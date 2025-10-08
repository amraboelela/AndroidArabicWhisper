#!/usr/bin/env python3

import os
import json
import numpy as np

# Set environment variables to prevent network access
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

from faster_whisper import WhisperModel

def find_whisper_ct2_path():
    """Find the whisper_ct2 directory"""
    if os.path.exists("../../app/src/main/assets/whisper_ct2/model.bin"):
        return os.path.abspath("../../app/src/main/assets/whisper_ct2")
    else:
        raise FileNotFoundError("Could not find whisper_ct2 directory")

def create_simple_tokenizer_json():
    """Create a minimal tokenizer.json file from vocabulary.json"""
    model_path = find_whisper_ct2_path()
    vocab_path = os.path.join(model_path, "vocabulary.json")
    tokenizer_path = os.path.join(model_path, "tokenizer.json")

    if os.path.exists(tokenizer_path):
        print("tokenizer.json already exists")
        return

    if not os.path.exists(vocab_path):
        print("vocabulary.json not found")
        return

    # Load vocabulary
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    # Create a minimal tokenizer config
    # This is a simplified structure - may need adjustment
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

def test_whisper_ct2_offline():
    """Test the local CTranslate2 Whisper model in offline mode"""
    try:
        # Create tokenizer.json if it doesn't exist
        create_simple_tokenizer_json()

        model_path = find_whisper_ct2_path()
        print(f"Testing CTranslate2 model at: {model_path}")

        print("\nLoading WhisperModel in offline mode...")
        model = WhisperModel(
            model_path,
            device="cpu",
            compute_type="int8"
        )

        print("Model loaded successfully!")

        # Test with real audio file
        print("\n=== Testing with data/001.wav ===")
        audio_file = "data/001.wav"

        if os.path.exists(audio_file):
            print(f"Loading audio file: {audio_file}")
            from faster_whisper.audio import decode_audio

            # Load the audio file
            audio = decode_audio(audio_file)
            print(f"Audio loaded: {len(audio)} samples ({len(audio)/16000:.2f} seconds)")

            # Log audio statistics
            print(f"Audio stats: min={audio.min():.6f}, max={audio.max():.6f}, mean={audio.mean():.6f}, std={audio.std():.6f}")
            print(f"First 20 samples: {audio[:20]}")

            # Transcribe the audio
            segments, info = model.transcribe(audio)

            print(f"Detected language: {info.language}")
            for segment in segments:
                print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

        # Test supported languages
        print(f"\n=== Model Information ===")
        print(f"Supported languages: {len(model.supported_languages)}")
        print(f"Is multilingual: {model.model.is_multilingual}")

        print("\n✅ CTranslate2 offline test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Error testing CTranslate2 model offline: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_whisper_ct2_offline()