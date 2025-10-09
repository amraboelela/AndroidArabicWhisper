#!/usr/bin/env python3

import os
import json
import numpy as np
import time
from datetime import datetime

def log_with_timestamp(message):
    """Print message with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")

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
        log_with_timestamp(f"Testing CTranslate2 model at: {model_path}")

        log_with_timestamp("Loading WhisperModel in offline mode...")
        model = WhisperModel(
            model_path,
            device="cpu",
            compute_type="int8"
        )

        log_with_timestamp("Model loaded successfully!")

        # Test with real audio file
        audio_file = "../../app/src/main/assets/002-01.wav"
        log_with_timestamp(f"=== Testing with {audio_file} ===")

        if os.path.exists(audio_file):
            log_with_timestamp(f"Loading audio file: {audio_file}")
            from faster_whisper.audio import decode_audio

            # Load the audio file
            audio = decode_audio(audio_file)
            log_with_timestamp(f"Audio loaded: {len(audio)} samples ({len(audio)/16000:.2f} seconds)")

            # Log audio statistics
            print(f"Audio stats: min={audio.min():.6f}, max={audio.max():.6f}, mean={audio.mean():.6f}, std={audio.std():.6f}")
            print(f"First 20 samples: {audio[:20]}")

            # Start timing
            log_with_timestamp("Starting transcription...")
            start_time = time.time()

            # Transcribe the audio with VAD enabled for better segment boundaries
            segments, info = model.transcribe(audio, language="ar", vad_filter=True)

            #log_with_timestamp("Transcription completed, processing segments...")

            # Collect segments into a list for JSON output
            segments_list = []
            for segment in segments:
                print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
                segments_list.append({
                    "text": segment.text,
                    "start": segment.start,
                    "end": segment.end,
                    "avg_logprob": segment.avg_logprob,
                    "words": []  # Empty for now, matching C++ output
                })

            # End timing
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"\n⏱️  Transcription took: {elapsed_time:.2f} seconds")

            # Create JSON output matching C++ format
            result_json = {
                "success": True,
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration,
                "segments": segments_list
            }

            # Print JSON output
            print("\n" + "=" * 50)
            print("FINAL TRANSCRIPTION RESULT:")
            print("=" * 50)
            print(json.dumps(result_json, indent=2, ensure_ascii=False))

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
