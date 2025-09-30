#!/usr/bin/env python3
"""
Minimal WhisperModel caller - only calls the model
Created by Amr Aboelela
"""

import os
import json
import sys

# Set environment variables
os.environ['HF_HUB_OFFLINE'] = '1'

from faster_whisper import WhisperModel

def main():
    if len(sys.argv) != 3:
        print(json.dumps({"success": False, "error": "Usage: model_path audio_file"}))
        sys.exit(1)

    model_path = sys.argv[1]
    audio_file = sys.argv[2]

    try:
        model = WhisperModel(model_path, device="cpu", compute_type="int8")
        segments, info = model.transcribe(audio_file, word_timestamps=True)

        # Convert to basic structures for JSON serialization
        segments_data = []
        for segment in segments:
            seg_data = {
                "text": segment.text,
                "start": segment.start,
                "end": segment.end,
                "avg_logprob": segment.avg_logprob,
                "words": []
            }

            if segment.words:
                for word in segment.words:
                    seg_data["words"].append({
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                        "probability": word.probability
                    })

            segments_data.append(seg_data)

        result = {
            "success": True,
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
            "segments": segments_data
        }

        print(json.dumps(result, ensure_ascii=False))

    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}, ensure_ascii=False))
        sys.exit(1)

if __name__ == "__main__":
    main()