# AndroidArabicWhisper

This project integrates the Arabic Quran Whisper model into an Android app.

The model generation is handled by [faster-whisper](faster-whisper/README.md), which provides instructions to:

* Convert the Hugging Face model (`tarteel-ai/whisper-base-ar-quran`) to CTranslate2 format
* Run Python transcription demos
* Generate the `whisper_ct2/` model folder for integration

---

## 📦 Integration

1. Follow the instructions in [faster-whisper README](../faster-whisper/README.md) to generate `whisper_ct2/`.
2. Copy `whisper_ct2/` into your Android project:

```

app/src/main/assets/whisper_ct2/

```

3. Load the model in your Android code (e.g., using `WhisperHelper`) and run transcription as needed.

## ✅ Notes

* The Android app uses the pre-converted model for offline transcription.
* You do **not** need to convert the model on Android; do it once in faster-whisper.
