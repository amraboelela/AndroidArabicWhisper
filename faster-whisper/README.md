# 📖 Faster-Whisper Arabic Quran Demo

This project demonstrates how to use [faster-whisper](https://github.com/guillaumekln/faster-whisper) with a Quran-specific Arabic Whisper model ([tarteel-ai/whisper-base-ar-quran](https://huggingface.co/tarteel-ai/whisper-base-ar-quran)) converted into [CTranslate2](https://github.com/OpenNMT/CTranslate2) format for fast and efficient inference.

## ▶️ Usage

Run the demo transcription, e.g.:

```bash
python load.py ~/develop/ai/stt/samples/002-01.wav
```

Example output:

```
Detected language: ar
[0.00s -> 3.25s] بسم الله الرحمن الرحيم
```

---

## 📂 Project Structure

```
faster-whisper/
├── load.py          # load model, generate whisper_ct2/, run transcription
└── whisper_ct2/     # converted model files (created by load.py)
```

