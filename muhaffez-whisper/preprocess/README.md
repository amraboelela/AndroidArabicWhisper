# Audio Preprocessing Pipeline

This directory contains the complete audio preprocessing pipeline for preparing Quranic audio data for training.

## Overview

The preprocessing pipeline converts raw Quranic audio recordings into training-ready data through five stages:

1. **Segmentation** - Split audio based on silence detection
2. **Transcription** - Generate text using Whisper
3. **Normalization** - Normalize Arabic text (remove diacritics)
4. **Mic Quality Conversion** - Downsample to 8kHz mobile mic quality
5. **Mel Feature Generation** - Extract Whisper-accurate mel spectrograms

## Pipeline Script

### `preprocess.sh` - Main Pipeline Orchestrator

Runs all five preprocessing stages in sequence.

**Usage:**
```bash
./preprocess.sh <dataset_name> <segment_name>
```

**Examples:**
```bash
./preprocess.sh Quran-A 002-04  # Process Al-Baqara part 4
./preprocess.sh Quran-A 001     # Process Al-Fatiha
```

**Output:**
- Raw audio segments (16kHz): `datasets/{dataset}/audio/raw/{surah}/{segment}-*.wav`
- Mic audio segments (8kHz): `datasets/{dataset}/audio/mic/{surah}/{segment}-*.wav`
- Transcriptions: `datasets/{dataset}/text/{segment}.txt`
- Mel features: `datasets/{dataset}/mels/{surah}/{segment}-*.pt`

## Individual Scripts

### 1. `segment_audio.py` - Audio Segmentation

Splits audio files based on silence detection using energy threshold.

**Parameters:**
- `threshold_db=-30`: Energy threshold in dB
- `min_silence_frames=11`: Minimum silence frames to split

**Usage:**
```bash
python3 segment_audio.py <dataset_name> <segment_name>
```

**How it works:**
- Detects silent regions where energy drops below threshold
- Splits audio at silence boundaries
- Saves 16kHz raw segments to `audio/raw/`

### 2. `transcribe_segments.py` - Transcription

Transcribes audio segments using OpenAI Whisper base model.

**Usage:**
```bash
python3 transcribe_segments.py <dataset_name> <segment_name>
```

**Features:**
- Uses `openai/whisper-base` model
- Forced Arabic language transcription
- Processes all segments in a part
- Saves one transcription per line

### 3. `normalize_text.py` - Text Normalization

Normalizes Arabic transcriptions by removing diacritics and extra whitespace.

**Usage:**
```bash
python3 normalize_text.py <dataset_name> <segment_name>
```

**Normalization:**
- Removes diacritics (َ ً ُ ِ ّ ْ ٌ ٍ)
- Removes extra whitespace
- Preserves word order and spacing

### 4. `convert_to_mic_quality.py` - Mobile Mic Quality

Converts 16kHz raw audio to 8kHz mobile microphone quality for realistic training conditions.

**Usage:**
```bash
python3 convert_to_mic_quality.py <dataset_name> [segment_name]
```

**Features:**
- Downsamples from 16kHz to 8kHz
- Converts stereo to mono
- Processes specific part or all parts (fallback)
- Skips already converted files

**Why 8kHz?**
Mobile microphones typically record at lower quality. Training on 8kHz audio makes the model robust to real-world mobile recording conditions.

### 5. `generate_mels.py` - Mel Feature Extraction

Generates Whisper-accurate mel spectrogram features from mic quality audio.

**Usage:**
```bash
python3 generate_mels.py <dataset_name> [segment_name]
```

**Features:**
- 100% Whisper-accurate mel extraction
- Uses Whisper's exact mel filterbank (mel_80.npz)
- Uses Whisper's STFT settings (n_fft=400, hop=160)
- 40 mel filterbanks (reduced from 80 for lighter model)
- Precomputed features saved as PyTorch tensors (.pt files)
- Processes specific part or all parts (fallback)

**Why precompute?**
Mel extraction is computationally expensive. Precomputing features:
- Speeds up training (no runtime feature extraction)
- Ensures consistency across training runs
- Saves disk space vs raw audio duplication

## Data Flow

```
Input: Long audio recording (e.g., 002-04_full.wav)
  ↓
[1] Segment Audio → audio/raw/002/002-04/002-04-{01..N}.wav (16kHz)
  ↓
[2] Transcribe → text/002-04.txt (one line per segment)
  ↓
[3] Normalize → text/002-04.txt (normalized, overwrites)
  ↓
[4] Convert to Mic → audio/mic/002/002-04/002-04-{01..N}.wav (8kHz)
  ↓
[5] Generate Mels → mels/002/002-04/002-04-{01..N}.pt (40 mels)
  ↓
Output: Training-ready data (8kHz audio + mels + normalized text)
```

## Directory Structure

```
datasets/
└── Quran-A/
    ├── audio/
    │   ├── raw/           # 16kHz segmented audio (from step 1)
    │   │   └── 002/
    │   │       └── 002-04/
    │   │           ├── 002-04-01.wav
    │   │           ├── 002-04-02.wav
    │   │           └── ...
    │   └── mic/           # 8kHz mobile mic quality (from step 4)
    │       └── 002/
    │           └── 002-04/
    │               ├── 002-04-01.wav
    │               └── ...
    ├── mels/              # Precomputed mel features (from step 5)
    │   └── 002/
    │       └── 002-04/
    │           ├── 002-04-01.pt
    │           └── ...
    └── text/              # Normalized transcriptions (from step 3)
        └── 002-04.txt
```

## Technical Details

### Mel Spectrogram Features (40 mels)
- **Sample Rate**: 16kHz (resampled from 8kHz input)
- **STFT**: n_fft=400, hop_length=160 (100 fps)
- **Mel Filterbank**: Whisper's mel_80.npz (using 40 mels)
- **Normalization**: Per-segment normalization (mean=0, std=1)
- **Format**: PyTorch tensor (time, 40)

### Audio Quality
- **Raw**: 16kHz, mono, 16-bit PCM
- **Mic**: 8kHz, mono, 16-bit PCM (mobile mic simulation)

### Text Format
- **Encoding**: UTF-8
- **One line per segment** (matches audio segment order)
- **Normalized**: No diacritics, single spaces

## Best Practices

1. **Always run the full pipeline** using `preprocess.sh` to ensure consistency
2. **Check transcription quality** after step 2 before continuing
3. **Verify segment count** matches between audio files and text lines
4. **Use fallback mode** to batch-process entire datasets:
   ```bash
   cd preprocessing
   python3 convert_to_mic_quality.py Quran-A    # Convert all parts
   python3 generate_mels.py Quran-A              # Generate all mels
   ```

## Troubleshooting

**Problem: Transcription mismatches**
- Check that Whisper model downloaded correctly
- Verify audio quality (16kHz, clear speech)
- Manually review and correct text file

**Problem: Too many/few segments**
- Adjust `threshold_db` in `segment_audio.py`
- Lower threshold = more segments (more sensitive)
- Higher threshold = fewer segments (less sensitive)

**Problem: Mel features missing**
- Ensure mic audio exists first (step 4 before step 5)
- Check that `openai-whisper` package installed
- Verify mel_filters.npz exists in whisper package

**Problem: Out of disk space**
- Mel features are compact (~10% of audio size)
- Raw audio can be archived after mic conversion
- Keep mic audio and mels for training
