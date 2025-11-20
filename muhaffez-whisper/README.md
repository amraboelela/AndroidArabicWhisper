# Muhaffez Whisper - Arabic Quranic Audio Transcription

This directory contains an encoder-decoder transformer model for Arabic Quranic audio transcription, optimized for Quranic recitation.

## Project Overview

### Architecture
- **Type**: Encoder-Decoder Transformer
- **Encoder**: Processes mel spectrogram audio features (80 mel bins)
- **Decoder**: Generates Arabic text transcriptions word by word
- **Vocabulary**: Arabic words extracted from Quranic text
- **Training**: Curriculum learning with 1.3-second chunks

### Model Components
- **Audio Encoder**: Transformer encoder processing mel spectrogram features
- **Text Decoder**: Transformer decoder generating Arabic word sequences
- **Vocabulary**: Word-level tokenization for Quranic Arabic
- **Feature Extraction**: Whisper-compatible mel spectrograms (100 fps)

## Directory Structure

```
muhaffez-whisper/
├── datasets/           # Training datasets
│   ├── Quran-A/       # Dataset with audio and text
│   │   ├── audio/     # Segmented audio files (.wav)
│   │   └── text/      # Transcription files (.txt)
│   ├── quran-simple-min.txt
│   └── quran-simple-norm.txt
├── models/            # Trained models and vocabulary
│   ├── muhaffez_whisper.pt
│   └── vocabulary.json
├── train/             # Training scripts
│   ├── train_full.py
│   ├── train_curriculum.py
│   └── train.sh
├── test/              # Testing scripts
│   ├── test_full.py
│   ├── test_curriculum.py
│   └── test.sh
├── preprocessing/     # Audio preprocessing pipeline
│   ├── segment_audio.sh
│   ├── segment_audio.py
│   ├── normalize_text.py
│   └── transcribe_segments.py
└── tools/    # Model implementations
    └── encoder_decoder_transformer.py
```

## Quick Start

### 1. Preprocess Audio
Prepare audio data by segmenting long recordings into smaller chunks:

```bash
cd preprocessing
./segment_audio.sh Quran-A 002-04
```

### 2. Train Model
Train using both curriculum learning and full segments:

```bash
cd train
./train.sh Quran-A 002-04
```

### 3. Test Model
Evaluate the trained model:

```bash
cd test
./test.sh Quran-A 002-04
```

## Training Approaches

### Full Segments Training
Trains on complete audio segments without chunking.

```bash
python3 train/train_full.py Quran-A 002-04
```

### Curriculum Learning
Trains progressively on increasing chunk sizes (1.3s per word):

```bash
python3 train/train_curriculum.py Quran-A 002-04
```

For each segment:
1. Calculate how many 1.3s chunks fit in the audio
2. Train on 1 chunk → 1 word, 2 chunks → 2 words, etc.
3. Stop at min(audio_chunks, word_count)

## Configuration

### Training Settings
- **Max Epochs**: 100
- **Learning Rate**: 1e-5
- **Learning Rate Scheduler**: 0.5x decay per epoch
- **Early Stopping**: min_delta=1e-3, patience=3
- **Normalization**: Per-segment mel normalization
- **Chunk Duration**: 1.3 seconds per word (curriculum)

## Model Files

### Trained Models
- `models/muhaffez_whisper.pt` - Main trained model
- `models/vocabulary.json` - Arabic word vocabulary

### Training Artifacts
- `models/muhaffez_whisper_backup_*.pt` - Backup models
- `checkpoint_*.pt` - Training checkpoints

## Usage Examples

### Training Entire Dataset
```bash
./train.sh Quran-A
```

### Training Specific Surah
```bash
./train.sh Quran-A 001  # Al-Fatiha
./train.sh Quran-A 002  # All parts of Al-Baqara
```

### Training Specific Part
```bash
./train.sh Quran-A 002-04  # Al-Baqara part 4 only
```

### Testing
Same usage patterns as training:
```bash
./test.sh Quran-A          # Test all
./test.sh Quran-A 002      # Test surah 002
./test.sh Quran-A 002-04   # Test part 002-04
```

## Model Architecture Details

### Encoder
- Processes mel spectrogram features (80 mel bins)
- 4 transformer encoder layers
- 4 attention heads
- d_model=128, d_ff=512

### Decoder
- Generates word-level Arabic transcription
- 4 transformer decoder layers
- 4 attention heads
- d_model=128, d_ff=512

### Feature Extraction
- Whisper-compatible mel spectrograms
- 16kHz sample rate
- 80 mel bins
- 100 fps (frames per second)
- Global Whisper normalization

## Data Format

### Audio Files
- Format: WAV, 16kHz, mono
- Naming: `{surah_part}-{segment}.wav` (e.g., `002-04-01.wav`)
- Location: `datasets/{dataset}/audio/{surah}/`

### Text Files
- Format: UTF-8 text, one transcription per line
- Naming: `{surah_part}.txt` (e.g., `002-04.txt`)
- Location: `datasets/{dataset}/text/`
- Content: Normalized Arabic text without diacritics

## References

- OpenAI Whisper: [whisper](https://github.com/openai/whisper)
- Tarteel AI: Arabic Quran speech recognition models
- Transformer architecture: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## License

Created by Amr Aboelela for Arabic Quranic audio transcription research.
