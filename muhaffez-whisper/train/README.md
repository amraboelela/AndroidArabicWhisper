# Training Scripts

This directory contains training scripts for the encoder-decoder model.

## Full Segments Training

Train on complete segments (full audio → full text), example:

```bash
python3 train_full.py Quran-A 002-04
```

## Curriculum Training

Train incrementally on segment chunks with 1.3-second chunks, one word per chunk, example:

```bash
python3 train_curriculum.py Quran-A 002-04
```

This will train on each segment (002-04-01.wav, 002-04-02.wav, etc.) progressively:
- Stage 1: First 1.3s → 1 word (per segment)
- Stage 2: First 2.6s → 2 words (per segment)
- Stage 3: First 3.9s → 3 words (per segment)
- ... until full segment audio → full segment transcription

## Run Both Training Suites

The master training script runs both curriculum and full segments training:

```bash
# Usage: ./train.sh [dataset_name] [surah_or_part]
```

**Examples:**

```bash
# Train all available datasets
./train.sh

# Train on entire dataset (all surah parts)
./train.sh Quran-A

# Train on Al-Fatiha (001)
./train.sh Quran-A 001

# Train on all parts of surah 002 (Al-Baqara)
./train.sh Quran-A 002

# Train on specific surah part only
./train.sh Quran-A 002-04
```

The script automatically detects what you want:
- **No parameters**: Trains on all datasets in ../datasets/
- **Dataset only**: Trains on all text files in the dataset
- **3-digit number (e.g., 002)**: Trains on all parts of that surah
- **Specific part (e.g., 002-04)**: Trains only on that part

### Logging

Log files are created per dataset and surah with automatic day rotation:
- **Format**: `log_train_{dataset}_{surah}.txt`
- **Examples**:
  - `log_train_Quran-A_001.txt` - Training for surah 001
  - `log_train_Quran-A_002.txt` - Training for surah 002
- **Day Rotation**: Backups saved as `.1` (Monday) through `.7` (Sunday)
  - Previous logs moved to `log_train_Quran-A_002.txt.{day}` before creating new log
  - Provides 7-day rolling history per surah
- **Content**: All output from both curriculum and full training for all parts of that surah

## Files

- **train_full.py** - Full segments training script (trains on full segments)
- **train_curriculum.py** - Curriculum learning script (trains on segment chunks progressively, each chunk is 1.3 seconds)
- **train.sh** - Runs both training suites in sequence

## Configuration

### Common Settings
- **Max Epochs**: 100
- **Learning Rate**: 1e-5
- **Learning Rate Scheduler**: 0.5x decay per epoch
- **Early Stopping**: min_delta=1e-3, patience=3
- **Normalization**: Per-segment mel normalization

### train_full.py Settings
- **Training Mode**: Full segments (complete segment audio → complete segment text)

### train_curriculum.py Settings
- **Training Mode**: Segment chunks (progressive chunking)
- **Chunk Duration**: 1.3 seconds per word (fixed, estimated once during initial setup)
- **Words Per Chunk**: 1 word

## How Curriculum Training Works

The curriculum training script:
1. Uses a fixed chunk duration of 1.3 seconds per word (estimated once during initial setup)
2. For each audio segment, calculates how many 1.3s chunks fit in the audio duration
3. Trains progressively on increasing chunks: 1 chunk (1.3s) → 1 word, 2 chunks (2.6s) → 2 words, etc.
4. Stops when reaching either the audio length limit or the number of words in the transcription
5. Saves the model after training all segments

This approach helps the model learn progressively, starting with simpler tasks (predicting 1 word from 1.3s) and gradually increasing difficulty based on the actual audio length of each segment.

### Example: Training on 002-04

When you run `python3 train_curriculum.py Quran-A 002-04`, it will:
- Load all segments from surah part 002-04 (e.g., 002-04-01.wav, 002-04-02.wav, ...)
- For each segment:
  - Calculate audio duration (e.g., 5.2 seconds)
  - Determine how many 1.3s chunks fit (e.g., 5.2 / 1.3 = 4 chunks)
  - Check word count in transcription (e.g., 6 words)
  - Train on min(4 chunks, 6 words) = 4 progressive steps:
    - 1 chunk (1.3s) → 1 word
    - 2 chunks (2.6s) → 2 words
    - 3 chunks (3.9s) → 3 words
    - 4 chunks (5.2s) → 4 words

Each training step runs for up to 100 epochs with early stopping (min_delta=1e-3, patience=3, min_epochs=5).
