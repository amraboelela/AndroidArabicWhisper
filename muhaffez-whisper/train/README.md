# Training Scripts

This directory contains training scripts for the encoder-decoder Whisper model.

## Training Suites

### 1. Full Training (`train_full.py`)
Trains on complete segments (full audio → full text):

```bash
# Train all parts
python3 train_full.py Quran-A all

# Train specific part
python3 train_full.py Quran-A 002-04
```

### 2. Augmented Training (`train_augmented.py`)
Trains on normal + augmented data (speed/pitch variations):

```bash
# Train all parts
python3 train_augmented.py Quran-A all

# Train specific part
python3 train_augmented.py Quran-A 002-04
```

**Augmented Data Includes:**
- Normal samples (original audio)
- Speed variations: ±10%, ±20%
- Pitch variations: ±2, ±4 semitones

### 3. Curriculum Training (`train_curriculum.py`)
Trains incrementally on segment chunks with 1.3-second chunks, one word per chunk:

```bash
# Train all parts
python3 train_curriculum.py Quran-A all

# Train specific part
python3 train_curriculum.py Quran-A 002-04
```

**Curriculum Strategy:**
- Stage 1: First 1.3s → 1 word (per segment)
- Stage 2: First 2.6s → 2 words (per segment)
- Stage 3: First 3.9s → 3 words (per segment)
- ... until full segment audio → full segment transcription

**NEW: Replay Buffer (10%)**
- Each epoch samples fresh 10% from augmented data (normal + variations)
- Prevents forgetting full-length sequences during curriculum training
- Different samples each epoch for better generalization

## Master Training Script

Run all three training suites in sequence:

```bash
# Usage: ./train.sh [dataset_name] [surah_or_part]
```

**Examples:**

```bash
# Train all available datasets
./train.sh all

# Train entire dataset (all surah parts)
./train.sh Quran-A all

# Train on Al-Fatiha (001)
./train.sh Quran-A 001

# Train on all parts of surah 002 (Al-Baqara)
./train.sh Quran-A 002

# Train on specific surah part
./train.sh Quran-A 002-04
```

**How the script works:**

- **`all` parameter** - Trains on all datasets in ../datasets/
- **Dataset + `all`** - Trains on all text files in the dataset
- **3-digit number (e.g., 002)** - Trains on all parts of that surah
- **Specific part (e.g., 002-04)** - Trains only on that part


## Logging

Log files are created per dataset with output to console and file:

- **Location**: `train/log.txt`
- **Content**: All output from full, augmented, and curriculum training
- **Format**: Timestamped entries with suite markers

## Accuracy Calculation

### Full & Augmented Training
Tests on full segments with 20% confidence threshold:
- Filters out predictions with <20% probability
- Measures word-by-word accuracy on full transcriptions
- **Frequency**: Every 10 epochs, or every epoch when accuracy > 90%

### Curriculum Training
Tests on curriculum-appropriate samples:
- Samples every 8th curriculum stage (1-word, 2-word, ..., full)
- Matches training distribution (short + long sequences)
- Uses same 20% confidence threshold
- **Frequency**: Every 10 epochs, or every epoch when accuracy > 90%

## Optimizer State Management

### Reset Optimizer

```bash
cd ../models
./reset_optimizer.sh
```

Resets optimizer states (LR, momentum) while preserving model weights:
- Sets LR back to 1e-3
- Clears momentum buffers
- Resets epoch counters and accuracy
- **Model weights preserved!**

Use this when you want to restart training with fresh optimizer state.

## Configuration

### Common Settings
- **Initial Learning Rate**: 1e-3
- **LR Decay**: 0.5x when loss increases
- **Minimum LR**: 1e-7
- **Optimizer**: AdamW (weight_decay=0.01)
- **Loss**: CrossEntropyLoss (label_smoothing=0.1)
- **Gradient Clipping**: 1.0

### Full Training
- Trains on full segments
- Early stop: Accuracy > 99% or LR ≤ 1e-7

### Augmented Training
- Normal segments + 8 augmentation variations
- Early stop: Accuracy > 99% or LR ≤ 1e-7

### Curriculum Training
- **Chunk Duration**: 1.3 seconds per word
- **Words Per Chunk**: 1 word
- **Replay Buffer**: 10% from augmented data (resampled each epoch)
- Early stop: Accuracy > 99% or LR ≤ 1e-7

## How Curriculum Training Works

1. **Collect Curriculum Stages**: For each segment, create progressive stages (1 word, 2 words, ..., full)
2. **Load Augmented Data Pool**: All normal + augmented samples for replay buffer
3. **Each Epoch**:
   - Sample fresh 10% replay buffer from augmented pool
   - Combine curriculum stages + replay samples
   - Shuffle and train on combined data
4. **Accuracy**: Test only on curriculum stages (not replay buffer)

### Example: 002-04 Segment

Audio duration: 5.2 seconds, Transcription: 6 words

Curriculum stages:
- Stage 1: 1.3s → 1 word
- Stage 2: 2.6s → 2 words
- Stage 3: 3.9s → 3 words
- Stage 4: 5.2s → 4 words

(Stops at 4 chunks because audio is only 5.2s)

## Files

### Main Scripts
- **train.sh** - Master script running all three suites
- **train_full.py** - Full segments training
- **train_augmented.py** - Augmented data training
- **train_curriculum.py** - Curriculum learning with replay buffer

### Model Scripts (`../models/`)

- **encoder_decoder_transformer.py** - Model architecture definition (encoder-decoder transformer)
- **init_model.py** - Initialize new model with random weights
- **inspect_muhaffez_whisper.py** - Inspect model architecture and parameters
- **export_muhaffez_to_onnx.py** - Export model to ONNX format for deployment
- **reset_optimizer.sh** - Reset optimizer states (LR, momentum, epoch) while preserving model weights

### Common Module (`common/`)
- **data_utils.py** - Mel features loading, tokenization
- **metrics.py** - Accuracy calculation (comprehensive, curriculum)
- **replay_buffer.py** - Replay buffer collection
- **data_collection.py** - Augmented data collection
- **training_loop.py** - Training epoch, LR updates
- **optimizer_state.py** - Checkpoint save/load

## Model Checkpoint Structure

```python
{
    'model_state_dict': {...},  # Model weights
    'full': {
        'epoch': int,
        'optimizer_state_dict': {...},
        'loss': float,
        'lr': float,
        'accuracy': float
    },
    'augmented': { ... },
    'curriculum': { ... }
}
```

Each training type maintains independent optimizer state, allowing you to train with different methods without interference.
