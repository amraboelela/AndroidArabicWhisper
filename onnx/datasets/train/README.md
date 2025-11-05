# Training Scripts

This directory contains training scripts for the encoder-decoder model.

## Quick Start

### Curriculum Training (Recommended)

Train incrementally with 1.3-second chunks, one word per chunk:

```bash
python3 train_curriculum.py 002-04
```

This will train:
- Stage 1: First 1.3s → 1 word
- Stage 2: First 2.6s → 2 words
- Stage 3: First 3.9s → 3 words
- ... until full audio → full transcription

### Single Segment Training

Train on a single segment without curriculum stages:

```bash
python3 train_full.py 002-04
```

## Files

- **train_curriculum.py** - Curriculum learning script (recommended)
- **train_full.py** - Single-segment training script
- **train.sh** - Runs all training suites in sequence

## Usage Examples

### Curriculum Training (Recommended)

```bash
# Train on Al-Fatiha with curriculum learning
python3 train_curriculum.py 001

# Train on Al-Baqara part 4 with curriculum learning
python3 train_curriculum.py 002-04

# With custom dataset
python3 train_curriculum.py 002-04 base
```

### Full Segment Training

```bash
# Train on a single segment (no curriculum stages)
python3 train_full.py 001
python3 train_full.py 002-04
python3 train_full.py 002-04 base
```

### Run All Training Suites

```bash
# Run complete training pipeline
./train.sh
```

## Configuration

### train_curriculum.py Settings
- **Chunk Duration**: 1.3 seconds per word (fixed)
- **Words Per Chunk**: 1 word
- **Max Epochs Per Stage**: 100
- **Learning Rate**: 1e-5 with 0.5x decay per epoch
- **Early Stopping**: Loss change < 0.001
- **Normalization**: Per-segment mel normalization

### train_full.py Settings
- **Max Epochs**: 100
- **Learning Rate**: 1e-5
- **Early Stopping**: Loss change < 0.001
- **Normalization**: Per-segment mel normalization

## How Curriculum Training Works

The curriculum training script automatically:
1. Analyzes your transcriptions to find the maximum number of words
2. Creates training stages: 1.3s → 1 word, 2.6s → 2 words, etc.
3. Trains the model incrementally from simple to complex
4. Saves checkpoints after each stage
5. Loads the best checkpoint before moving to the next stage

This approach helps the model learn progressively, starting with simpler tasks (predicting 1 word from 1.3s) and gradually increasing difficulty.

### Example: Training on 002-04

When you run `python3 train_curriculum.py 002-04`, it will:
- Analyze the data (65 segments, 556 words)
- Create 20 stages based on max words (19) + final stage (full)
- Train progressively:
  - Stage 1: 1.3s → 1 word
  - Stage 2: 2.6s → 2 words
  - ...
  - Stage 19: 24.7s → 19 words
  - Stage 20: Full audio → Full text

Each stage trains for up to 100 epochs with early stopping when the loss plateaus.
