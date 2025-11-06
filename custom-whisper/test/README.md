# Testing Scripts

This directory contains test scripts for evaluating the encoder-decoder model on Arabic Quranic audio transcription.

## Test Suites

### Full Segments Testing

**test_full.py** - Test on full segments (complete segment audio → complete segment text)

Tests the model on complete audio segments without chunking. This evaluates how well the model handles full transcriptions.

**Usage:**
```bash
python3 test_full.py <dataset_name> <surah_part>
```

**Examples:**
```bash
# Test on Al-Fatiha (001)
python3 test_full.py Quran-A 001

# Test on Al-Baqara part 4
python3 test_full.py Quran-A 002-04
```

**Output:**
- Token-level accuracy
- Match indicators (✓/✗) for each segment
- Full transcription comparison

### Curriculum Testing

**test_curriculum.py** - Test using curriculum approach (progressive chunk sizes)

Tests the model progressively through increasing chunk sizes, starting from 1.3s → 1 word and building up to full segments. This evaluates how well the model handles different audio lengths.

**Usage:**
```bash
python3 test_curriculum.py <dataset_name> <surah_part>
```

**Examples:**
```bash
# Test on Al-Fatiha (001)
python3 test_curriculum.py Quran-A 001

# Test on Al-Baqara part 4
python3 test_curriculum.py Quran-A 002-04
```

**Output:**
- Stage-by-stage results (1.3s → 1 word, 2.6s → 2 words, etc.)
- Token accuracy for each stage
- Overall curriculum test accuracy

## Master Test Script

**test.sh** - Run both test suites

This script runs both test_full.py and test_curriculum.py for the specified dataset and surah parts.

**Usage:**
```bash
./test.sh <dataset_name>                    # Test all surahs in the dataset
./test.sh <dataset_name> <surah>            # Test all parts of specific surah (e.g., 002)
./test.sh <dataset_name> <surah_part>       # Test specific surah part (e.g., 002-04)
```

**Examples:**
```bash
# Test all surah parts in Quran-A dataset
./test.sh Quran-A

# Test all parts of surah 002 (Al-Baqara)
# This will test 002-01, 002-02, 002-03, 002-04, etc.
./test.sh Quran-A 002

# Test only surah part 002-04
./test.sh Quran-A 002-04
```

**Output:**
- Detailed logs for each test suite
- Summary with total runs, passed/failed counts
- Execution time for each suite
- Logs saved to `log_test.txt`

## Test Workflow

For a typical testing workflow:

1. **Test specific part:**
   ```bash
   ./test.sh Quran-A 002-04
   ```

2. **Test entire surah:**
   ```bash
   ./test.sh Quran-A 002
   ```

3. **Test entire dataset:**
   ```bash
   ./test.sh Quran-A
   ```

## Understanding Test Results

### Token Accuracy
Both test scripts calculate token-level accuracy by comparing expected vs generated words:
- **Expected:** Ground truth transcription
- **Generated:** Model output
- **Accuracy:** (Correct tokens / Total tokens) × 100%

### Match Indicators
- **✓** - Exact match between normalized expected and generated text
- **✗** - Mismatch detected

### Normalization
Arabic text is normalized by removing diacritics before comparison to focus on word accuracy rather than vocalization marks.

## Files Required

Before testing, ensure you have:

1. **Model file:** `../models/encoder_decoder_model.pt`
2. **Vocabulary:** `../models/vocabulary.json`
3. **Text files:** `../datasets/<dataset_name>/text/<surah_part>.txt`
4. **Audio files:** `../datasets/<dataset_name>/audio/<surah_num>/<surah_part>-*.wav`

Example structure:
```
custom-whisper/
├── models/
│   ├── encoder_decoder_model.pt
│   └── vocabulary.json
├── datasets/
│   └── Quran-A/
│       ├── text/
│       │   ├── 001.txt
│       │   ├── 002-01.txt
│       │   ├── 002-02.txt
│       │   ├── 002-03.txt
│       │   └── 002-04.txt
│       └── audio/
│           ├── 001/
│           │   ├── 001-01.wav
│           │   ├── 001-02.wav
│           │   └── ...
│           └── 002/
│               ├── 002-01-01.wav
│               ├── 002-01-02.wav
│               ├── 002-04-01.wav
│               ├── 002-04-02.wav
│               └── ...
└── test/
    ├── test_full.py
    ├── test_curriculum.py
    ├── test.sh
    └── README.md (this file)
```

## Notes

- **Segments**: Individual audio files (e.g., 002-04-01.wav, 002-04-02.wav)
- **Surah Parts**: Groups of segments (e.g., 002-04 contains multiple segments)
- **Device**: Tests automatically use GPU (Metal/CUDA) if available, otherwise CPU
- **Reproducibility**: Random seed set to 42 for consistent results
- **Curriculum Stages**: Automatically calculated based on the longest transcription in the dataset
