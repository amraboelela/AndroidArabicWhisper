# Test 002-04 Segments

## New Test File: test_002_04.py

This test file evaluates the encoder-decoder model specifically on **002-04 segments** (Al-Baqara Part 4) at different audio durations:

### Test Configurations:

1. **Full audio** → Full transcription (max 50 tokens)
2. **First 1 second** → First 2 words (max 10 tokens)
3. **First 3 seconds** → First 5 words (max 20 tokens)
4. **First 4 seconds** → First 8 words (max 30 tokens)

### Usage:

```bash
cd datasets/test
python3 test_002_04.py [dataset_name]
```

Default dataset: `base`

### Output Format:

The test shows detailed output for the first 5 segments, then provides summary statistics for all 71 segments:

```
[01/71] 002-04-01.wav
Expected: يود احدهم
Generated: يود احدهم
Match: ✓

...

Full audio - RESULTS
======================================================================
Accuracy: 50/71 (70.4%)
```

## Changes Made to Other Test Files:

All test files now use **2-digit padding** for both segment numbers and file suffixes:

**File Renames:**
- `test_001_1.py` → `test_001_01.py`
- `test_001_3.py` → `test_001_03.py`
- `test_002_1.py` → `test_002_01.py`
- `test_002_3.py` → `test_002_03.py`

**Segment Number Format:**
- `test_002.py`: Changed `[Segment {i}/...]` → `[{i:02d}/...]`
- `test_002_01.py`: Changed to 2-digit format
- `test_002_03.py`: Changed to 2-digit format
- `test_002_04.py`: Uses 2-digit format

### Example Output:
```
Before: [Segment 1/71]
After:  [01/71]
```

This provides better visual alignment and readability in the test output.

## File Structure:

```
datasets/test/
├── test_001.py       - Full Al-Fatiha (001) full audio
├── test_001_01.py    - First 1 second → first word
├── test_001_03.py    - First 3 seconds → first 2 words
├── test_002.py       - Full Al-Baqara (002-01, 002-02, 002-03) full audio
├── test_002_01.py    - First 1 second → first word
├── test_002_03.py    - First 3 seconds → first 2 words
└── test_002_04.py    - 002-04 only: full + 1s + 3s + 4s (NEW)
```
