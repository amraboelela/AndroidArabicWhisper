# Audio Segmentation Methods

This directory contains different approaches for segmenting Quran recitation audio.

## Scripts

### 1. `segment_audio_001.py` - Energy-Based Segmentation
Uses simple energy threshold detection to find silent regions.

**Parameters:**
- `threshold_db=-30`: Energy threshold in dB (default -20, changed to -30 for Android)
- `min_silence_frames=1`: Minimum number of 50ms frames for silence (default 2, changed to 1 for Android)

**Usage:**
```bash
python3 segment_audio_001.py <audio_file> [output_dir]
```

**Results for 001.wav (Al-Fatiha):**
- With default params (-20dB, 2 frames): 21 segments
- With Android params (-30dB, 1 frame): 8 segments ✅ RECOMMENDED

### 2. `segment_audio_vad.py` - VAD-Based Segmentation
Uses Silero VAD (Voice Activity Detection) model for intelligent speech detection.

**Optimized Parameters for Quran Recitation:**
- `threshold=0.05`: Very low threshold for soft speech (default 0.5 too high)
- `min_speech_duration_ms=50`: Allow very short segments
- `max_speech_duration_s=inf`: No maximum limit
- `min_silence_duration_ms=500`: Minimum 500ms silence to split
- `speech_pad_ms=200`: More padding to prevent chopping endings
- **Energy-based fallback**: Checks remaining audio after last VAD segment

**Usage:**
```bash
python3 segment_audio_vad.py <audio_file> [output_dir]
```

**Results for 001.wav (with optimized params):**
- 3 segments
- ✅ Successfully captures entire audio including final "آمين" (ends at 43.13s)
- ✅ WORKING - with proper threshold tuning

**Key Fix:** The default Silero VAD threshold (0.5) is too high for soft Quran recitation. Lowering to 0.05 makes it sensitive enough to detect quiet endings.

### 3. `compare_segmentation.py` - Comparison Tool
Compares energy-based and VAD-based segmentation side-by-side.

**Usage:**
```bash
python3 compare_segmentation.py <audio_file>
```

## Analysis: VAD Threshold Tuning for Soft Speech

The Silero VAD model **default threshold (0.5) is too high** for soft Quran recitations. This causes it to miss quiet endings.

### Original Issue (threshold=0.5):
- VAD stopped at 31.36s
- Missing the last 11+ seconds: "وَلَا الضَّالِّينَ آمِين"
- Speech probabilities at end: < 0.03 (below 0.5 threshold)

### After Fix (threshold=0.05):
- ✅ VAD now captures full audio (0.00s - 43.13s)
- ✅ Successfully detects soft endings
- Energy levels at end: -15 to -20 dB (audible but quiet)

**Root Cause:** Soft recitation at the end has lower energy, which the VAD model interprets as low speech probability. The default threshold was rejecting these quiet but clearly audible speech segments.

**Solution:** Lower threshold + energy-based fallback ensures no speech is missed.

## Recommendations

### For Fine-Grained Segmentation (8 segments)
**Use energy-based segmentation:**
- `threshold_db=-30`
- `min_silence_frames=1`

This produces well-balanced verse-level segments.

### For Coarse-Grained Segmentation (3-4 segments)
**Use VAD-based segmentation:**
- `threshold=0.05` (critical for soft speech)
- `min_speech_duration_ms=50`
- `min_silence_duration_ms=500`
- `speech_pad_ms=200`

This produces longer segments, merging across brief pauses.

**Both methods now successfully capture all speech including soft endings!**

### Energy-Based Segmentation Details (8 segments):
1. Segment 1: 1.55s - أعوذ بالله من الشيطان الرجيم
2. Segment 2: 1.30s - (continuation)
3. Segment 3: 2.10s - بسم الله الرحمن الرحيم
4. Segment 4: 4.50s - الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ
5. Segment 5: 3.50s - الرَّحْمَٰنِ الرَّحِيمِ
6. Segment 6: 8.10s - مَالِكِ يَوْمِ الدِّينِ إِيَّاكَ نَعْبُدُ وَإِيَّاكَ نَسْتَعِينُ
7. Segment 7: 3.50s - اهْدِنَا الصِّرَاطَ الْمُسْتَقِيمَ
8. Segment 8: 16.05s - صِرَاطَ الَّذِينَ أَنْعَمْتَ عَلَيْهِمْ غَيْرِ الْمَغْضُوبِ عَلَيْهِمْ وَلَا الضَّالِّينَ آمِين

Total: 40.60s (out of 43.13s total)
