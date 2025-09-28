/**
 * Unit Tests for WhisperModel Utility Functions Implementation
 * Tests helper functions for feature processing, compression, and timestamps
 * Created by Amr Aboelela
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "whisper_model.h"
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <algorithm>
#include <numeric>

class WhisperModelUtilsTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Initialize test data
    sample_features = {
      {0.1f, 0.2f, 0.3f, 0.4f, 0.5f},
      {0.15f, 0.25f, 0.35f, 0.45f, 0.55f},
      {0.2f, 0.3f, 0.4f, 0.5f, 0.6f}
    };

    large_features = {
      {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f},
      {0.15f, 0.25f, 0.35f, 0.45f, 0.55f, 0.65f, 0.75f, 0.85f},
      {0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f}
    };

    sample_audio = {0.1f, -0.2f, 0.3f, -0.1f, 0.05f, -0.15f, 0.25f, -0.05f};

    // Setup speech chunks for timestamp restoration
    speech_chunks = {
      {{"start", 0.0f}, {"end", 10.0f}, {"original_start", 5.0f}, {"original_end", 15.0f}},
      {{"start", 10.0f}, {"end", 20.0f}, {"original_start", 25.0f}, {"original_end", 35.0f}}
    };

    // Setup alignment data
    alignment = {
      {"word1", 0.85f},
      {".", 0.95f},
      {"word2", 0.75f},
      {"!", 0.90f}
    };

    prepend_punctuations = {"\"", "'", "¿", "(", "[", "{", "-"};
    append_punctuations = {"\"", "'", ".", "。", ",", "，", "!", "！", "?", "？", ":", "：", ")", "]", "}", "、"};
  }

  std::vector<std::vector<float>> sample_features;
  std::vector<std::vector<float>> large_features;
  std::vector<float> sample_audio;
  std::vector<std::map<std::string, float>> speech_chunks;
  std::vector<std::pair<std::string, float>> alignment;
  std::vector<std::string> prepend_punctuations;
  std::vector<std::string> append_punctuations;
};

// Test slice_features function
TEST_F(WhisperModelUtilsTest, SliceFeaturesBasic) {
  int start = 1;
  int length = 3;

  auto sliced = slice_features(sample_features, start, length);

  EXPECT_EQ(sliced.size(), sample_features.size());
  for (size_t i = 0; i < sliced.size(); ++i) {
    EXPECT_EQ(sliced[i].size(), static_cast<size_t>(length));
    EXPECT_EQ(sliced[i][0], sample_features[i][start]);
    EXPECT_EQ(sliced[i][2], sample_features[i][start + 2]);
  }
}

TEST_F(WhisperModelUtilsTest, SliceFeaturesStartOutOfBounds) {
  int start = 10; // Beyond feature size
  int length = 3;

  auto sliced = slice_features(sample_features, start, length);

  EXPECT_EQ(sliced.size(), sample_features.size());
  for (const auto& row : sliced) {
    EXPECT_TRUE(row.empty());
  }
}

TEST_F(WhisperModelUtilsTest, SliceFeaturesEmptyInput) {
  std::vector<std::vector<float>> empty_features;
  int start = 0;
  int length = 3;

  auto sliced = slice_features(empty_features, start, length);

  EXPECT_TRUE(sliced.empty());
}

TEST_F(WhisperModelUtilsTest, SliceFeaturesLengthExceedsSize) {
  int start = 2;
  int length = 10; // Longer than available

  auto sliced = slice_features(sample_features, start, length);

  EXPECT_EQ(sliced.size(), sample_features.size());
  for (size_t i = 0; i < sliced.size(); ++i) {
    EXPECT_EQ(sliced[i].size(), sample_features[i].size() - start);
  }
}

// Test pad_or_trim function
TEST_F(WhisperModelUtilsTest, PadOrTrimPadding) {
  // Features smaller than target length (3000)
  auto padded = pad_or_trim(sample_features);

  EXPECT_EQ(padded.size(), sample_features.size());
  for (size_t i = 0; i < padded.size(); ++i) {
    EXPECT_EQ(padded[i].size(), 3000); // TARGET_LENGTH

    // Original values should be preserved
    for (size_t j = 0; j < sample_features[i].size(); ++j) {
      EXPECT_EQ(padded[i][j], sample_features[i][j]);
    }

    // Padded values should be zero
    for (size_t j = sample_features[i].size(); j < padded[i].size(); ++j) {
      EXPECT_EQ(padded[i][j], 0.0f);
    }
  }
}

TEST_F(WhisperModelUtilsTest, PadOrTrimTrimming) {
  // Create features larger than target length
  std::vector<std::vector<float>> large_features_for_trim(3);
  for (auto& row : large_features_for_trim) {
    row.resize(4000, 0.5f); // Larger than 3000
  }

  auto trimmed = pad_or_trim(large_features_for_trim);

  EXPECT_EQ(trimmed.size(), large_features_for_trim.size());
  for (const auto& row : trimmed) {
    EXPECT_EQ(row.size(), 3000); // TARGET_LENGTH
  }
}

TEST_F(WhisperModelUtilsTest, PadOrTrimEmptyInput) {
  std::vector<std::vector<float>> empty_features;
  auto result = pad_or_trim(empty_features);

  EXPECT_TRUE(result.empty());
}

TEST_F(WhisperModelUtilsTest, PadOrTrimExactSize) {
  // Create features exactly at target length
  std::vector<std::vector<float>> exact_features(3);
  for (auto& row : exact_features) {
    row.resize(3000, 0.25f);
  }

  auto result = pad_or_trim(exact_features);

  EXPECT_EQ(result.size(), exact_features.size());
  for (size_t i = 0; i < result.size(); ++i) {
    EXPECT_EQ(result[i].size(), 3000);
    EXPECT_EQ(result[i], exact_features[i]);
  }
}

// Test get_ctranslate2_storage function
TEST_F(WhisperModelUtilsTest, GetCTranslate2Storage) {
  auto storage = get_ctranslate2_storage(sample_features);

  // Verify storage dimensions
  auto shape = storage.shape();
  EXPECT_EQ(shape.size(), 2);
  EXPECT_EQ(shape[0], static_cast<long>(sample_features.size()));
  EXPECT_EQ(shape[1], static_cast<long>(sample_features[0].size()));

  // Verify data type
  EXPECT_EQ(storage.dtype(), ctranslate2::DataType::FLOAT);
}

TEST_F(WhisperModelUtilsTest, GetCTranslate2StorageDataIntegrity) {
  // Create simple test data
  std::vector<std::vector<float>> test_features = {
    {1.0f, 2.0f, 3.0f},
    {4.0f, 5.0f, 6.0f}
  };

  auto storage = get_ctranslate2_storage(test_features);

  // Verify shape
  auto shape = storage.shape();
  EXPECT_EQ(shape[0], 2); // 2 rows
  EXPECT_EQ(shape[1], 3); // 3 columns

  // Verify flattened data ordering (row-major)
  auto data = storage.data<float>();
  EXPECT_EQ(data[0], 1.0f); // [0][0]
  EXPECT_EQ(data[1], 2.0f); // [0][1]
  EXPECT_EQ(data[2], 3.0f); // [0][2]
  EXPECT_EQ(data[3], 4.0f); // [1][0]
  EXPECT_EQ(data[4], 5.0f); // [1][1]
  EXPECT_EQ(data[5], 6.0f); // [1][2]
}

// Test get_compression_ratio function
TEST_F(WhisperModelUtilsTest, GetCompressionRatioNormalText) {
  std::string text = "This is a test string with some repetitive content. This is a test string.";
  float ratio = get_compression_ratio(text);

  EXPECT_GT(ratio, 1.0f); // Should be compressible
  EXPECT_LT(ratio, 10.0f); // Should be reasonable
}

TEST_F(WhisperModelUtilsTest, GetCompressionRatioEmptyText) {
  std::string empty_text = "";
  float ratio = get_compression_ratio(empty_text);

  EXPECT_EQ(ratio, 1.0f); // Should return 1.0 for empty text
}

TEST_F(WhisperModelUtilsTest, GetCompressionRatioHighlyCompressible) {
  std::string repetitive_text = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
  float ratio = get_compression_ratio(repetitive_text);

  EXPECT_GT(ratio, 5.0f); // Should be highly compressible
}

TEST_F(WhisperModelUtilsTest, GetCompressionRatioRandomText) {
  std::string random_text = "xqp2w9ebrjkas8df7gh3klm5n6vct4yui1oz";
  float ratio = get_compression_ratio(random_text);

  EXPECT_GE(ratio, 1.0f); // Should be at least 1.0
  EXPECT_LT(ratio, 3.0f); // Random text shouldn't compress much
}

// Test merge_punctuations function
TEST_F(WhisperModelUtilsTest, MergePunctuationsPrepend) {
  std::vector<std::pair<std::string, float>> test_alignment = {
    {"\"", 0.9f},
    {"Hello", 0.85f},
    {"world", 0.8f}
  };

  merge_punctuations(test_alignment, prepend_punctuations, append_punctuations);

  EXPECT_EQ(test_alignment.size(), 2); // Should merge quote with "Hello"
  EXPECT_EQ(test_alignment[0].first, "\"Hello");
  EXPECT_EQ(test_alignment[1].first, "world");
}

TEST_F(WhisperModelUtilsTest, MergePunctuationsAppend) {
  std::vector<std::pair<std::string, float>> test_alignment = {
    {"Hello", 0.85f},
    {".", 0.9f},
    {"world", 0.8f}
  };

  merge_punctuations(test_alignment, prepend_punctuations, append_punctuations);

  EXPECT_EQ(test_alignment.size(), 2); // Should merge period with "Hello"
  EXPECT_EQ(test_alignment[0].first, "Hello.");
  EXPECT_EQ(test_alignment[1].first, "world");
}

TEST_F(WhisperModelUtilsTest, MergePunctuationsEmpty) {
  std::vector<std::pair<std::string, float>> empty_alignment;

  EXPECT_NO_THROW({
    merge_punctuations(empty_alignment, prepend_punctuations, append_punctuations);
  });

  EXPECT_TRUE(empty_alignment.empty());
}

TEST_F(WhisperModelUtilsTest, MergePunctuationsNoPunctuation) {
  std::vector<std::pair<std::string, float>> test_alignment = {
    {"Hello", 0.85f},
    {"world", 0.8f},
    {"test", 0.75f}
  };

  auto original_size = test_alignment.size();
  merge_punctuations(test_alignment, prepend_punctuations, append_punctuations);

  EXPECT_EQ(test_alignment.size(), original_size); // No change expected
  EXPECT_EQ(test_alignment[0].first, "Hello");
  EXPECT_EQ(test_alignment[1].first, "world");
  EXPECT_EQ(test_alignment[2].first, "test");
}

// Test restore_speech_timestamps function
TEST_F(WhisperModelUtilsTest, RestoreSpeechTimestamps) {
  std::vector<Segment> segments;

  // Create test segment
  Segment seg;
  seg.start = 2.0f;
  seg.end = 8.0f;
  seg.text = "Test segment";

  // Create words with timestamps relative to speech chunks
  Word word1;
  word1.start = 3.0f;
  word1.end = 5.0f;
  word1.word = "Test";

  Word word2;
  word2.start = 5.0f;
  word2.end = 7.0f;
  word2.word = "segment";

  seg.words = {word1, word2};
  segments.push_back(seg);

  int sampling_rate = 16000;
  auto restored = restore_speech_timestamps(segments, speech_chunks, sampling_rate);

  EXPECT_EQ(restored.size(), segments.size());
  EXPECT_TRUE(restored[0].words.has_value());

  // Timestamps should be restored to original audio timeline
  auto restored_words = restored[0].words.value();
  EXPECT_EQ(restored_words.size(), 2);

  // Check that timestamps were modified (restored)
  EXPECT_NE(restored_words[0].start, word1.start);
  EXPECT_NE(restored_words[0].end, word1.end);
}

TEST_F(WhisperModelUtilsTest, RestoreSpeechTimestampsEmptyChunks) {
  std::vector<Segment> segments;
  Segment seg;
  seg.start = 1.0f;
  seg.end = 3.0f;
  seg.text = "Test";
  segments.push_back(seg);

  std::vector<std::map<std::string, float>> empty_chunks;
  int sampling_rate = 16000;

  auto restored = restore_speech_timestamps(segments, empty_chunks, sampling_rate);

  EXPECT_EQ(restored.size(), segments.size());
  EXPECT_EQ(restored[0].start, seg.start); // Should remain unchanged
  EXPECT_EQ(restored[0].end, seg.end);
}

// Test normalize_features function
TEST_F(WhisperModelUtilsTest, NormalizeFeatures) {
  auto normalized = normalize_features(sample_features);

  EXPECT_EQ(normalized.size(), sample_features.size());

  for (size_t i = 0; i < normalized.size(); ++i) {
    EXPECT_EQ(normalized[i].size(), sample_features[i].size());

    // Calculate mean and std dev of normalized row
    float sum = std::accumulate(normalized[i].begin(), normalized[i].end(), 0.0f);
    float mean = sum / normalized[i].size();

    // Mean should be close to 0 after normalization
    EXPECT_NEAR(mean, 0.0f, 1e-6f);

    // Calculate standard deviation
    float sq_sum = 0.0f;
    for (float val : normalized[i]) {
      sq_sum += (val - mean) * (val - mean);
    }
    float std_dev = std::sqrt(sq_sum / normalized[i].size());

    // Standard deviation should be close to 1 after normalization
    EXPECT_NEAR(std_dev, 1.0f, 1e-6f);
  }
}

TEST_F(WhisperModelUtilsTest, NormalizeFeaturesEmptyInput) {
  std::vector<std::vector<float>> empty_features;
  auto normalized = normalize_features(empty_features);

  EXPECT_TRUE(normalized.empty());
}

TEST_F(WhisperModelUtilsTest, NormalizeFeaturesConstantValues) {
  std::vector<std::vector<float>> constant_features = {
    {2.0f, 2.0f, 2.0f, 2.0f},
    {3.0f, 3.0f, 3.0f, 3.0f}
  };

  auto normalized = normalize_features(constant_features);

  EXPECT_EQ(normalized.size(), constant_features.size());

  // Constant values should result in zero after normalization (due to zero std dev handling)
  for (const auto& row : normalized) {
    for (float val : row) {
      EXPECT_EQ(val, 2.0f); // Should remain unchanged due to std_dev check
    }
  }
}

// Test apply_log_mel_spectrogram function
TEST_F(WhisperModelUtilsTest, ApplyLogMelSpectrogram) {
  auto log_mel = apply_log_mel_spectrogram(sample_features);

  EXPECT_EQ(log_mel.size(), sample_features.size());

  for (size_t i = 0; i < log_mel.size(); ++i) {
    EXPECT_EQ(log_mel[i].size(), sample_features[i].size());

    for (size_t j = 0; j < log_mel[i].size(); ++j) {
      // Log values should be negative for values < 1
      if (sample_features[i][j] < 1.0f) {
        EXPECT_LT(log_mel[i][j], 0.0f);
      }

      // Verify log transformation: log_mel[i][j] = log(max(sample_features[i][j], 1e-10))
      float expected = std::log(std::max(sample_features[i][j], 1e-10f));
      EXPECT_NEAR(log_mel[i][j], expected, 1e-6f);
    }
  }
}

TEST_F(WhisperModelUtilsTest, ApplyLogMelSpectrogramZeroValues) {
  std::vector<std::vector<float>> zero_features = {
    {0.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 0.0f}
  };

  auto log_mel = apply_log_mel_spectrogram(zero_features);

  for (const auto& row : log_mel) {
    for (float val : row) {
      EXPECT_EQ(val, std::log(1e-10f)); // Should use minimum threshold
    }
  }
}

// Test calculate_signal_to_noise_ratio function
TEST_F(WhisperModelUtilsTest, CalculateSignalToNoiseRatio) {
  float snr = calculate_signal_to_noise_ratio(sample_audio);

  EXPECT_GE(snr, 0.0f); // SNR should be non-negative
  EXPECT_LT(snr, 100.0f); // Should be reasonable
}

TEST_F(WhisperModelUtilsTest, CalculateSignalToNoiseRatioEmptyAudio) {
  std::vector<float> empty_audio;
  float snr = calculate_signal_to_noise_ratio(empty_audio);

  EXPECT_EQ(snr, 0.0f); // Should return 0 for empty audio
}

TEST_F(WhisperModelUtilsTest, CalculateSignalToNoiseRatioHighSignal) {
  std::vector<float> high_signal_audio = {1.0f, -1.0f, 0.8f, -0.8f, 0.9f, -0.9f};
  float snr = calculate_signal_to_noise_ratio(high_signal_audio);

  EXPECT_GT(snr, 20.0f); // High signal should have high SNR
}

// Test is_silent_segment function
TEST_F(WhisperModelUtilsTest, IsSilentSegmentTrue) {
  std::vector<float> silent_audio = {0.001f, -0.002f, 0.003f, -0.001f};
  bool is_silent = is_silent_segment(silent_audio, 0.01f);

  EXPECT_TRUE(is_silent);
}

TEST_F(WhisperModelUtilsTest, IsSilentSegmentFalse) {
  std::vector<float> loud_audio = {0.1f, -0.2f, 0.15f, -0.1f};
  bool is_silent = is_silent_segment(loud_audio, 0.01f);

  EXPECT_FALSE(is_silent);
}

TEST_F(WhisperModelUtilsTest, IsSilentSegmentEmpty) {
  std::vector<float> empty_audio;
  bool is_silent = is_silent_segment(empty_audio);

  EXPECT_TRUE(is_silent); // Empty audio should be considered silent
}

TEST_F(WhisperModelUtilsTest, IsSilentSegmentThreshold) {
  std::vector<float> borderline_audio = {0.015f, 0.005f, -0.01f, 0.008f};

  bool silent_low_threshold = is_silent_segment(borderline_audio, 0.02f);
  bool silent_high_threshold = is_silent_segment(borderline_audio, 0.005f);

  EXPECT_TRUE(silent_low_threshold); // Should be silent with high threshold
  EXPECT_FALSE(silent_high_threshold); // Should not be silent with low threshold
}

// Test detect_speech_activity function
TEST_F(WhisperModelUtilsTest, DetectSpeechActivity) {
  int sampling_rate = 16000;
  float frame_duration = 0.025f;
  float energy_threshold = 0.01f;

  auto speech_segments = detect_speech_activity(sample_audio, sampling_rate, frame_duration, energy_threshold);

  // Should return valid segments
  for (const auto& segment : speech_segments) {
    EXPECT_GE(segment.first, 0.0f); // Start time should be non-negative
    EXPECT_GT(segment.second, segment.first); // End should be after start
  }
}

TEST_F(WhisperModelUtilsTest, DetectSpeechActivitySilentAudio) {
  std::vector<float> silent_audio(1600, 0.001f); // 0.1 seconds of near-silence
  int sampling_rate = 16000;

  auto speech_segments = detect_speech_activity(silent_audio, sampling_rate);

  EXPECT_TRUE(speech_segments.empty()); // Should detect no speech in silent audio
}

TEST_F(WhisperModelUtilsTest, DetectSpeechActivityLoudAudio) {
  std::vector<float> loud_audio(1600, 0.1f); // 0.1 seconds of loud audio
  int sampling_rate = 16000;

  auto speech_segments = detect_speech_activity(loud_audio, sampling_rate);

  EXPECT_FALSE(speech_segments.empty()); // Should detect speech in loud audio
  EXPECT_NEAR(speech_segments[0].first, 0.0f, 0.05f); // Should start near beginning
}

TEST_F(WhisperModelUtilsTest, DetectSpeechActivityEmptyAudio) {
  std::vector<float> empty_audio;
  int sampling_rate = 16000;

  auto speech_segments = detect_speech_activity(empty_audio, sampling_rate);

  EXPECT_TRUE(speech_segments.empty()); // Should return no segments for empty audio
}

// Test edge cases and integration
TEST_F(WhisperModelUtilsTest, IntegrationFeatureProcessingPipeline) {
  // Test complete feature processing pipeline
  auto sliced = slice_features(sample_features, 1, 3);
  EXPECT_FALSE(sliced.empty());

  auto padded = pad_or_trim(sliced);
  EXPECT_EQ(padded[0].size(), 3000);

  auto normalized = normalize_features(padded);
  EXPECT_EQ(normalized.size(), padded.size());

  auto log_mel = apply_log_mel_spectrogram(normalized);
  EXPECT_EQ(log_mel.size(), normalized.size());

  auto storage = get_ctranslate2_storage(log_mel);
  EXPECT_EQ(storage.shape()[0], static_cast<long>(log_mel.size()));
  EXPECT_EQ(storage.shape()[1], 3000);
}

TEST_F(WhisperModelUtilsTest, IntegrationAudioProcessingPipeline) {
  // Test complete audio processing pipeline
  float snr = calculate_signal_to_noise_ratio(sample_audio);
  EXPECT_GE(snr, 0.0f);

  bool is_silent = is_silent_segment(sample_audio);
  EXPECT_FALSE(is_silent); // Sample audio should not be silent

  auto speech_segments = detect_speech_activity(sample_audio, 16000);
  // Should return reasonable results for sample audio
}

TEST_F(WhisperModelUtilsTest, ErrorHandlingRobustness) {
  // Test error handling across utility functions
  std::vector<std::vector<float>> empty_features;

  EXPECT_NO_THROW({
    auto sliced = slice_features(empty_features, 0, 5);
    EXPECT_TRUE(sliced.empty());
  });

  EXPECT_NO_THROW({
    auto padded = pad_or_trim(empty_features);
    EXPECT_TRUE(padded.empty());
  });

  EXPECT_NO_THROW({
    auto normalized = normalize_features(empty_features);
    EXPECT_TRUE(normalized.empty());
  });

  std::string empty_text = "";
  EXPECT_NO_THROW({
    float ratio = get_compression_ratio(empty_text);
    EXPECT_EQ(ratio, 1.0f);
  });
}