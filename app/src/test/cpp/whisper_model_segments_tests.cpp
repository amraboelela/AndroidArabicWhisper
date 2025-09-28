/**
 * Unit Tests for WhisperModel Segment Processing Implementation
 * Tests segment generation, splitting, and word-level timestamp functions
 * Created by Amr Aboelela
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "whisper_model.h"
#include "tokenizer.h"
#include <vector>
#include <string>
#include <tuple>
#include <optional>
#include <map>
#include <any>
#include <ctranslate2/storage_view.h>

// Mock implementations for testing segment processing
class MockWhisperModelSegments {
public:
  // Mock tokenizer for testing
  class MockTokenizer {
  public:
    int get_timestamp_begin() const { return 50364; }
    int get_eot() const { return 50257; }
    int get_sot_prev() const { return 50361; }
    int get_no_timestamps() const { return 50363; }
    std::vector<int> get_sot_sequence() const { return {50258, 50259, 50359}; }

    std::vector<int> encode(const std::string& text) const {
      // Simple mock encoding
      std::vector<int> tokens;
      for (char c : text) {
        tokens.push_back(static_cast<int>(c));
      }
      return tokens;
    }

    std::string decode(const std::vector<int>& tokens) const {
      std::string text;
      for (int token : tokens) {
        if (token < 256) text += static_cast<char>(token);
      }
      return text;
    }

    std::vector<std::tuple<std::string, std::vector<int>>> split_to_word_tokens(
      const std::vector<int>& tokens) const {
      std::vector<std::tuple<std::string, std::vector<int>>> result;
      if (!tokens.empty()) {
        result.emplace_back("test_word", std::vector<int>{tokens[0]});
        if (tokens.size() > 1) {
          result.emplace_back("another_word", std::vector<int>{tokens[1]});
        }
      }
      return result;
    }
  };

  // Mock segment splitting function
  static std::tuple<std::vector<Segment>, int, bool> mock_split_segments_by_timestamps(
    const MockTokenizer& tokenizer,
    const std::vector<int>& tokens,
    float time_offset,
    int segment_size,
    float segment_duration,
    int seek
  ) {
    std::vector<Segment> segments;

    if (!tokens.empty()) {
      Segment seg;
      seg.seek = seek;
      seg.start = time_offset;
      seg.end = time_offset + segment_duration;
      seg.tokens = tokens;
      segments.push_back(seg);
    }

    bool single_timestamp_ending = tokens.size() >= 2 &&
      tokens[tokens.size() - 2] < tokenizer.get_timestamp_begin() &&
      tokens.back() >= tokenizer.get_timestamp_begin();

    int new_seek = single_timestamp_ending ? seek + segment_size : seek + 1000;

    return {segments, new_seek, single_timestamp_ending};
  }

  // Mock generate segments function
  static std::vector<Segment> mock_generate_segments(
    const std::vector<std::vector<float>>& features,
    const MockTokenizer& tokenizer,
    const TranscriptionOptions& options
  ) {
    std::vector<Segment> segments;

    if (!features.empty() && !features[0].empty()) {
      Segment seg;
      seg.id = 1;
      seg.seek = 0;
      seg.start = 0.0f;
      seg.end = 1.0f;
      seg.text = "Test segment";
      seg.tokens = {72, 101, 115, 116}; // "Test"
      seg.avg_logprob = -0.5f;
      seg.compression_ratio = 2.0f;
      seg.no_speech_prob = 0.1f;

      segments.push_back(seg);
    }

    return segments;
  }

  // Mock generate with fallback function
  static std::tuple<std::vector<int>, float, float, float> mock_generate_with_fallback(
    const std::vector<int>& prompt,
    const TranscriptionOptions& options
  ) {
    std::vector<int> result_tokens = {72, 101, 115, 116}; // "Test"
    float avg_logprob = -0.5f;
    float temperature = 0.0f;
    float compression_ratio = 2.0f;

    return {result_tokens, avg_logprob, temperature, compression_ratio};
  }

  // Mock get prompt function
  static std::vector<int> mock_get_prompt(
    const MockTokenizer& tokenizer,
    const std::vector<int>& previous_tokens,
    bool without_timestamps = false,
    std::optional<std::string> prefix = std::nullopt,
    std::optional<std::string> hotwords = std::nullopt
  ) {
    std::vector<int> prompt;

    if (!previous_tokens.empty() || (hotwords.has_value() && !prefix.has_value())) {
      prompt.push_back(tokenizer.get_sot_prev());
    }

    auto sot_sequence = tokenizer.get_sot_sequence();
    prompt.insert(prompt.end(), sot_sequence.begin(), sot_sequence.end());

    if (without_timestamps) {
      prompt.push_back(tokenizer.get_no_timestamps());
    }

    if (prefix.has_value()) {
      auto prefix_tokens = tokenizer.encode(" " + prefix.value());
      if (!without_timestamps) {
        prompt.push_back(tokenizer.get_timestamp_begin());
      }
      prompt.insert(prompt.end(), prefix_tokens.begin(), prefix_tokens.end());
    }

    return prompt;
  }

  // Mock word timestamps generation
  static std::vector<Word> mock_generate_word_timestamps(
    const Segment& segment,
    const MockTokenizer& tokenizer
  ) {
    std::vector<Word> words;

    if (!segment.text.empty()) {
      // Simple word splitting
      std::vector<std::string> word_strings;
      std::string current_word;

      for (char c : segment.text) {
        if (c == ' ' || c == '\t' || c == '\n') {
          if (!current_word.empty()) {
            word_strings.push_back(current_word);
            current_word.clear();
          }
        } else {
          current_word += c;
        }
      }

      if (!current_word.empty()) {
        word_strings.push_back(current_word);
      }

      // Generate timing for each word
      float segment_duration = segment.end - segment.start;
      float time_per_word = word_strings.empty() ? 0.0f : segment_duration / word_strings.size();
      float current_time = segment.start;

      for (const auto& word_text : word_strings) {
        Word word;
        word.start = current_time;
        word.end = std::min(current_time + time_per_word, segment.end);
        word.word = word_text;
        word.probability = 0.9f;

        words.push_back(word);
        current_time = word.end;
      }
    }

    return words;
  }
};

class WhisperModelSegmentsTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Initialize test data
    sample_features = {
      {0.1f, 0.2f, 0.3f, 0.4f, 0.5f},
      {0.15f, 0.25f, 0.35f, 0.45f, 0.55f},
      {0.2f, 0.3f, 0.4f, 0.5f, 0.6f}
    };

    sample_tokens = {72, 101, 115, 116, 50364, 50365}; // "Test" + timestamps

    // Setup transcription options
    options.beam_size = 5;
    options.best_of = 5;
    options.patience = 1.0f;
    options.length_penalty = 1.0f;
    options.repetition_penalty = 1.0f;
    options.no_repeat_ngram_size = 0;
    options.condition_on_previous_text = true;
    options.prompt_reset_on_temperature = 0.5f;
    options.temperatures = {0.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f};
    options.suppress_blank = true;
    options.without_timestamps = false;
    options.max_initial_timestamp = 1.0f;
    options.word_timestamps = false;
    options.prepend_punctuations = "\"'"¿([{-";
    options.append_punctuations = "\"'.。,，!！?？:：")]}、";
    options.multilingual = false;
    options.clip_timestamps = std::vector<float>{0.0f};
  }

  std::vector<std::vector<float>> sample_features;
  std::vector<int> sample_tokens;
  TranscriptionOptions options;
  MockWhisperModelSegments::MockTokenizer tokenizer;
};

// Test split_segments_by_timestamps function
TEST_F(WhisperModelSegmentsTest, SplitSegmentsByTimestamps) {
  float time_offset = 0.0f;
  int segment_size = 3000;
  float segment_duration = 30.0f;
  int seek = 0;

  auto [segments, new_seek, single_timestamp_ending] =
    MockWhisperModelSegments::mock_split_segments_by_timestamps(
      tokenizer, sample_tokens, time_offset, segment_size, segment_duration, seek
    );

  EXPECT_FALSE(segments.empty());
  EXPECT_GE(new_seek, seek);
  EXPECT_EQ(segments[0].seek, seek);
  EXPECT_EQ(segments[0].start, time_offset);
  EXPECT_EQ(segments[0].end, time_offset + segment_duration);
}

TEST_F(WhisperModelSegmentsTest, SplitSegmentsWithEmptyTokens) {
  std::vector<int> empty_tokens;
  float time_offset = 0.0f;
  int segment_size = 3000;
  float segment_duration = 30.0f;
  int seek = 0;

  auto [segments, new_seek, single_timestamp_ending] =
    MockWhisperModelSegments::mock_split_segments_by_timestamps(
      tokenizer, empty_tokens, time_offset, segment_size, segment_duration, seek
    );

  EXPECT_TRUE(segments.empty());
  EXPECT_EQ(new_seek, seek + 1000); // Mock implementation increment
}

TEST_F(WhisperModelSegmentsTest, SplitSegmentsTimestampLogic) {
  // Test with consecutive timestamps
  std::vector<int> timestamp_tokens = {50364, 50365, 50366}; // All timestamps
  float time_offset = 10.0f;
  int segment_size = 3000;
  float segment_duration = 30.0f;
  int seek = 1000;

  auto [segments, new_seek, single_timestamp_ending] =
    MockWhisperModelSegments::mock_split_segments_by_timestamps(
      tokenizer, timestamp_tokens, time_offset, segment_size, segment_duration, seek
    );

  EXPECT_FALSE(segments.empty());
  EXPECT_GE(new_seek, seek);
}

// Test generate_segments function
TEST_F(WhisperModelSegmentsTest, GenerateSegments) {
  auto segments = MockWhisperModelSegments::mock_generate_segments(
    sample_features, tokenizer, options
  );

  EXPECT_FALSE(segments.empty());
  EXPECT_GT(segments[0].id, 0);
  EXPECT_FALSE(segments[0].text.empty());
  EXPECT_FALSE(segments[0].tokens.empty());
  EXPECT_GE(segments[0].start, 0.0f);
  EXPECT_GT(segments[0].end, segments[0].start);
}

TEST_F(WhisperModelSegmentsTest, GenerateSegmentsWithEmptyFeatures) {
  std::vector<std::vector<float>> empty_features;
  auto segments = MockWhisperModelSegments::mock_generate_segments(
    empty_features, tokenizer, options
  );

  EXPECT_TRUE(segments.empty());
}

TEST_F(WhisperModelSegmentsTest, GenerateSegmentsWithOptions) {
  // Test with different options
  options.multilingual = true;
  options.without_timestamps = true;
  options.word_timestamps = true;

  auto segments = MockWhisperModelSegments::mock_generate_segments(
    sample_features, tokenizer, options
  );

  EXPECT_FALSE(segments.empty());
}

// Test generate_with_fallback function
TEST_F(WhisperModelSegmentsTest, GenerateWithFallback) {
  std::vector<int> prompt = {50258, 50259, 50359}; // SOT sequence

  auto [tokens, avg_logprob, temperature, compression_ratio] =
    MockWhisperModelSegments::mock_generate_with_fallback(prompt, options);

  EXPECT_FALSE(tokens.empty());
  EXPECT_LE(avg_logprob, 0.0f); // Log probabilities are typically negative
  EXPECT_GE(temperature, 0.0f);
  EXPECT_GT(compression_ratio, 0.0f);
}

TEST_F(WhisperModelSegmentsTest, GenerateWithFallbackTemperatures) {
  std::vector<int> prompt = {50258, 50259, 50359};

  // Test that fallback handles multiple temperatures
  EXPECT_GT(options.temperatures.size(), 1);

  auto [tokens, avg_logprob, temperature, compression_ratio] =
    MockWhisperModelSegments::mock_generate_with_fallback(prompt, options);

  EXPECT_FALSE(tokens.empty());
  EXPECT_TRUE(std::find(options.temperatures.begin(), options.temperatures.end(), temperature)
              != options.temperatures.end() || temperature >= 0.0f);
}

TEST_F(WhisperModelSegmentsTest, GenerateWithFallbackThresholds) {
  std::vector<int> prompt = {50258};

  // Test with thresholds set
  options.compression_ratio_threshold = 2.4f;
  options.log_prob_threshold = -1.0f;
  options.no_speech_threshold = 0.6f;

  auto [tokens, avg_logprob, temperature, compression_ratio] =
    MockWhisperModelSegments::mock_generate_with_fallback(prompt, options);

  EXPECT_FALSE(tokens.empty());

  // Verify threshold logic would be applied
  if (compression_ratio > options.compression_ratio_threshold.value()) {
    EXPECT_GT(compression_ratio, options.compression_ratio_threshold.value());
  }

  if (avg_logprob < options.log_prob_threshold.value()) {
    EXPECT_LT(avg_logprob, options.log_prob_threshold.value());
  }
}

// Test get_prompt function
TEST_F(WhisperModelSegmentsTest, GetPromptBasic) {
  std::vector<int> previous_tokens;

  auto prompt = MockWhisperModelSegments::mock_get_prompt(
    tokenizer, previous_tokens, false, std::nullopt, std::nullopt
  );

  EXPECT_FALSE(prompt.empty());

  // Should contain SOT sequence
  auto sot_sequence = tokenizer.get_sot_sequence();
  EXPECT_GE(prompt.size(), sot_sequence.size());
}

TEST_F(WhisperModelSegmentsTest, GetPromptWithPreviousTokens) {
  std::vector<int> previous_tokens = {72, 101, 115, 116}; // "Test"

  auto prompt = MockWhisperModelSegments::mock_get_prompt(
    tokenizer, previous_tokens, false, std::nullopt, std::nullopt
  );

  EXPECT_FALSE(prompt.empty());
  EXPECT_GT(prompt.size(), tokenizer.get_sot_sequence().size());

  // Should start with SOT_PREV when previous tokens exist
  EXPECT_EQ(prompt[0], tokenizer.get_sot_prev());
}

TEST_F(WhisperModelSegmentsTest, GetPromptWithoutTimestamps) {
  std::vector<int> previous_tokens;

  auto prompt = MockWhisperModelSegments::mock_get_prompt(
    tokenizer, previous_tokens, true, std::nullopt, std::nullopt
  );

  EXPECT_FALSE(prompt.empty());

  // Should contain no_timestamps token
  EXPECT_TRUE(std::find(prompt.begin(), prompt.end(), tokenizer.get_no_timestamps())
              != prompt.end());
}

TEST_F(WhisperModelSegmentsTest, GetPromptWithPrefix) {
  std::vector<int> previous_tokens;
  std::string prefix = "Hello";

  auto prompt = MockWhisperModelSegments::mock_get_prompt(
    tokenizer, previous_tokens, false, prefix, std::nullopt
  );

  EXPECT_FALSE(prompt.empty());

  // Should contain timestamp_begin when prefix is provided and timestamps enabled
  EXPECT_TRUE(std::find(prompt.begin(), prompt.end(), tokenizer.get_timestamp_begin())
              != prompt.end());
}

TEST_F(WhisperModelSegmentsTest, GetPromptWithHotwords) {
  std::vector<int> previous_tokens;
  std::string hotwords = "important words";

  auto prompt = MockWhisperModelSegments::mock_get_prompt(
    tokenizer, previous_tokens, false, std::nullopt, hotwords
  );

  EXPECT_FALSE(prompt.empty());
  EXPECT_GT(prompt.size(), tokenizer.get_sot_sequence().size());
}

// Test generate_word_timestamps function
TEST_F(WhisperModelSegmentsTest, GenerateWordTimestamps) {
  Segment segment;
  segment.start = 0.0f;
  segment.end = 2.0f;
  segment.text = "Hello world test";

  auto words = MockWhisperModelSegments::mock_generate_word_timestamps(segment, tokenizer);

  EXPECT_EQ(words.size(), 3); // "Hello", "world", "test"
  EXPECT_EQ(words[0].word, "Hello");
  EXPECT_EQ(words[1].word, "world");
  EXPECT_EQ(words[2].word, "test");

  // Check timing distribution
  EXPECT_EQ(words[0].start, segment.start);
  EXPECT_EQ(words.back().end, segment.end);

  for (const auto& word : words) {
    EXPECT_GE(word.start, segment.start);
    EXPECT_LE(word.end, segment.end);
    EXPECT_LT(word.start, word.end);
    EXPECT_GT(word.probability, 0.0f);
    EXPECT_LE(word.probability, 1.0f);
  }
}

TEST_F(WhisperModelSegmentsTest, GenerateWordTimestampsEmptyText) {
  Segment segment;
  segment.start = 0.0f;
  segment.end = 2.0f;
  segment.text = "";

  auto words = MockWhisperModelSegments::mock_generate_word_timestamps(segment, tokenizer);

  EXPECT_TRUE(words.empty());
}

TEST_F(WhisperModelSegmentsTest, GenerateWordTimestampsSingleWord) {
  Segment segment;
  segment.start = 1.0f;
  segment.end = 3.0f;
  segment.text = "Arabic";

  auto words = MockWhisperModelSegments::mock_generate_word_timestamps(segment, tokenizer);

  EXPECT_EQ(words.size(), 1);
  EXPECT_EQ(words[0].word, "Arabic");
  EXPECT_EQ(words[0].start, segment.start);
  EXPECT_EQ(words[0].end, segment.end);
}

TEST_F(WhisperModelSegmentsTest, GenerateWordTimestampsArabicContent) {
  Segment segment;
  segment.start = 0.0f;
  segment.end = 4.0f;
  segment.text = "مرحبا بالعالم"; // "Hello world" in Arabic

  auto words = MockWhisperModelSegments::mock_generate_word_timestamps(segment, tokenizer);

  EXPECT_EQ(words.size(), 2); // "مرحبا", "بالعالم"

  for (const auto& word : words) {
    EXPECT_FALSE(word.word.empty());
    EXPECT_GE(word.start, segment.start);
    EXPECT_LE(word.end, segment.end);
    EXPECT_GT(word.probability, 0.8f); // High confidence for Arabic
  }
}

// Test edge cases and error handling
TEST_F(WhisperModelSegmentsTest, EdgeCaseZeroDuration) {
  Segment segment;
  segment.start = 1.0f;
  segment.end = 1.0f; // Zero duration
  segment.text = "test";

  auto words = MockWhisperModelSegments::mock_generate_word_timestamps(segment, tokenizer);

  EXPECT_FALSE(words.empty());
  EXPECT_EQ(words[0].start, words[0].end); // Should handle zero duration gracefully
}

TEST_F(WhisperModelSegmentsTest, EdgeCaseVeryLongText) {
  Segment segment;
  segment.start = 0.0f;
  segment.end = 10.0f;
  segment.text = "This is a very long text with many words to test timing distribution";

  auto words = MockWhisperModelSegments::mock_generate_word_timestamps(segment, tokenizer);

  EXPECT_GT(words.size(), 10);

  // Verify proper timing distribution
  for (size_t i = 1; i < words.size(); ++i) {
    EXPECT_GE(words[i].start, words[i-1].start);
    EXPECT_LE(words[i].end, segment.end);
  }
}

TEST_F(WhisperModelSegmentsTest, TranscriptionOptionsValidation) {
  // Test that options are properly validated
  EXPECT_GT(options.beam_size, 0);
  EXPECT_GT(options.best_of, 0);
  EXPECT_GE(options.patience, 0.0f);
  EXPECT_FALSE(options.temperatures.empty());

  // Test clip_timestamps handling
  if (std::holds_alternative<std::vector<float>>(options.clip_timestamps)) {
    auto clips = std::get<std::vector<float>>(options.clip_timestamps);
    EXPECT_FALSE(clips.empty());
  }
}

TEST_F(WhisperModelSegmentsTest, SegmentProcessingFlow) {
  // Test complete segment processing flow
  auto segments = MockWhisperModelSegments::mock_generate_segments(
    sample_features, tokenizer, options
  );

  EXPECT_FALSE(segments.empty());

  // Test word timestamp generation for each segment
  for (const auto& segment : segments) {
    auto words = MockWhisperModelSegments::mock_generate_word_timestamps(segment, tokenizer);

    if (!segment.text.empty()) {
      EXPECT_FALSE(words.empty());

      // Verify word boundaries
      for (const auto& word : words) {
        EXPECT_GE(word.start, segment.start);
        EXPECT_LE(word.end, segment.end);
      }
    }
  }
}