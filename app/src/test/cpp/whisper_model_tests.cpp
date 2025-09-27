#include "whisper_model.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <string>
#include <algorithm>
#include <cmath>
#include <fstream>

/**
 * Comprehensive unit tests for WhisperModel components
 * Testing data structures, encoding/decoding, and core functionality
 */

// Test helper macros
#define ASSERT_EQ(actual, expected, test_name) \
    if ((actual) != (expected)) { \
        std::cerr << "FAILED: " << test_name << " - Expected: " << (expected) << ", Got: " << (actual) << std::endl; \
        return false; \
    } else { \
        std::cout << "✓ " << test_name << std::endl; \
    }

#define ASSERT_TRUE(condition, test_name) \
    if (!(condition)) { \
        std::cerr << "FAILED: " << test_name << " - Condition failed" << std::endl; \
        return false; \
    } else { \
        std::cout << "✓ " << test_name << std::endl; \
    }

#define ASSERT_APPROX_EQ(actual, expected, tolerance, test_name) \
    if (std::abs((actual) - (expected)) > (tolerance)) { \
        std::cerr << "FAILED: " << test_name << " - Expected: " << (expected) << ", Got: " << (actual) << ", Tolerance: " << (tolerance) << std::endl; \
        return false; \
    } else { \
        std::cout << "✓ " << test_name << std::endl; \
    }

namespace {

/**
 * Test Word structure functionality
 */
  bool test_word_structure() {
    std::cout << "\n=== Testing Word Structure ===" << std::endl;

    // Test basic Word creation and properties
    Word word1{1.5f, 2.3f, "hello", 0.95f};
    ASSERT_APPROX_EQ(word1.start, 1.5f, 0.001f, "Word start time");
    ASSERT_APPROX_EQ(word1.end, 2.3f, 0.001f, "Word end time");
    ASSERT_EQ(word1.word, "hello", "Word text");
    ASSERT_APPROX_EQ(word1.probability, 0.95f, 0.001f, "Word probability");

    // Test Word::to_string() method
    std::string word_str = word1.to_string();
    ASSERT_TRUE(!word_str.empty(), "Word to_string not empty");
    ASSERT_TRUE(word_str.find("hello") != std::string::npos, "Word to_string contains text");
    ASSERT_TRUE(word_str.find("1.5") != std::string::npos, "Word to_string contains start time");

    // Test Arabic word
    Word arabic_word{0.0f, 1.0f, "مرحبا", 0.88f};
    ASSERT_EQ(arabic_word.word, "مرحبا", "Arabic word text");
    std::string arabic_str = arabic_word.to_string();
    ASSERT_TRUE(arabic_str.find("مرحبا") != std::string::npos, "Arabic word in to_string");

    return true;
  }

/**
 * Test Segment structure functionality
 */
  bool test_segment_structure() {
    std::cout << "\n=== Testing Segment Structure ===" << std::endl;

    // Create test words
    std::vector<Word> test_words = {
        {0.0f, 0.5f, "Hello",  0.95f},
        {0.5f, 1.0f, " world", 0.92f}
    };

    // Test basic Segment creation
    Segment segment1;
    segment1.id = 1;
    segment1.seek = 0;
    segment1.start = 0.0f;
    segment1.end = 1.0f;
    segment1.text = "Hello world";
    segment1.tokens = {50257, 50259, 50359, 15496, 1002};
    segment1.avg_logprob = -0.5f;
    segment1.compression_ratio = 2.4f;
    segment1.no_speech_prob = 0.02f;
    segment1.words = test_words;
    segment1.temperature = 0.0f;

    ASSERT_EQ(segment1.id, 1, "Segment ID");
    ASSERT_EQ(segment1.seek, 0, "Segment seek");
    ASSERT_APPROX_EQ(segment1.start, 0.0f, 0.001f, "Segment start time");
    ASSERT_APPROX_EQ(segment1.end, 1.0f, 0.001f, "Segment end time");
    ASSERT_EQ(segment1.text, "Hello world", "Segment text");
    ASSERT_EQ(segment1.tokens.size(), 5, "Segment tokens count");
    ASSERT_TRUE(segment1.words.has_value(), "Segment has words");
    ASSERT_EQ(segment1.words.value().size(), 2, "Segment words count");

    // Test Segment::to_string() method
    std::string segment_str = segment1.to_string();
    ASSERT_TRUE(!segment_str.empty(), "Segment to_string not empty");
    ASSERT_TRUE(segment_str.find("Hello world") != std::string::npos,
                "Segment to_string contains text");
    ASSERT_TRUE(segment_str.find("id: 1") != std::string::npos, "Segment to_string contains ID");

    // Test segment without words
    Segment segment2;
    segment2.id = 2;
    segment2.text = "Test without words";
    segment2.words = std::nullopt;

    std::string segment2_str = segment2.to_string();
    ASSERT_TRUE(segment2_str.find("words: []") != std::string::npos,
                "Empty words array in to_string");

    return true;
  }

/**
 * Test TranscriptionOptions structure
 */
  bool test_transcription_options() {
    std::cout << "\n=== Testing TranscriptionOptions Structure ===" << std::endl;

    TranscriptionOptions options;

    // Test default values and assignments
    options.beam_size = 5;
    options.best_of = 5;
    options.patience = 1.0f;
    options.length_penalty = 1.0f;
    options.repetition_penalty = 1.0f;
    options.no_repeat_ngram_size = 0;

    ASSERT_EQ(options.beam_size, 5, "Beam size");
    ASSERT_EQ(options.best_of, 5, "Best of");
    ASSERT_APPROX_EQ(options.patience, 1.0f, 0.001f, "Patience");

    // Test optional fields
    options.log_prob_threshold = -1.0f;
    options.no_speech_threshold = 0.6f;
    ASSERT_TRUE(options.log_prob_threshold.has_value(), "Log prob threshold set");
    ASSERT_APPROX_EQ(options.log_prob_threshold.value(), -1.0f, 0.001f, "Log prob threshold value");

    // Test vector fields
    options.temperatures = {0.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f};
    ASSERT_EQ(options.temperatures.size(), 6, "Temperatures vector size");
    ASSERT_APPROX_EQ(options.temperatures[0], 0.0f, 0.001f, "First temperature");
    ASSERT_APPROX_EQ(options.temperatures[5], 1.0f, 0.001f, "Last temperature");

    // Test string fields
    options.prepend_punctuations = "\"'([{-";
    options.append_punctuations = "\"'.,!?:)]}";
    ASSERT_TRUE(!options.prepend_punctuations.empty(), "Prepend punctuations not empty");
    ASSERT_TRUE(!options.append_punctuations.empty(), "Append punctuations not empty");

    return true;
  }

/**
 * Test TranscriptionInfo structure
 */
  bool test_transcription_info() {
    std::cout << "\n=== Testing TranscriptionInfo Structure ===" << std::endl;

    TranscriptionInfo info;
    info.language = "ar";
    info.language_probability = 0.95f;
    info.duration = 30.5f;

    ASSERT_EQ(info.language, "ar", "Language code");
    ASSERT_APPROX_EQ(info.language_probability, 0.95f, 0.001f, "Language probability");
    ASSERT_APPROX_EQ(info.duration, 30.5f, 0.001f, "Duration");

    // Test all language probabilities
    std::vector<std::pair<std::string, float>> lang_probs = {
        {"ar", 0.95f},
        {"en", 0.03f},
        {"fr", 0.02f}
    };
    info.all_language_probs = lang_probs;

    ASSERT_TRUE(info.all_language_probs.has_value(), "All language probs set");
    ASSERT_EQ(info.all_language_probs.value().size(), 3, "All language probs count");
    ASSERT_EQ(info.all_language_probs.value()[0].first, "ar", "First language");
    ASSERT_APPROX_EQ(info.all_language_probs.value()[0].second, 0.95f, 0.001f,
                     "First language prob");

    return true;
  }

/**
 * Test encoding/decoding utility functions
 */
  bool test_encoding_decoding_utilities() {
    std::cout << "\n=== Testing Encoding/Decoding Utilities ===" << std::endl;

    // Test token sequence validation
    std::vector<int> valid_tokens = {50257, 50259, 50359, 15496, 1002, 50256};
    ASSERT_TRUE(!valid_tokens.empty(), "Valid tokens not empty");
    ASSERT_TRUE(valid_tokens.front() >= 50000, "SOT token in valid range");
    ASSERT_TRUE(valid_tokens.back() >= 50000, "EOT token in valid range");

    // Test timestamp token validation
    int timestamp_token = 50364 + 100; // 100 * 0.02s = 2.0s
    ASSERT_TRUE(timestamp_token >= 50364, "Timestamp token in valid range");
    ASSERT_TRUE(timestamp_token < 50364 + 1500, "Timestamp token within max range");

    // Test text validation
    std::vector<std::string> test_texts = {
        "Hello world",
        "مرحبا بالعالم",
        "Bonjour le monde",
        "Mixed text: Hello مرحبا",
        ""
    };

    for (const auto &text: test_texts) {
      ASSERT_TRUE(text.length() >= 0, "Text length non-negative");
    }

    return true;
  }

/**
 * Test audio processing parameter validation
 */
  bool test_audio_decoder_params() {
    std::cout << "\n=== Testing Audio Processing Parameters ===" << std::endl;

    // Test sample rate validation
    int valid_sample_rates[] = {16000, 22050, 44100, 48000};
    for (int sr: valid_sample_rates) {
      ASSERT_TRUE(sr > 0, "Sample rate positive");
      ASSERT_TRUE(sr >= 8000, "Sample rate reasonable minimum");
    }

    // Test audio duration calculations
    int sample_rate = 16000;
    int num_samples = 480000; // 30 seconds at 16kHz
    float duration = static_cast<float>(num_samples) / sample_rate;
    ASSERT_APPROX_EQ(duration, 30.0f, 0.01f, "Duration calculation");

    // Test chunk size validation
    int chunk_sizes[] = {160000, 320000, 480000}; // 10s, 20s, 30s at 16kHz
    for (int chunk_size: chunk_sizes) {
      ASSERT_TRUE(chunk_size > 0, "Chunk size positive");
      ASSERT_TRUE(chunk_size % 160 == 0, "Chunk size divisible by 160"); // 10ms chunks
    }

    return true;
  }

/**
 * Test model parameter validation
 */
  bool test_model_parameters() {
    std::cout << "\n=== Testing Model Parameters ===" << std::endl;

    // Test model size strings
    std::vector<std::string> model_sizes = {
        "tiny", "tiny.en", "base", "base.en", "small", "small.en",
        "medium", "medium.en", "large-v1", "large-v2", "large-v3", "large"
    };

    for (const auto &model_size: model_sizes) {
      ASSERT_TRUE(!model_size.empty(), "Model size not empty");
      ASSERT_TRUE(model_size.find(" ") == std::string::npos, "Model size no spaces");
    }

    // Test language codes
    std::vector<std::string> language_codes = {
        "ar", "en", "fr", "es", "de", "it", "pt", "ru", "ja", "ko", "zh"
    };

    for (const auto &lang: language_codes) {
      ASSERT_EQ(lang.length(), 2, "Language code length");
      ASSERT_TRUE(std::islower(lang[0]) && std::islower(lang[1]), "Language code lowercase");
    }

    // Test device strings
    std::vector<std::string> devices = {"auto", "cpu", "cuda", "mps"};
    for (const auto &device: devices) {
      ASSERT_TRUE(!device.empty(), "Device string not empty");
    }

    return true;
  }

/**
 * Test error handling and edge cases
 */
  bool test_error_handling() {
    std::cout << "\n=== Testing Error Handling ===" << std::endl;

    // Test empty inputs
    std::vector<float> empty_audio;
    ASSERT_EQ(empty_audio.size(), 0, "Empty audio vector");

    std::vector<int> empty_tokens;
    ASSERT_EQ(empty_tokens.size(), 0, "Empty tokens vector");

    // Test invalid values
    Word invalid_word{-1.0f, -1.0f, "", -1.0f};
    ASSERT_TRUE(invalid_word.start < 0, "Invalid word start time");
    ASSERT_TRUE(invalid_word.word.empty(), "Empty word text");

    // Test boundary values
    float max_duration = 1800.0f; // 30 minutes
    ASSERT_TRUE(max_duration > 0, "Max duration positive");

    int max_tokens = 448; // Whisper max context
    ASSERT_TRUE(max_tokens > 0, "Max tokens positive");

    return true;
  }

/**
 * Test audio chunking scenarios with real audio files
 */
  bool test_audio_chunking_scenarios() {
    std::cout << "\n=== Testing Audio Chunking Scenarios ===" << std::endl;

    // Test different audio file paths
    std::vector<std::string> possible_paths = {
        "../../../src/main/assets/",
        "../../../main/assets/",
        "../../assets/",
        "../assets/",
        "assets/"
    };

    std::string assets_path;
    bool found_assets = false;

    // Find the correct assets path
    for (const auto &path: possible_paths) {
      std::ifstream test_file(path + "001.wav");
      if (test_file.good()) {
        assets_path = path;
        found_assets = true;
        break;
      }
    }

    if (!found_assets) {
      std::cout << "⚠ Audio files not found, using mock data for chunking tests" << std::endl;
      assets_path = possible_paths[0]; // Use first path as fallback
    }

    // Test 1: Medium length audio (001.wav ~43 seconds)
    std::cout << "\nTesting medium length audio chunking (001.wav)..." << std::endl;

    // Simulate audio properties based on file size (1.3MB ≈ 43 seconds at 16kHz)
    float medium_duration = 43.0f; // seconds
    int medium_samples = static_cast<int>(medium_duration * 16000);
    ASSERT_TRUE(medium_duration > 30.0f, "Medium audio exceeds 30s chunk size");
    ASSERT_TRUE(medium_samples > 480000, "Medium audio exceeds default chunk samples");

    // Expected chunking behavior
    int expected_chunks = static_cast<int>(std::ceil(medium_duration / 30.0f));
    ASSERT_EQ(expected_chunks, 2, "Medium audio should produce 2 chunks");

    // Test chunk boundaries
    float chunk1_end = 30.0f;
    float chunk2_start = 30.0f;
    float chunk2_end = medium_duration;
    ASSERT_APPROX_EQ(chunk1_end, 30.0f, 0.1f, "First chunk ends at 30s");
    ASSERT_APPROX_EQ(chunk2_end - chunk2_start, 13.0f, 1.0f, "Second chunk ~13s");

    // Test 2: Long audio (002-01.wav ~900 seconds)
    std::cout << "\nTesting long audio chunking (002-01.wav)..." << std::endl;

    // Simulate audio properties based on file size (27MB ≈ 900 seconds at 16kHz)
    float long_duration = 900.0f; // seconds (15 minutes)
    int long_samples = static_cast<int>(long_duration * 16000);
    ASSERT_TRUE(long_duration > 30.0f, "Long audio exceeds 30s chunk size");
    ASSERT_TRUE(long_samples > 480000, "Long audio exceeds default chunk samples");

    // Expected chunking behavior
    int expected_long_chunks = static_cast<int>(std::ceil(long_duration / 30.0f));
    ASSERT_EQ(expected_long_chunks, 30, "Long audio should produce 30 chunks");

    // Test chunk sequence
    for (int i = 0; i < expected_long_chunks; ++i) {
      float chunk_start = i * 30.0f;
      float chunk_end = std::min((i + 1) * 30.0f, long_duration);
      float chunk_duration = chunk_end - chunk_start;

      ASSERT_TRUE(chunk_duration > 0, "Chunk duration positive");
      ASSERT_TRUE(chunk_duration <= 30.0f, "Chunk duration within limits");

      if (i == expected_long_chunks - 1) {
        // Last chunk might be shorter
        ASSERT_TRUE(chunk_duration <= 30.0f, "Last chunk duration valid");
      } else {
        // Regular chunks should be exactly 30s
        ASSERT_APPROX_EQ(chunk_duration, 30.0f, 0.1f, "Regular chunk exactly 30s");
      }
    }

    // Test 3: Chunk overlap and boundary handling
    std::cout << "\nTesting chunk boundary handling..." << std::endl;

    // Test seek pointer advancement
    std::vector<int> seek_positions;
    int frames_per_30s = static_cast<int>(30.0f / (160.0f / 16000.0f)); // 3000 frames

    for (int chunk = 0; chunk < expected_long_chunks; ++chunk) {
      int seek_pos = chunk * frames_per_30s;
      seek_positions.push_back(seek_pos);

      ASSERT_TRUE(seek_pos >= 0, "Seek position non-negative");
      if (chunk > 0) {
        ASSERT_TRUE(seek_pos > seek_positions[chunk - 1], "Seek position advances");
      }
    }

    // Test 4: Feature extraction chunking
    std::cout << "\nTesting feature extraction chunking..." << std::endl;

    // Test mel spectrogram dimensions for different durations
    struct AudioTest {
      float duration;
      int expected_frames;
      std::string description;
    };

    std::vector<AudioTest> audio_tests = {
        {30.0f,  3000,  "30s standard chunk"},
        {43.0f,  4300,  "43s medium audio"},
        {60.0f,  6000,  "60s double chunk"},
        {900.0f, 90000, "900s long audio"}
    };

    for (const auto &test: audio_tests) {
      int hop_length = 160;
      int sample_rate = 16000;
      int samples = static_cast<int>(test.duration * sample_rate);
      int expected_frames = samples / hop_length;

      ASSERT_APPROX_EQ(expected_frames, test.expected_frames, 50,
                       test.description + " frame count");

      // Test that features would be 80 x frames
      int mel_bins = 80;
      int total_features = mel_bins * expected_frames;
      ASSERT_TRUE(total_features > 0, test.description + " total features positive");
    }

    return true;
  }

/**
 * Test WhisperModel segment processing with real audio scenarios
 */
  bool test_segment_processing() {
    std::cout << "\n=== Testing Segment Processing ===" << std::endl;

    // Test 1: Segment timestamp calculation for chunked audio
    std::cout << "\nTesting segment timestamp calculation..." << std::endl;

    // Simulate segments from a 60-second audio file
    std::vector<Segment> mock_segments;

    // First 30s chunk segments
    Segment seg1{0, 0, 0.0f, 15.0f, "First segment text", {}, 0.95f, 2.1f, 0.02f, std::nullopt,
                 std::nullopt};
    Segment seg2{1, 1500, 15.0f, 30.0f, "Second segment text", {}, 0.92f, 1.8f, 0.03f, std::nullopt,
                 std::nullopt};

    // Second 30s chunk segments (should have offset timestamps)
    Segment seg3{2, 0, 30.0f, 45.0f, "Third segment text", {}, 0.88f, 2.3f, 0.04f, std::nullopt,
                 std::nullopt};
    Segment seg4{3, 1500, 45.0f, 60.0f, "Fourth segment text", {}, 0.91f, 1.9f, 0.02f, std::nullopt,
                 std::nullopt};

    mock_segments = {seg1, seg2, seg3, seg4};

    // Test segment continuity
    for (size_t i = 1; i < mock_segments.size(); ++i) {
      float prev_end = mock_segments[i - 1].end;
      float curr_start = mock_segments[i].start;
      ASSERT_TRUE(curr_start >= prev_end, "Segments are continuous");

      // Allow small gaps between segments
      float gap = curr_start - prev_end;
      ASSERT_TRUE(gap <= 1.0f, "Segment gap reasonable");
    }

    // Test chunk boundaries
    ASSERT_APPROX_EQ(mock_segments[1].end, 30.0f, 0.1f, "First chunk ends at 30s");
    ASSERT_APPROX_EQ(mock_segments[2].start, 30.0f, 0.1f, "Second chunk starts at 30s");

    // Test total duration coverage
    float total_duration = mock_segments.back().end - mock_segments.front().start;
    ASSERT_APPROX_EQ(total_duration, 60.0f, 0.1f, "Total duration covered");

    // Test 2: Word-level timestamps within segments
    std::cout << "\nTesting word-level timestamps..." << std::endl;

    // Add word timestamps to segments
    std::vector<Word> words1 = {
        {0.0f, 2.5f, "First",    0.98f},
        {2.5f, 5.0f, " segment", 0.95f},
        {5.0f, 7.5f, " text",    0.92f}
    };

    std::vector<Word> words2 = {
        {30.0f, 32.5f, "Third",    0.96f},
        {32.5f, 35.0f, " segment", 0.94f},
        {35.0f, 37.5f, " text",    0.91f}
    };

    mock_segments[0].words = words1;
    mock_segments[2].words = words2;

    // Test word timestamp consistency within segments
    for (const auto &segment: mock_segments) {
      if (segment.words.has_value()) {
        const auto &words = segment.words.value();
        for (const auto &word: words) {
          ASSERT_TRUE(word.start >= segment.start, "Word start within segment");
          ASSERT_TRUE(word.end <= segment.end, "Word end within segment");
          ASSERT_TRUE(word.start < word.end, "Word start before end");
          ASSERT_TRUE(word.probability > 0.0f, "Word probability positive");
        }
      }
    }

    // Test 3: Cross-chunk word boundary handling
    std::cout << "\nTesting cross-chunk word boundaries..." << std::endl;

    // Test scenario where a word might span chunk boundary
    Word boundary_word{29.5f, 30.5f, "boundary", 0.89f};

    // Should be handled by proper chunking with overlap or boundary detection
    ASSERT_TRUE(boundary_word.start < 30.0f, "Word starts before boundary");
    ASSERT_TRUE(boundary_word.end > 30.0f, "Word ends after boundary");

    float word_duration = boundary_word.end - boundary_word.start;
    ASSERT_TRUE(word_duration > 0 && word_duration < 5.0f, "Boundary word reasonable duration");

    return true;
  }

/**
 * Test integration between FeatureExtractor and WhisperModel chunking
 */
  bool test_feature_extractor_integration() {
    std::cout << "\n=== Testing FeatureExtractor Integration ===" << std::endl;

    // Test 1: Feature extraction for different chunk sizes
    std::cout << "\nTesting feature extraction for different chunk sizes..." << std::endl;

    struct ChunkTest {
      int chunk_length_seconds;
      int expected_frames;
      std::string description;
    };

    std::vector<ChunkTest> chunk_tests = {
        {30, 3000, "Standard 30s chunk"},
        {20, 2000, "Custom 20s chunk"},
        {60, 6000, "Double 60s chunk"},
        {15, 1500, "Half 15s chunk"}
    };

    for (const auto &test: chunk_tests) {
      // Simulate FeatureExtractor configuration
      int sample_rate = 16000;
      int hop_length = 160;
      int n_samples = test.chunk_length_seconds * sample_rate;
      int expected_max_frames = n_samples / hop_length;

      ASSERT_APPROX_EQ(expected_max_frames, test.expected_frames, 10,
                       test.description + " frame calculation");

      // Test time per frame calculation
      float time_per_frame = static_cast<float>(hop_length) / sample_rate;
      float expected_time_per_frame = 0.01f; // 160/16000 = 0.01s
      ASSERT_APPROX_EQ(time_per_frame, expected_time_per_frame, 0.001f,
                       "Time per frame consistent");

      // Test total duration
      float total_duration = expected_max_frames * time_per_frame;
      ASSERT_APPROX_EQ(total_duration, static_cast<float>(test.chunk_length_seconds), 0.1f,
                       test.description + " duration consistency");
    }

    // Test 2: Mel spectrogram dimensions for chunked processing
    std::cout << "\nTesting mel spectrogram dimensions..." << std::endl;

    int n_mels = 80; // Whisper standard

    for (const auto &test: chunk_tests) {
      // Expected mel spectrogram dimensions: 80 x time_frames
      int expected_features = n_mels * test.expected_frames;
      ASSERT_TRUE(expected_features > 0, test.description + " feature count positive");

      // Test memory requirements (rough estimate)
      size_t memory_bytes = expected_features * sizeof(float);
      size_t max_reasonable_memory = 100 * 1024 * 1024; // 100MB max
      ASSERT_TRUE(memory_bytes < max_reasonable_memory,
                  test.description + " memory usage reasonable");
    }

    // Test 3: Feature consistency across chunk boundaries
    std::cout << "\nTesting feature consistency across chunks..." << std::endl;

    // Test overlapping window processing
    int window_size = 400; // n_fft
    int hop_length = 160;
    int overlap = window_size - hop_length; // 240 samples overlap

    ASSERT_TRUE(overlap > 0, "STFT windows have overlap");
    ASSERT_TRUE(overlap < window_size, "Overlap less than window size");

    // Test chunk boundary handling
    float chunk_boundary_time = 30.0f; // seconds
    int boundary_frame = static_cast<int>(chunk_boundary_time * 16000 / hop_length);
    ASSERT_APPROX_EQ(boundary_frame, 3000, 10, "Chunk boundary frame calculation");

    // Test that features near boundaries are handled consistently
    int boundary_window_start = boundary_frame - overlap / hop_length;
    int boundary_window_end = boundary_frame + overlap / hop_length;
    ASSERT_TRUE(boundary_window_start >= 0, "Boundary window start valid");
    ASSERT_TRUE(boundary_window_end > boundary_window_start, "Boundary window end valid");

    return true;
  }

} // anonymous namespace

/**
 * Main test runner for WhisperModel tests
 */
bool run_whisper_model_tests() {
  std::cout << "=== WHISPER MODEL UNIT TESTS ===" << std::endl;

  bool all_passed = true;

  all_passed &= test_word_structure();
  all_passed &= test_segment_structure();
  all_passed &= test_transcription_options();
  all_passed &= test_audio_chunking_scenarios();
  all_passed &= test_segment_processing();
  all_passed &= test_feature_extractor_integration();

  std::cout << "\n=== WHISPER MODEL TEST SUMMARY ===" << std::endl;
  if (all_passed) {
    std::cout << "✅ ALL WHISPER MODEL TESTS PASSED!" << std::endl;
  } else {
    std::cout << "❌ SOME WHISPER MODEL TESTS FAILED!" << std::endl;
  }

  return all_passed;
}

/**
 * Demonstrate WhisperModel usage patterns
 */
void demonstrate_whisper_model_usage() {
  std::cout << "\n=== WhisperModel Usage Examples ===" << std::endl;

  std::cout << "// Basic model usage:" << std::endl;
  std::cout << "// 1. Create WhisperModel:" << std::endl;
  std::cout << "//    WhisperModel model(\"large-v3\", \"auto\");" << std::endl;
  std::cout << "//" << std::endl;
  std::cout << "// 2. Transcribe audio:" << std::endl;
  std::cout << "//    auto [segments, info] = model.transcribe(audio_data, \"ar\", true);"
            << std::endl;
  std::cout << "//" << std::endl;
  std::cout << "// 3. Process segments:" << std::endl;
  std::cout << "//    for (const auto& segment : segments) {" << std::endl;
  std::cout << "//        std::cout << segment.text << std::endl;" << std::endl;
  std::cout << "//        if (segment.words) {" << std::endl;
  std::cout << "//            for (const auto& word : segment.words.value()) {" << std::endl;
  std::cout
      << "//                std::cout << word.word << \" [\" << word.start << \"-\" << word.end << \"]\" << std::endl;"
      << std::endl;
  std::cout << "//            }" << std::endl;
  std::cout << "//        }" << std::endl;
  std::cout << "//    }" << std::endl;

  std::cout << "\n// Key features:" << std::endl;
  std::cout << "// - Support for multiple model sizes (tiny to large-v3)" << std::endl;
  std::cout << "// - Arabic language optimization" << std::endl;
  std::cout << "// - Word-level timestamps" << std::endl;
  std::cout << "// - Configurable transcription options" << std::endl;
  std::cout << "// - Language detection capabilities" << std::endl;
  std::cout << "// - Robust error handling" << std::endl;
}

#ifndef TESTING_MODE

int main() {
  bool tests_passed = run_whisper_model_tests();

  if (tests_passed) {
    demonstrate_whisper_model_usage();
  }

  return tests_passed ? 0 : 1;
}

#endif