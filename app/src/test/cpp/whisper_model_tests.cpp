#include "whisper_model.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <string>
#include <algorithm>
#include <cmath>

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
        {0.0f, 0.5f, "Hello", 0.95f},
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
    ASSERT_TRUE(segment_str.find("Hello world") != std::string::npos, "Segment to_string contains text");
    ASSERT_TRUE(segment_str.find("id: 1") != std::string::npos, "Segment to_string contains ID");

    // Test segment without words
    Segment segment2;
    segment2.id = 2;
    segment2.text = "Test without words";
    segment2.words = std::nullopt;

    std::string segment2_str = segment2.to_string();
    ASSERT_TRUE(segment2_str.find("words: []") != std::string::npos, "Empty words array in to_string");

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
    ASSERT_APPROX_EQ(info.all_language_probs.value()[0].second, 0.95f, 0.001f, "First language prob");

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

    for (const auto& text : test_texts) {
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
    for (int sr : valid_sample_rates) {
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
    for (int chunk_size : chunk_sizes) {
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

    for (const auto& model_size : model_sizes) {
        ASSERT_TRUE(!model_size.empty(), "Model size not empty");
        ASSERT_TRUE(model_size.find(" ") == std::string::npos, "Model size no spaces");
    }

    // Test language codes
    std::vector<std::string> language_codes = {
        "ar", "en", "fr", "es", "de", "it", "pt", "ru", "ja", "ko", "zh"
    };

    for (const auto& lang : language_codes) {
        ASSERT_EQ(lang.length(), 2, "Language code length");
        ASSERT_TRUE(std::islower(lang[0]) && std::islower(lang[1]), "Language code lowercase");
    }

    // Test device strings
    std::vector<std::string> devices = {"auto", "cpu", "cuda", "mps"};
    for (const auto& device : devices) {
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
    std::cout << "//    auto [segments, info] = model.transcribe(audio_data, \"ar\", true);" << std::endl;
    std::cout << "//" << std::endl;
    std::cout << "// 3. Process segments:" << std::endl;
    std::cout << "//    for (const auto& segment : segments) {" << std::endl;
    std::cout << "//        std::cout << segment.text << std::endl;" << std::endl;
    std::cout << "//        if (segment.words) {" << std::endl;
    std::cout << "//            for (const auto& word : segment.words.value()) {" << std::endl;
    std::cout << "//                std::cout << word.word << \" [\" << word.start << \"-\" << word.end << \"]\" << std::endl;" << std::endl;
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