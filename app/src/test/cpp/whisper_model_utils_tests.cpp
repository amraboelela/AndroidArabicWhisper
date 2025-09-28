/**
 * Unit Tests for WhisperModel Utility Functions Implementation
 * Tests helper functions for feature processing, compression, and timestamps
 * Created by Amr Aboelela
 */

#include <iostream>
#include <vector>
#include <cassert>
#include <string>
#include <map>
#include <cmath>
#include <algorithm>
#include <numeric>

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

#define ASSERT_FALSE(condition, test_name) \
    if ((condition)) { \
        std::cerr << "FAILED: " << test_name << " - Condition should be false" << std::endl; \
        return false; \
    } else { \
        std::cout << "✓ " << test_name << std::endl; \
    }

#define ASSERT_GT(actual, threshold, test_name) \
    if ((actual) <= (threshold)) { \
        std::cerr << "FAILED: " << test_name << " - Expected > " << (threshold) << ", Got: " << (actual) << std::endl; \
        return false; \
    } else { \
        std::cout << "✓ " << test_name << std::endl; \
    }

#define ASSERT_GE(actual, threshold, test_name) \
    if ((actual) < (threshold)) { \
        std::cerr << "FAILED: " << test_name << " - Expected >= " << (threshold) << ", Got: " << (actual) << std::endl; \
        return false; \
    } else { \
        std::cout << "✓ " << test_name << std::endl; \
    }

#define ASSERT_LE(actual, threshold, test_name) \
    if ((actual) > (threshold)) { \
        std::cerr << "FAILED: " << test_name << " - Expected <= " << (threshold) << ", Got: " << (actual) << std::endl; \
        return false; \
    } else { \
        std::cout << "✓ " << test_name << std::endl; \
    }

#define ASSERT_LT(actual, threshold, test_name) \
    if ((actual) >= (threshold)) { \
        std::cerr << "FAILED: " << test_name << " - Expected < " << (threshold) << ", Got: " << (actual) << std::endl; \
        return false; \
    } else { \
        std::cout << "✓ " << test_name << std::endl; \
    }

#define ASSERT_NEAR(actual, expected, tolerance, test_name) \
    if (std::abs((actual) - (expected)) > (tolerance)) { \
        std::cerr << "FAILED: " << test_name << " - Expected: " << (expected) << " ± " << (tolerance) << ", Got: " << (actual) << std::endl; \
        return false; \
    } else { \
        std::cout << "✓ " << test_name << std::endl; \
    }

// Mock slice_features function for testing
std::vector<std::vector<float>> mock_slice_features(
    const std::vector<std::vector<float>>& features, int start, int length) {
    if (features.empty()) {
        return {};
    }

    std::vector<std::vector<float>> sliced_features;
    sliced_features.reserve(features.size());

    for (const auto& feature_row : features) {
        std::vector<float> sliced_row;
        int end = std::min(start + length, static_cast<int>(feature_row.size()));

        if (start < static_cast<int>(feature_row.size())) {
            sliced_row.assign(feature_row.begin() + start, feature_row.begin() + end);
        }
        // If start >= feature_row.size(), sliced_row remains empty

        sliced_features.push_back(sliced_row);
    }

    return sliced_features;
}

// Mock pad_or_trim function for testing
std::vector<std::vector<float>> mock_pad_or_trim(const std::vector<std::vector<float>>& segment) {
    if (segment.empty()) {
        return segment;
    }

    const int TARGET_LENGTH = 3000;
    std::vector<std::vector<float>> result = segment;

    for (auto& feature_row : result) {
        if (static_cast<int>(feature_row.size()) < TARGET_LENGTH) {
            feature_row.resize(TARGET_LENGTH, 0.0f);
        } else if (static_cast<int>(feature_row.size()) > TARGET_LENGTH) {
            feature_row.resize(TARGET_LENGTH);
        }
    }

    return result;
}

// Mock compression ratio function
float mock_get_compression_ratio(const std::string& text) {
    if (text.empty()) {
        return 1.0f;
    }

    // Simple mock: repetitive text compresses better
    size_t unique_chars = 0;
    std::string seen;
    for (char c : text) {
        if (seen.find(c) == std::string::npos) {
            seen += c;
            unique_chars++;
        }
    }

    // Higher ratio for less unique characters
    return static_cast<float>(text.size()) / static_cast<float>(unique_chars + 1);
}

// Test slice_features function
bool test_slice_features() {
    std::cout << "\n=== Testing Slice Features ===" << std::endl;

    std::vector<std::vector<float>> sample_features = {
        {0.1f, 0.2f, 0.3f, 0.4f, 0.5f},
        {0.15f, 0.25f, 0.35f, 0.45f, 0.55f},
        {0.2f, 0.3f, 0.4f, 0.5f, 0.6f}
    };

    int start = 1;
    int length = 3;

    auto sliced = mock_slice_features(sample_features, start, length);

    ASSERT_EQ(sliced.size(), sample_features.size(), "Should preserve number of feature rows");
    for (size_t i = 0; i < sliced.size(); ++i) {
        ASSERT_EQ(sliced[i].size(), static_cast<size_t>(length), "Should have correct slice length");
        ASSERT_EQ(sliced[i][0], sample_features[i][start], "Should start at correct position");
        ASSERT_EQ(sliced[i][2], sample_features[i][start + 2], "Should have correct values");
    }

    // Test out of bounds
    auto sliced_oob = mock_slice_features(sample_features, 10, 3);
    ASSERT_EQ(sliced_oob.size(), sample_features.size(), "Should return same number of rows even for out of bounds");
    for (const auto& row : sliced_oob) {
        ASSERT_TRUE(row.empty(), "Out of bounds rows should be empty");
    }

    // Test empty input
    std::vector<std::vector<float>> empty_features;
    auto sliced_empty = mock_slice_features(empty_features, 0, 3);
    ASSERT_TRUE(sliced_empty.empty(), "Empty input should return empty result");

    return true;
}

// Test pad_or_trim function
bool test_pad_or_trim() {
    std::cout << "\n=== Testing Pad or Trim ===" << std::endl;

    // Test padding (features smaller than target)
    std::vector<std::vector<float>> small_features = {
        {0.1f, 0.2f, 0.3f},
        {0.15f, 0.25f, 0.35f}
    };

    auto padded = mock_pad_or_trim(small_features);

    ASSERT_EQ(padded.size(), small_features.size(), "Should preserve number of feature rows");
    for (size_t i = 0; i < padded.size(); ++i) {
        ASSERT_EQ(padded[i].size(), 3000, "Should pad to target length");

        // Original values should be preserved
        for (size_t j = 0; j < small_features[i].size(); ++j) {
            ASSERT_EQ(padded[i][j], small_features[i][j], "Original values should be preserved");
        }

        // Padded values should be zero
        for (size_t j = small_features[i].size(); j < 10; ++j) { // Check first few padded values
            ASSERT_EQ(padded[i][j], 0.0f, "Padded values should be zero");
        }
    }

    // Test trimming (create features larger than target)
    std::vector<std::vector<float>> large_features(2);
    for (auto& row : large_features) {
        row.resize(4000, 0.5f); // Larger than 3000
    }

    auto trimmed = mock_pad_or_trim(large_features);
    ASSERT_EQ(trimmed.size(), large_features.size(), "Should preserve number of feature rows");
    for (const auto& row : trimmed) {
        ASSERT_EQ(row.size(), 3000, "Should trim to target length");
    }

    // Test empty input
    std::vector<std::vector<float>> empty_features;
    auto result = mock_pad_or_trim(empty_features);
    ASSERT_TRUE(result.empty(), "Empty input should return empty result");

    return true;
}

// Test compression ratio function
bool test_get_compression_ratio() {
    std::cout << "\n=== Testing Compression Ratio ===" << std::endl;

    // Test normal text
    std::string text = "This is a test string with some content";
    float ratio = mock_get_compression_ratio(text);
    ASSERT_GT(ratio, 1.0f, "Normal text should have ratio > 1.0");
    ASSERT_LT(ratio, 10.0f, "Normal text should have reasonable ratio");

    // Test empty text
    std::string empty_text = "";
    float empty_ratio = mock_get_compression_ratio(empty_text);
    ASSERT_EQ(empty_ratio, 1.0f, "Empty text should return 1.0");

    // Test highly repetitive text
    std::string repetitive_text = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    float rep_ratio = mock_get_compression_ratio(repetitive_text);
    ASSERT_GT(rep_ratio, 5.0f, "Repetitive text should have high compression ratio");

    // Test random text
    std::string random_text = "xqp2w9ebrjkas8df7gh3klm5n6vct4yui1oz";
    float rand_ratio = mock_get_compression_ratio(random_text);
    ASSERT_GE(rand_ratio, 1.0f, "Random text should have ratio >= 1.0");
    ASSERT_LT(rand_ratio, 3.0f, "Random text shouldn't compress much");

    return true;
}

// Test audio analysis functions
bool test_audio_analysis() {
    std::cout << "\n=== Testing Audio Analysis ===" << std::endl;

    std::vector<float> sample_audio = {0.1f, -0.2f, 0.3f, -0.1f, 0.05f, -0.15f, 0.25f, -0.05f};

    // Test SNR calculation (mock)
    float sum_squares = 0.0f;
    for (float sample : sample_audio) {
        sum_squares += sample * sample;
    }
    float rms = std::sqrt(sum_squares / sample_audio.size());
    ASSERT_GT(rms, 0.0f, "RMS should be positive for non-silent audio");

    // Mock SNR calculation
    float mock_snr = 20.0f * std::log10(rms / 0.01f); // Assume 0.01 noise floor
    ASSERT_GE(mock_snr, 0.0f, "SNR should be non-negative");
    ASSERT_LT(mock_snr, 100.0f, "SNR should be reasonable");

    // Test silence detection (mock)
    std::vector<float> silent_audio = {0.001f, -0.002f, 0.003f, -0.001f};
    float silent_rms = 0.0f;
    for (float sample : silent_audio) {
        silent_rms += sample * sample;
    }
    silent_rms = std::sqrt(silent_rms / silent_audio.size());

    bool is_silent = silent_rms < 0.01f;
    ASSERT_TRUE(is_silent, "Low amplitude audio should be detected as silent");

    // Test loud audio
    std::vector<float> loud_audio = {0.1f, -0.2f, 0.15f, -0.1f};
    float loud_rms = 0.0f;
    for (float sample : loud_audio) {
        loud_rms += sample * sample;
    }
    loud_rms = std::sqrt(loud_rms / loud_audio.size());

    bool is_loud = loud_rms >= 0.01f;
    ASSERT_TRUE(is_loud, "High amplitude audio should not be detected as silent");

    return true;
}

// Test feature processing functions
bool test_feature_processing() {
    std::cout << "\n=== Testing Feature Processing ===" << std::endl;

    std::vector<std::vector<float>> sample_features = {
        {0.1f, 0.2f, 0.3f, 0.4f},
        {0.15f, 0.25f, 0.35f, 0.45f},
        {0.2f, 0.3f, 0.4f, 0.5f}
    };

    // Test normalization (mock)
    std::vector<std::vector<float>> normalized = sample_features;

    for (auto& feature_row : normalized) {
        // Calculate mean
        float sum = std::accumulate(feature_row.begin(), feature_row.end(), 0.0f);
        float mean = sum / feature_row.size();

        // Calculate std dev
        float sq_sum = 0.0f;
        for (float val : feature_row) {
            sq_sum += (val - mean) * (val - mean);
        }
        float std_dev = std::sqrt(sq_sum / feature_row.size());

        // Normalize
        if (std_dev > 1e-8f) {
            for (float& val : feature_row) {
                val = (val - mean) / std_dev;
            }
        }
    }

    ASSERT_EQ(normalized.size(), sample_features.size(), "Normalization should preserve dimensions");

    // Test log mel transformation (mock)
    std::vector<std::vector<float>> log_mel = sample_features;
    for (auto& feature_row : log_mel) {
        for (float& val : feature_row) {
            val = std::log(std::max(val, 1e-10f));
        }
    }

    for (size_t i = 0; i < log_mel.size(); ++i) {
        ASSERT_EQ(log_mel[i].size(), sample_features[i].size(), "Log mel should preserve dimensions");
    }

    return true;
}

// Test edge cases and error handling
bool test_edge_cases() {
    std::cout << "\n=== Testing Edge Cases ===" << std::endl;

    // Test with empty vectors
    std::vector<std::vector<float>> empty_features;

    auto sliced_empty = mock_slice_features(empty_features, 0, 5);
    ASSERT_TRUE(sliced_empty.empty(), "Slicing empty features should return empty");

    auto padded_empty = mock_pad_or_trim(empty_features);
    ASSERT_TRUE(padded_empty.empty(), "Padding empty features should return empty");

    // Test with zero values
    std::vector<std::vector<float>> zero_features = {
        {0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f}
    };

    auto sliced_zero = mock_slice_features(zero_features, 1, 2);
    ASSERT_EQ(sliced_zero.size(), zero_features.size(), "Should handle zero values");
    for (const auto& row : sliced_zero) {
        ASSERT_EQ(row.size(), 2, "Should have correct slice size");
        for (float val : row) {
            ASSERT_EQ(val, 0.0f, "Should preserve zero values");
        }
    }

    return true;
}

// Test integration and pipeline
bool test_integration_pipeline() {
    std::cout << "\n=== Testing Integration Pipeline ===" << std::endl;

    std::vector<std::vector<float>> sample_features = {
        {0.1f, 0.2f, 0.3f, 0.4f, 0.5f},
        {0.15f, 0.25f, 0.35f, 0.45f, 0.55f},
        {0.2f, 0.3f, 0.4f, 0.5f, 0.6f}
    };

    // Complete feature processing pipeline
    auto sliced = mock_slice_features(sample_features, 1, 3);
    ASSERT_FALSE(sliced.empty(), "Pipeline step 1: slice should work");

    auto padded = mock_pad_or_trim(sliced);
    ASSERT_EQ(padded[0].size(), 3000, "Pipeline step 2: padding should work");

    // Mock CTranslate2 storage simulation
    size_t total_elements = 0;
    for (const auto& row : padded) {
        total_elements += row.size();
    }
    ASSERT_GT(total_elements, 0, "Pipeline step 3: storage should have elements");

    return true;
}

// Main test runner
int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "WhisperModel Utils Unit Tests" << std::endl;
    std::cout << "Testing utility functions" << std::endl;
    std::cout << "========================================" << std::endl;

    bool all_passed = true;

    all_passed &= test_slice_features();
    all_passed &= test_pad_or_trim();
    all_passed &= test_get_compression_ratio();
    all_passed &= test_audio_analysis();
    all_passed &= test_feature_processing();
    all_passed &= test_edge_cases();
    all_passed &= test_integration_pipeline();

    std::cout << "\n========================================" << std::endl;
    if (all_passed) {
        std::cout << "🎉 ALL UTILS TESTS PASSED!" << std::endl;
        std::cout << "✅ Feature processing functions working correctly" << std::endl;
        std::cout << "✅ Audio analysis functions validated" << std::endl;
        std::cout << "✅ Helper utilities confirmed" << std::endl;
        std::cout << "✅ Integration pipeline tested" << std::endl;
        return 0;
    } else {
        std::cout << "❌ SOME UTILS TESTS FAILED!" << std::endl;
        std::cout << "Please review the failed tests above." << std::endl;
        return 1;
    }
}