#include "audio.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <string>
#include <algorithm>
#include <cmath>
#include <random>

/**
 * Unit tests for audio processing functionality
 * Testing audio decoding, padding/trimming, and preprocessing
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
 * Generate synthetic audio for testing
 */
std::vector<float> generate_sine_wave(int sample_rate, float duration, float frequency, float amplitude = 0.5f) {
    int num_samples = static_cast<int>(sample_rate * duration);
    std::vector<float> audio(num_samples);

    for (int i = 0; i < num_samples; i++) {
        audio[i] = amplitude * std::sin(2.0f * M_PI * frequency * i / sample_rate);
    }

    return audio;
}

/**
 * Generate white noise for testing
 */
std::vector<float> generate_white_noise(int num_samples, float amplitude = 0.1f) {
    std::vector<float> audio(num_samples);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-amplitude, amplitude);

    for (int i = 0; i < num_samples; i++) {
        audio[i] = dis(gen);
    }

    return audio;
}

/**
 * Test pad_or_trim functionality
 */
bool test_pad_or_trim() {
    std::cout << "\n=== Testing Pad or Trim Functionality ===" << std::endl;

    // Test trimming (audio longer than target)
    std::vector<float> long_audio = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto trimmed = AudioDecoder::pad_or_trim(long_audio, 5);
    ASSERT_EQ(trimmed.size(), 5, "Trimmed audio length");
    ASSERT_EQ(trimmed[0], 1.0f, "Trimmed audio first element");
    ASSERT_EQ(trimmed[4], 5.0f, "Trimmed audio last element");

    // Test padding (audio shorter than target)
    std::vector<float> short_audio = {1, 2, 3};
    auto padded = AudioDecoder::pad_or_trim(short_audio, 7);
    ASSERT_EQ(padded.size(), 7, "Padded audio length");
    ASSERT_EQ(padded[0], 1.0f, "Padded audio first element");
    ASSERT_EQ(padded[2], 3.0f, "Padded audio original last element");
    ASSERT_EQ(padded[3], 0.0f, "Padded audio zero padding");
    ASSERT_EQ(padded[6], 0.0f, "Padded audio final zero");

    // Test exact length (no padding or trimming)
    std::vector<float> exact_audio = {1, 2, 3, 4, 5};
    auto unchanged = AudioDecoder::pad_or_trim(exact_audio, 5);
    ASSERT_EQ(unchanged.size(), 5, "Unchanged audio length");
    ASSERT_EQ(unchanged[0], 1.0f, "Unchanged audio first element");
    ASSERT_EQ(unchanged[4], 5.0f, "Unchanged audio last element");

    return true;
}

/**
 * Test pad_or_trim with different target lengths
 */
bool test_pad_or_trim_various_lengths() {
    std::cout << "\n=== Testing Pad or Trim Various Lengths ===" << std::endl;

    std::vector<float> test_audio = generate_sine_wave(16000, 1.0f, 440.0f); // 1 second of 440Hz

    // Test common Whisper chunk sizes
    std::vector<size_t> target_lengths = {160000, 320000, 480000}; // 10s, 20s, 30s at 16kHz

    for (size_t target : target_lengths) {
        auto result = AudioDecoder::pad_or_trim(test_audio, target);
        ASSERT_EQ(result.size(), target, "Target length " + std::to_string(target));

        if (target > test_audio.size()) {
            // Should be padded
            ASSERT_EQ(result[0], test_audio[0], "First element preserved in padding");
            ASSERT_EQ(result[test_audio.size()], 0.0f, "Zero padding added");
        } else {
            // Should be trimmed
            ASSERT_EQ(result[0], test_audio[0], "First element preserved in trimming");
            ASSERT_EQ(result.size(), target, "Trimmed to target length");
        }
    }

    return true;
}

/**
 * Test pad_or_trim edge cases
 */
bool test_pad_or_trim_edge_cases() {
    std::cout << "\n=== Testing Pad or Trim Edge Cases ===" << std::endl;

    // Test empty input
    std::vector<float> empty_audio;
    auto padded_empty = AudioDecoder::pad_or_trim(empty_audio, 5);
    ASSERT_EQ(padded_empty.size(), 5, "Padded empty audio length");
    ASSERT_EQ(padded_empty[0], 0.0f, "Padded empty audio all zeros");
    ASSERT_EQ(padded_empty[4], 0.0f, "Padded empty audio all zeros end");

    // Test zero target length
    std::vector<float> some_audio = {1, 2, 3, 4, 5};
    auto zero_length = AudioDecoder::pad_or_trim(some_audio, 0);
    ASSERT_EQ(zero_length.size(), 0, "Zero target length");

    // Test single element
    std::vector<float> single_element = {42.0f};
    auto single_padded = AudioDecoder::pad_or_trim(single_element, 3);
    ASSERT_EQ(single_padded.size(), 3, "Single element padded length");
    ASSERT_EQ(single_padded[0], 42.0f, "Single element preserved");
    ASSERT_EQ(single_padded[1], 0.0f, "Single element padding");

    auto single_trimmed = AudioDecoder::pad_or_trim(single_element, 1);
    ASSERT_EQ(single_trimmed.size(), 1, "Single element trimmed length");
    ASSERT_EQ(single_trimmed[0], 42.0f, "Single element preserved in trim");

    return true;
}

/**
 * Test audio processing with realistic parameters
 */
bool test_realistic_audio_decoder() {
    std::cout << "\n=== Testing Realistic Audio Processing ===" << std::endl;

    // Generate 5-second audio clip at 16kHz
    int sample_rate = 16000;
    float duration = 5.0f;
    auto audio = generate_sine_wave(sample_rate, duration, 1000.0f); // 1kHz tone

    ASSERT_EQ(audio.size(), sample_rate * 5, "Generated audio length");

    // Test typical Whisper preprocessing (30-second chunks)
    size_t whisper_chunk_size = 30 * sample_rate; // 30 seconds
    auto whisper_chunk = AudioDecoder::pad_or_trim(audio, whisper_chunk_size);

    ASSERT_EQ(whisper_chunk.size(), whisper_chunk_size, "Whisper chunk size");

    // Original 5 seconds should be preserved
    ASSERT_EQ(whisper_chunk[0], audio[0], "Original audio preserved at start");
    ASSERT_APPROX_EQ(whisper_chunk[audio.size() - 1], audio[audio.size() - 1], 0.0001f, "Original audio preserved at end");

    // Padding should be zeros
    ASSERT_EQ(whisper_chunk[audio.size()], 0.0f, "Padding starts with zero");
    ASSERT_EQ(whisper_chunk[whisper_chunk_size - 1], 0.0f, "Padding ends with zero");

    return true;
}

/**
 * Test audio processing with different sample rates
 */
bool test_different_sample_rates() {
    std::cout << "\n=== Testing Different Sample Rates ===" << std::endl;

    std::vector<int> sample_rates = {8000, 16000, 22050, 44100, 48000};

    for (int sr : sample_rates) {
        // Generate 1-second audio
        auto audio = generate_sine_wave(sr, 1.0f, 440.0f);
        ASSERT_EQ(audio.size(), sr, "Audio length matches sample rate");

        // Test padding to 2 seconds
        auto padded = AudioDecoder::pad_or_trim(audio, sr * 2);
        ASSERT_EQ(padded.size(), sr * 2, "Padded to 2 seconds");

        // Test trimming to 0.5 seconds
        auto trimmed = AudioDecoder::pad_or_trim(audio, sr / 2);
        ASSERT_EQ(trimmed.size(), sr / 2, "Trimmed to 0.5 seconds");
    }

    return true;
}

/**
 * Test audio processing preserves signal characteristics
 */
bool test_signal_preservation() {
    std::cout << "\n=== Testing Signal Preservation ===" << std::endl;

    // Generate test signals
    auto sine_wave = generate_sine_wave(16000, 2.0f, 500.0f, 0.8f);
    auto noise = generate_white_noise(16000, 0.2f);

    // Test that padding preserves original signal
    auto padded_sine = AudioDecoder::pad_or_trim(sine_wave, 48000); // Pad to 3 seconds

    // Check that original signal is preserved
    bool signal_preserved = true;
    for (size_t i = 0; i < sine_wave.size(); i++) {
        if (std::abs(padded_sine[i] - sine_wave[i]) > 1e-6f) {
            signal_preserved = false;
            break;
        }
    }
    ASSERT_TRUE(signal_preserved, "Sine wave signal preserved in padding");

    // Test that trimming preserves beginning of signal
    auto trimmed_sine = AudioDecoder::pad_or_trim(sine_wave, 8000); // Trim to 0.5 seconds

    bool beginning_preserved = true;
    for (size_t i = 0; i < trimmed_sine.size(); i++) {
        if (std::abs(trimmed_sine[i] - sine_wave[i]) > 1e-6f) {
            beginning_preserved = false;
            break;
        }
    }
    ASSERT_TRUE(beginning_preserved, "Sine wave beginning preserved in trimming");

    return true;
}

/**
 * Test memory efficiency and performance
 */
bool test_memory_efficiency() {
    std::cout << "\n=== Testing Memory Efficiency ===" << std::endl;

    // Test with large audio arrays
    size_t large_size = 1000000; // 1M samples ≈ 62.5 seconds at 16kHz
    std::vector<float> large_audio(large_size, 0.5f);

    // Test trimming large array
    auto trimmed_large = AudioDecoder::pad_or_trim(large_audio, 16000); // Trim to 1 second
    ASSERT_EQ(trimmed_large.size(), 16000, "Large array trimmed correctly");
    ASSERT_EQ(trimmed_large[0], 0.5f, "Large array trimming preserves values");

    // Test padding to larger size
    std::vector<float> small_audio(1000, 0.3f);
    auto padded_large = AudioDecoder::pad_or_trim(small_audio, large_size);
    ASSERT_EQ(padded_large.size(), large_size, "Small array padded to large size");
    ASSERT_EQ(padded_large[0], 0.3f, "Original values preserved");
    ASSERT_EQ(padded_large[999], 0.3f, "Last original value preserved");
    ASSERT_EQ(padded_large[1000], 0.0f, "Padding is zero");

    return true;
}

/**
 * Test stereo audio handling concepts
 */
bool test_stereo_concepts() {
    std::cout << "\n=== Testing Stereo Audio Concepts ===" << std::endl;

    // Test the concept of stereo processing
    // (Note: decode_audio_split_stereo is not fully implemented, so we test the concept)

    // Simulate stereo data (interleaved L-R samples)
    std::vector<float> stereo_interleaved;
    for (int i = 0; i < 1000; i++) {
        stereo_interleaved.push_back(0.5f);  // Left channel
        stereo_interleaved.push_back(-0.5f); // Right channel
    }

    ASSERT_EQ(stereo_interleaved.size(), 2000, "Stereo interleaved data size");

    // Simulate channel separation
    std::vector<float> left_channel, right_channel;
    for (size_t i = 0; i < stereo_interleaved.size(); i += 2) {
        left_channel.push_back(stereo_interleaved[i]);
        right_channel.push_back(stereo_interleaved[i + 1]);
    }

    ASSERT_EQ(left_channel.size(), 1000, "Left channel size");
    ASSERT_EQ(right_channel.size(), 1000, "Right channel size");
    ASSERT_EQ(left_channel[0], 0.5f, "Left channel value");
    ASSERT_EQ(right_channel[0], -0.5f, "Right channel value");

    // Test mono conversion (average of channels)
    std::vector<float> mono_mixed;
    for (size_t i = 0; i < left_channel.size(); i++) {
        mono_mixed.push_back((left_channel[i] + right_channel[i]) / 2.0f);
    }

    ASSERT_EQ(mono_mixed.size(), 1000, "Mono mixed size");
    ASSERT_APPROX_EQ(mono_mixed[0], 0.0f, 0.0001f, "Mono mixed value");

    return true;
}

/**
 * Test audio quality metrics
 */
bool test_audio_quality_metrics() {
    std::cout << "\n=== Testing Audio Quality Metrics ===" << std::endl;

    // Generate reference audio
    auto reference = generate_sine_wave(16000, 1.0f, 440.0f, 0.5f);

    // Test RMS calculation
    float rms = 0.0f;
    for (float sample : reference) {
        rms += sample * sample;
    }
    rms = std::sqrt(rms / reference.size());

    // For a sine wave with amplitude 0.5, RMS should be approximately 0.5/sqrt(2) ≈ 0.354
    ASSERT_APPROX_EQ(rms, 0.354f, 0.01f, "RMS calculation for sine wave");

    // Test peak detection
    float peak = *std::max_element(reference.begin(), reference.end(),
        [](float a, float b) { return std::abs(a) < std::abs(b); });
    ASSERT_APPROX_EQ(std::abs(peak), 0.5f, 0.01f, "Peak amplitude detection");

    // Test zero crossing rate for sine wave
    int zero_crossings = 0;
    for (size_t i = 1; i < reference.size(); i++) {
        if ((reference[i-1] >= 0 && reference[i] < 0) ||
            (reference[i-1] < 0 && reference[i] >= 0)) {
            zero_crossings++;
        }
    }

    // 440Hz sine wave should have ~880 zero crossings per second
    ASSERT_TRUE(zero_crossings > 800 && zero_crossings < 950, "Zero crossing rate reasonable");

    return true;
}

} // anonymous namespace

/**
 * Main test runner for audio processing tests
 */
bool run_audio_decoder_tests() {
    std::cout << "=== AUDIO PROCESSING UNIT TESTS ===" << std::endl;

    bool all_passed = true;

    all_passed &= test_pad_or_trim();
    all_passed &= test_pad_or_trim_various_lengths();
    all_passed &= test_pad_or_trim_edge_cases();
    all_passed &= test_realistic_audio_decoder();
    all_passed &= test_different_sample_rates();
    all_passed &= test_signal_preservation();
    all_passed &= test_memory_efficiency();
    all_passed &= test_stereo_concepts();
    all_passed &= test_audio_quality_metrics();

    std::cout << "\n=== AUDIO PROCESSING TEST SUMMARY ===" << std::endl;
    if (all_passed) {
        std::cout << "✅ ALL AUDIO PROCESSING TESTS PASSED!" << std::endl;
    } else {
        std::cout << "❌ SOME AUDIO PROCESSING TESTS FAILED!" << std::endl;
    }

    return all_passed;
}

/**
 * Demonstrate audio processing usage
 */
void demonstrate_audio_decoder_usage() {
    std::cout << "\n=== Audio Processing Usage Examples ===" << std::endl;

    std::cout << "// Basic audio processing:" << std::endl;
    std::cout << "// 1. Load and decode audio file:" << std::endl;
    std::cout << "//    auto audio = AudioDecoder::decode_audio(\"speech.wav\", 16000);" << std::endl;
    std::cout << "//" << std::endl;
    std::cout << "// 2. Preprocess for Whisper (30-second chunks):" << std::endl;
    std::cout << "//    auto chunk = AudioDecoder::pad_or_trim(audio, 30 * 16000);" << std::endl;
    std::cout << "//" << std::endl;
    std::cout << "// 3. Handle stereo audio:" << std::endl;
    std::cout << "//    auto [left, right] = AudioDecoder::decode_audio_split_stereo(\"stereo.wav\");" << std::endl;
    std::cout << "//    auto mono = AudioDecoder::pad_or_trim(left, 30 * 16000); // Use left channel" << std::endl;

    std::cout << "\n// Common preprocessing patterns:" << std::endl;
    std::cout << "// - Whisper input: pad_or_trim(audio, 480000)  // 30s at 16kHz" << std::endl;
    std::cout << "// - Real-time chunks: pad_or_trim(audio, 160000)  // 10s at 16kHz" << std::endl;
    std::cout << "// - Short utterances: pad_or_trim(audio, 80000)  // 5s at 16kHz" << std::endl;

    std::cout << "\n// Quality considerations:" << std::endl;
    std::cout << "// - Always use 16kHz for Whisper compatibility" << std::endl;
    std::cout << "// - Padding preserves original signal quality" << std::endl;
    std::cout << "// - Trimming removes end of audio, not beginning" << std::endl;
    std::cout << "// - Memory-efficient for large audio files" << std::endl;
    std::cout << "// - Supports various input formats through AudioDecoder" << std::endl;
}

#ifndef TESTING_MODE
int main() {
    bool tests_passed = run_audio_decoder_tests();

    if (tests_passed) {
        demonstrate_audio_decoder_usage();
    }

    return tests_passed ? 0 : 1;
}
#endif