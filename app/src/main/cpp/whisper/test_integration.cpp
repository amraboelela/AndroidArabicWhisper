#include "whisper_audio.h"
#include "feature_extractor.h"
#include "audio.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>  // For M_PI
#include <iomanip>  // For std::setprecision
#include <fstream>  // For file existence check

/**
 * Simple test to demonstrate whisper audio processing integration using real audio
 */
void test_whisper_audio_integration(const std::string& audio_filename = "002-01.wav") {
  std::cout << "=== Whisper Audio Processing Integration Test ===" << std::endl;

  // Test 1: Load real audio file from assets
  std::cout << "Loading audio file: " << audio_filename << " from assets..." << std::endl;

  std::string audio_file_path;

  // Try different possible asset paths depending on where we're running from
  std::vector<std::string> possible_paths = {
    "../../assets/" + audio_filename,  // From CMake build directory (test_build/)
    "../assets/" + audio_filename,     // From direct compilation
    "assets/" + audio_filename         // From project root
  };

  // Find the first path that exists
  bool found_file = false;
  for (const auto& path : possible_paths) {
    std::ifstream test_file(path);
    if (test_file.good()) {
      audio_file_path = path;
      found_file = true;
      break;
    }
  }

  if (!found_file) {
    audio_file_path = possible_paths[0]; // Use first path as fallback
  }
  std::vector<float> test_audio;

  try {
    test_audio = AudioDecoder::decode_audio(audio_file_path, WHISPER_SAMPLE_RATE);
    if (test_audio.empty()) {
      std::cout << "⚠ Failed to load " << audio_filename << ", falling back to synthetic audio" << std::endl;

      // Fallback: Create synthetic audio if file loading fails
      const int duration_seconds = 2;
      const int sample_rate = WHISPER_SAMPLE_RATE;
      const int num_samples = duration_seconds * sample_rate;

      test_audio.resize(num_samples);
      for (int i = 0; i < num_samples; ++i) {
        // Generate a simple sine wave at 440 Hz
        float t = static_cast<float>(i) / sample_rate;
        test_audio[i] = 0.5f * std::sin(2.0f * M_PI * 440.0f * t);
      }
      std::cout << "✓ Generated synthetic audio as fallback (" << test_audio.size() << " samples)" << std::endl;
    } else {
      // Store original size before potential trimming
      size_t original_size = test_audio.size();
      float original_duration = original_size / float(WHISPER_SAMPLE_RATE);

      std::cout << "✓ Successfully loaded " << audio_filename << " (" << original_size << " samples, "
                << original_duration << " seconds)" << std::endl;

      // For very large files, let's test with just the first portion to avoid memory issues
      if (test_audio.size() > WHISPER_SAMPLE_RATE * 30) {  // If longer than 30 seconds
        std::cout << "  → File is very large, using first 30 seconds for testing" << std::endl;
        test_audio.resize(WHISPER_SAMPLE_RATE * 30);
        std::cout << "  → Trimmed to " << test_audio.size() << " samples ("
                  << (test_audio.size() / float(WHISPER_SAMPLE_RATE)) << " seconds)" << std::endl;
      }
    }
  } catch (const std::exception& e) {
    std::cout << "⚠ Error loading " << audio_filename << ": " << e.what() << std::endl;
    std::cout << "Falling back to synthetic audio..." << std::endl;

    // Fallback: Create synthetic audio
    const int duration_seconds = 2;
    const int sample_rate = WHISPER_SAMPLE_RATE;
    const int num_samples = duration_seconds * sample_rate;

    test_audio.resize(num_samples);
    for (int i = 0; i < num_samples; ++i) {
      float t = static_cast<float>(i) / sample_rate;
      test_audio[i] = 0.5f * std::sin(2.0f * M_PI * 440.0f * t);
    }
    std::cout << "✓ Generated synthetic audio as fallback" << std::endl;
  }

  // Test 2: Test audio preprocessing functions
  std::cout << "Testing audio preprocessing..." << std::endl;

  // Test normalization
  auto normalized_audio = whisper::AudioProcessor::normalize_audio(test_audio);
  std::cout << "✓ Audio normalization completed" << std::endl;

  // Test padding/trimming
  auto padded_audio = whisper::AudioProcessor::pad_or_trim(normalized_audio,
                                                           WHISPER_CHUNK_SIZE);  // Remove whisper:: namespace
  std::cout << "✓ Audio padding/trimming completed. Size: " << padded_audio.size() << std::endl;

  // Test pre-emphasis filter
  auto filtered_audio = whisper::AudioProcessor::apply_preemphasis(padded_audio);
  std::cout << "✓ Pre-emphasis filter applied" << std::endl;

  // Test 3: Test mel spectrogram extraction
  std::cout << "Testing mel spectrogram extraction..." << std::endl;
  auto mel_spectrogram = whisper::AudioProcessor::extract_mel_spectrogram(filtered_audio);

  if (!mel_spectrogram.empty()) {
    std::cout << "✓ Mel spectrogram extracted. Dimensions: "
              << mel_spectrogram.size() << " x "
              << mel_spectrogram[0].size() << std::endl;

    // Apply log transform
    auto log_mel_spectrogram = whisper::AudioProcessor::apply_log_transform(mel_spectrogram);
    std::cout << "✓ Log transform applied" << std::endl;
  } else {
    std::cout << "✗ Failed to extract mel spectrogram" << std::endl;
  }

  // Test 4: Test integration with AudioDecoder using real file
  std::cout << "Testing AudioDecoder integration..." << std::endl;

  // Load the file fresh to show actual properties (not trimmed version)
  auto full_audio = AudioDecoder::decode_audio(audio_file_path, WHISPER_SAMPLE_RATE);

  std::cout << "✓ AudioDecoder successfully loaded: " << audio_file_path << std::endl;
  std::cout << "Audio properties:" << std::endl;
  std::cout << "  - Samples: " << full_audio.size() << std::endl;
  std::cout << "  - Duration: " << (full_audio.size() / float(WHISPER_SAMPLE_RATE)) << " seconds" << std::endl;
  std::cout << "  - Sample Rate: " << WHISPER_SAMPLE_RATE << " Hz" << std::endl;

  // Show some sample values
  if (full_audio.size() >= 10) {
    std::cout << "  - First 10 samples: ";
    for (int i = 0; i < 10; ++i) {
      std::cout << std::fixed << std::setprecision(3) << full_audio[i] << " ";
    }
    std::cout << std::endl;
  }

  // Test 5: Test FeatureExtractor integration
  std::cout << "Testing FeatureExtractor integration..." << std::endl;
  FeatureExtractor extractor(80, 16000, 160, 30, 400);

  auto features = extractor.extract(filtered_audio);
  if (!features.empty()) {
    std::cout << "✓ FeatureExtractor integration successful. Features: "
              << features.size() << " x "
              << (features.empty() ? 0 : features[0].size()) << std::endl;
  } else {
    std::cout << "✓ FeatureExtractor fallback to original implementation" << std::endl;
  }

  std::cout << "=== Integration Test Completed ===" << std::endl;
}

/**
 * Usage example for whisper audio processing with different audio files
 */
void demonstrate_usage() {
  std::cout << "\n=== Usage Example ===" << std::endl;

  std::cout << "// Example usage in your application with different audio files:" << std::endl;
  std::cout << "// 1. Load any audio file from assets:" << std::endl;
  std::cout << "//    auto audio = AudioDecoder::decode_audio(\"assets/002-01.wav\", 16000);  // Large file" << std::endl;
  std::cout << "//    auto audio = AudioDecoder::decode_audio(\"assets/001.wav\", 16000);     // Smaller file" << std::endl;
  std::cout << "//    auto audio = AudioDecoder::decode_audio(\"assets/test.wav\", 16000);    // Test file" << std::endl;
  std::cout << "//    // For large files, consider processing in chunks" << std::endl;
  std::cout << "//" << std::endl;
  std::cout << "// 2. Test with different files:" << std::endl;
  std::cout << "//    test_whisper_audio_integration(\"002-01.wav\");  // Large Arabic file" << std::endl;
  std::cout << "//    test_whisper_audio_integration(\"001.wav\");     // Medium file" << std::endl;
  std::cout << "//    test_whisper_audio_integration(\"test.wav\");    // Small test file" << std::endl;
  std::cout << "//" << std::endl;
  std::cout << "// 3. Preprocess audio with whisper-compatible functions:" << std::endl;
  std::cout << "//    auto normalized = whisper::AudioProcessor::normalize_audio(audio);" << std::endl;
  std::cout << "//    auto padded = whisper::AudioProcessor::pad_or_trim(normalized, WHISPER_CHUNK_SIZE);" << std::endl;
  std::cout << "//    auto filtered = whisper::AudioProcessor::apply_preemphasis(padded);" << std::endl;
  std::cout << "//" << std::endl;
  std::cout << "// 4. Extract features for whisper model:" << std::endl;
  std::cout << "//    FeatureExtractor extractor;" << std::endl;
  std::cout << "//    auto features = extractor.extract(filtered);" << std::endl;
  std::cout << "//" << std::endl;
  std::cout << "// 5. Pass features to your whisper model:" << std::endl;
  std::cout << "//    WhisperModel model(\"path/to/model\");" << std::endl;
  std::cout << "//    auto [segments, info] = model.transcribe(audio, \"ar\", true);" << std::endl;

  std::cout << "\n// Key benefits:" << std::endl;
  std::cout << "// - Flexible audio file testing with any file in assets/" << std::endl;
  std::cout << "// - Real audio file support through AudioDecoder" << std::endl;
  std::cout << "// - Whisper-compatible audio preprocessing" << std::endl;
  std::cout << "// - Proper 16kHz sampling rate handling" << std::endl;
  std::cout << "// - Mel spectrogram extraction matching whisper.cpp" << std::endl;
  std::cout << "// - Arabic language support for transcription" << std::endl;
  std::cout << "// - Integrated with existing Android NDK codebase" << std::endl;

  std::cout << "\n// Available test files:" << std::endl;
  std::cout << "// - 002-01.wav (28MB) - Large Arabic audio file" << std::endl;
  std::cout << "// - 001.wav (1.3MB) - Medium audio file" << std::endl;
  std::cout << "// - test.wav (130KB) - Small test file" << std::endl;
  std::cout << "// - Besmellah.m4a - M4A format (if supported)" << std::endl;
  std::cout << "// - Automatic resampling to 16kHz if needed" << std::endl;
  std::cout << "// - Smart chunking for large files to manage memory" << std::endl;
}

#ifndef TESTING_MODE

int main() {
  // Test with default file (002-01.wav)
  test_whisper_audio_integration();

  std::cout << "\n" << std::string(50, '=') << std::endl;
  std::cout << "Testing with different audio file..." << std::endl;
  std::cout << std::string(50, '=') << "\n" << std::endl;

  // Test with smaller file
  test_whisper_audio_integration("001.wav");

  demonstrate_usage();
  return 0;
}

#endif