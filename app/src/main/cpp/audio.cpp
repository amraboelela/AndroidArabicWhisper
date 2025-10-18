#include "audio.h"
#include "whisper_audio.h"

#include <iostream>
#include <stdexcept>
#include <fstream>
#include <cmath>
#include <numeric>
#include <algorithm>

// --- Audio Class Implementation ---

std::vector<float> Audio::decode_audio(
  const std::string& input_file,
  int sampling_rate
) {
  // Use whisper-compatible audio processing
  auto audio = whisper::AudioProcessor::load_audio(input_file);

  if (audio.empty()) {
  std::cerr << "Failed to load audio from: " << input_file << std::endl;
  return {};
  }

  // Ensure the audio is at the requested sampling rate
  if (sampling_rate != WHISPER_SAMPLE_RATE) {
  audio = whisper::AudioProcessor::resample_audio(audio, WHISPER_SAMPLE_RATE);
  }

  // std::cout << "Successfully loaded audio with " << audio.size() << " samples" << std::endl;
  return audio;
}

std::pair<std::vector<float>, std::vector<float>> Audio::decode_audio_split_stereo(
  const std::string& input_file,
  int sampling_rate
) {
  // Load as mono first, then duplicate for stereo output if needed
  auto mono_audio = decode_audio(input_file, sampling_rate);

  if (mono_audio.empty()) {
  return std::make_pair(std::vector<float>{}, std::vector<float>{});
  }

  // For simplicity, return the same mono audio for both channels
  // In a real implementation, you'd load actual stereo audio
  return std::make_pair(mono_audio, mono_audio);
}

std::vector<float> Audio::pad_or_trim(
  const std::vector<float>& array,
  size_t length
) {
  if (array.size() == length) {
    return array;
  } else if (array.size() > length) {
    // Trim
    return std::vector<float>(array.begin(), array.begin() + length);
  } else {
    // Pad with zeros
    std::vector<float> padded = array;
    padded.resize(length, 0.0f);
    return padded;
  }
}

void Audio::_ignore_invalid_frames() {
  // Implementation for frame validation if needed
}

void Audio::_group_frames() {
  // Implementation for frame grouping if needed
}

void Audio::_resample_frames() {
  // Implementation for frame resampling if needed
}
