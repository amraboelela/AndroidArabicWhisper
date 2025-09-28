/**
 * WhisperModel Utility Functions Implementation
 * Contains helper functions for feature processing, compression, and timestamps
 * Created by Amr Aboelela
 */

#include "whisper_model.h"
#include "utils.h"
#include <ctranslate2/storage_view.h>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <numeric>
#include <cassert>
#include <zlib.h>
#include <cstring>
#include <utility>

// Helper function implementations

std::vector<std::vector<float>>
slice_features(const std::vector<std::vector<float>> &features, int start, int length) {
  if (features.empty() || start >= static_cast<int>(features[0].size())) {
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

    sliced_features.push_back(sliced_row);
  }

  return sliced_features;
}

std::vector<std::vector<float>>
pad_or_trim(const std::vector<std::vector<float>> &segment) {
  if (segment.empty()) {
    return segment;
  }

  const int TARGET_LENGTH = 3000; // 30 seconds * 100 frames/second
  std::vector<std::vector<float>> result = segment;

  // Pad or trim the time dimension (second dimension)
  for (auto& feature_row : result) {
    if (static_cast<int>(feature_row.size()) < TARGET_LENGTH) {
      // Pad with zeros
      feature_row.resize(TARGET_LENGTH, 0.0f);
    } else if (static_cast<int>(feature_row.size()) > TARGET_LENGTH) {
      // Trim to target length
      feature_row.resize(TARGET_LENGTH);
    }
  }

  return result;
}

ctranslate2::StorageView get_ctranslate2_storage(const std::vector<std::vector<float>> &segment) {
  // Flatten 2D vector into contiguous memory and wrap in StorageView
  std::vector<float> contiguous;
  for (const auto &row: segment)
  contiguous.insert(contiguous.end(), row.begin(), row.end());

  // Create shape for 2D tensor: [num_rows, num_cols]
  ctranslate2::Shape shape = {static_cast<long>(segment.size()), static_cast<long>(segment[0].size())};
  return ctranslate2::StorageView(shape, contiguous);
}

float get_compression_ratio(const std::string &text) {
  if (text.empty()) {
    return 1.0f;
  }

  std::vector<unsigned char> compressed(text.size() * 2);
  uLongf compressed_size = compressed.size();
  int res = compress(compressed.data(), &compressed_size,
         reinterpret_cast<const unsigned char *>(text.data()), text.size());

  if (res != Z_OK) {
    return 1.0f;
  }

  return static_cast<float>(text.size()) / static_cast<float>(compressed_size);
}

void merge_punctuations(std::vector<std::pair<std::string, float>>& alignment,
           const std::vector<std::string>& prepend_punctuations,
           const std::vector<std::string>& append_punctuations) {
  if (alignment.empty()) {
    return;
  }

  // Merge prepend punctuations with following words
  for (size_t i = 0; i < alignment.size() - 1; ++i) {
    const auto& current_word = alignment[i].first;

    // Check if current word is a prepend punctuation
    auto it = std::find(prepend_punctuations.begin(), prepend_punctuations.end(), current_word);
    if (it != prepend_punctuations.end()) {
      // Merge with next word
      alignment[i + 1].first = current_word + alignment[i + 1].first;
      // Remove current punctuation entry
      alignment.erase(alignment.begin() + i);
      --i; // Adjust index after removal
    }
  }

  // Merge append punctuations with preceding words
  for (size_t i = 1; i < alignment.size(); ++i) {
    const auto& current_word = alignment[i].first;

    // Check if current word is an append punctuation
    auto it = std::find(append_punctuations.begin(), append_punctuations.end(), current_word);
    if (it != append_punctuations.end()) {
      // Merge with previous word
      alignment[i - 1].first += current_word;
      // Remove current punctuation entry
      alignment.erase(alignment.begin() + i);
      --i; // Adjust index after removal
    }
  }
}

// SpeechTimestampsMap helper class
class SpeechTimestampsMap {
private:
  std::vector<std::map<std::string, float>> speech_chunks_;
  int sampling_rate_;
  std::vector<float> chunk_starts_;
  std::vector<float> chunk_ends_;

public:
  SpeechTimestampsMap(const std::vector<std::map<std::string, float>> &speech_chunks,
          int sampling_rate)
    : speech_chunks_(speech_chunks), sampling_rate_(sampling_rate) {

    // Build chunk timing lookup tables
    for (const auto& chunk : speech_chunks) {
      if (chunk.count("start") && chunk.count("end")) {
        chunk_starts_.push_back(chunk.at("start"));
        chunk_ends_.push_back(chunk.at("end"));
      }
    }
  }

  int get_chunk_index(float t) const {
    // Find the chunk that contains time t
    for (size_t i = 0; i < chunk_starts_.size(); ++i) {
      if (t >= chunk_starts_[i] && t <= chunk_ends_[i]) {
        return static_cast<int>(i);
      }
    }

    // If not found in any chunk, return the closest one
    float min_distance = std::numeric_limits<float>::max();
    int closest_chunk = 0;

    for (size_t i = 0; i < chunk_starts_.size(); ++i) {
      float distance = std::min(std::abs(t - chunk_starts_[i]), std::abs(t - chunk_ends_[i]));
      if (distance < min_distance) {
        min_distance = distance;
        closest_chunk = static_cast<int>(i);
      }
    }

    return closest_chunk;
  }

  float get_original_time(float t, int chunk_index = -1, bool is_end = false) const {
    if (chunk_index < 0) {
      chunk_index = get_chunk_index(t);
    }

    if (chunk_index >= 0 && chunk_index < static_cast<int>(speech_chunks_.size())) {
      const auto& chunk = speech_chunks_[chunk_index];

      // Map relative time within chunk to original audio time
      if (chunk.count("original_start") && chunk.count("original_end")) {
        float chunk_start = chunk_starts_[chunk_index];
        float chunk_end = chunk_ends_[chunk_index];
        float chunk_duration = chunk_end - chunk_start;

        if (chunk_duration > 0) {
          float relative_position = (t - chunk_start) / chunk_duration;
          float original_start = chunk.at("original_start");
          float original_end = chunk.at("original_end");
          float original_duration = original_end - original_start;

          return original_start + relative_position * original_duration;
        }
      }
    }

    // Fallback: return original time
    return t;
  }
};

std::vector<Segment> restore_speech_timestamps(
  std::vector<Segment> segments,
  const std::vector<std::map<std::string, float>> &speech_chunks,
  int sampling_rate
) {
  if (speech_chunks.empty()) {
    return segments; // No restoration needed
  }

  SpeechTimestampsMap ts_map(speech_chunks, sampling_rate);

  for (auto &segment: segments) {
    if (segment.words.has_value() && !segment.words.value().empty()) {
      std::vector<Word> words;
      for (auto word: segment.words.value()) {
        float middle = (word.start + word.end) / 2.0f;
        int chunk_index = ts_map.get_chunk_index(middle);

        // Restore original timestamps for word boundaries
        word.start = ts_map.get_original_time(word.start, chunk_index, false);
        word.end = ts_map.get_original_time(word.end, chunk_index, true);

        words.push_back(word);
      }

      // Update segment boundaries based on word timestamps
      if (!words.empty()) {
        segment.start = words.front().start;
        segment.end = words.back().end;
        segment.words = words;
      }
    } else {
      // Restore segment timestamps directly
      int start_chunk = ts_map.get_chunk_index(segment.start);
      int end_chunk = ts_map.get_chunk_index(segment.end);

      segment.start = ts_map.get_original_time(segment.start, start_chunk, false);
      segment.end = ts_map.get_original_time(segment.end, end_chunk, true);
    }
  }

  return segments;
}

// Additional utility functions for feature processing

std::vector<std::vector<float>> normalize_features(const std::vector<std::vector<float>>& features) {
  if (features.empty() || features[0].empty()) {
    return features;
  }

  std::vector<std::vector<float>> normalized = features;

  // Normalize each mel bin (row) independently
  for (auto& feature_row : normalized) {
    // Calculate mean and std dev
    float sum = std::accumulate(feature_row.begin(), feature_row.end(), 0.0f);
    float mean = sum / feature_row.size();

    float sq_sum = 0.0f;
    for (float val : feature_row) {
      sq_sum += (val - mean) * (val - mean);
    }
    float std_dev = std::sqrt(sq_sum / feature_row.size());

    // Normalize: (x - mean) / std_dev
    if (std_dev > 1e-8f) { // Avoid division by zero
      for (float& val : feature_row) {
        val = (val - mean) / std_dev;
      }
    }
  }

  return normalized;
}

std::vector<std::vector<float>> apply_log_mel_spectrogram(const std::vector<std::vector<float>>& mel_features) {
  std::vector<std::vector<float>> log_mel = mel_features;

  // Apply log transformation: log(max(features, 1e-10))
  for (auto& feature_row : log_mel) {
    for (float& val : feature_row) {
      val = std::log(std::max(val, 1e-10f));
    }
  }

  return log_mel;
}

float calculate_signal_to_noise_ratio(const std::vector<float>& audio) {
  if (audio.empty()) {
    return 0.0f;
  }

  // Calculate RMS (Root Mean Square) for signal power
  float sum_squares = 0.0f;
  for (float sample : audio) {
    sum_squares += sample * sample;
  }
  float rms = std::sqrt(sum_squares / audio.size());

  // Estimate noise level (simplified approach using quieter segments)
  std::vector<float> sorted_audio = audio;
  std::sort(sorted_audio.begin(), sorted_audio.end(), [](float a, float b) {
    return std::abs(a) < std::abs(b);
  });

  // Use bottom 10% as noise estimate
  size_t noise_samples = sorted_audio.size() / 10;
  float noise_sum_squares = 0.0f;
  for (size_t i = 0; i < noise_samples; ++i) {
    noise_sum_squares += sorted_audio[i] * sorted_audio[i];
  }
  float noise_rms = std::sqrt(noise_sum_squares / noise_samples);

  // Calculate SNR in dB
  if (noise_rms > 1e-10f) {
    return 20.0f * std::log10(rms / noise_rms);
  }

  return 60.0f; // Very high SNR if noise is negligible
}

bool is_silent_segment(const std::vector<float>& audio_segment, float threshold = 0.01f) {
  if (audio_segment.empty()) {
    return true;
  }

  // Calculate RMS energy
  float sum_squares = 0.0f;
  for (float sample : audio_segment) {
    sum_squares += sample * sample;
  }
  float rms = std::sqrt(sum_squares / audio_segment.size());

  return rms < threshold;
}

std::vector<std::pair<float, float>> detect_speech_activity(
  const std::vector<float>& audio,
  int sampling_rate,
  float frame_duration = 0.025f, // 25ms frames
  float energy_threshold = 0.01f
) {
  std::vector<std::pair<float, float>> speech_segments;

  int frame_size = static_cast<int>(frame_duration * sampling_rate);
  int hop_size = frame_size / 2; // 50% overlap

  std::vector<bool> is_speech;

  // Analyze each frame for speech activity
  for (int i = 0; i + frame_size <= static_cast<int>(audio.size()); i += hop_size) {
    std::vector<float> frame(audio.begin() + i, audio.begin() + i + frame_size);
    bool frame_has_speech = !is_silent_segment(frame, energy_threshold);
    is_speech.push_back(frame_has_speech);
  }

  // Convert frame-level decisions to time segments
  float time_per_frame = static_cast<float>(hop_size) / sampling_rate;
  bool in_speech = false;
  float speech_start = 0.0f;

  for (size_t i = 0; i < is_speech.size(); ++i) {
    float current_time = i * time_per_frame;

    if (is_speech[i] && !in_speech) {
      // Speech starts
      speech_start = current_time;
      in_speech = true;
    } else if (!is_speech[i] && in_speech) {
      // Speech ends
      speech_segments.emplace_back(speech_start, current_time);
      in_speech = false;
    }
  }

  // Handle case where speech continues to the end
  if (in_speech) {
    float end_time = static_cast<float>(audio.size()) / sampling_rate;
    speech_segments.emplace_back(speech_start, end_time);
  }

  return speech_segments;
}