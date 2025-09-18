#pragma once

#include <vector>
#include <string>
#include <optional>
#include <map>

// Simple chunk metadata (mirrors Python dicts with start/end samples)
struct Chunk {
  int start;
  int end;
};

// Returned chunk metadata (offset, duration, segments)
struct ChunkMetadata {
  double offset;        // seconds
  double duration;      // seconds
  std::vector<Chunk> segments;
};

struct VadOptions {
  float threshold = 0.5f;
  // If neg_threshold is std::nullopt, we compute it as max(threshold-0.15, 0.01)
  std::optional<float> neg_threshold = std::nullopt;
  int min_speech_duration_ms = 0;
  double max_speech_duration_s = std::numeric_limits<double>::infinity();
  int min_silence_duration_ms = 2000;
  int speech_pad_ms = 400;
};

class SileroVADModel {
public:
  // Provide full paths to the encoder and decoder ONNX files
  SileroVADModel(const std::string& encoder_path, const std::string& decoder_path);
  ~SileroVADModel();

  // Call the VAD model. Input is a 2D batch as vector<float> flattened in row-major:
  // shape: (batch_size, num_samples) -> flattened.size() == batch_size * num_samples
  // Returns output shape: (batch_size, num_frames) flattened to batch-major order
  std::vector<float> operator()(const std::vector<float>& audio_2d,
                                int batch_size,
                                int num_samples,
                                int context_size_samples = 64);

private:
  // PIMPL or direct members from ONNX Runtime
  struct Impl;
  Impl* impl_;
};

// Main API: returns vector of speech segments (start/end samples)
std::vector<Chunk> get_speech_timestamps(const std::vector<float>& audio,
                                         const VadOptions& vad_options = VadOptions(),
                                         int sampling_rate = 16000);

// Merge chunks into audio chunks up to max_duration seconds and return audio arrays + metadata.
// audio is 1D float array.
// Returns pair of (vector of audio chunks as vector<float>), vector<ChunkMetadata>
std::pair<std::vector<std::vector<float>>, std::vector<ChunkMetadata>>
collect_chunks(const std::vector<float>& audio,
               const std::vector<Chunk>& chunks,
               int sampling_rate = 16000,
               double max_duration = std::numeric_limits<double>::infinity());

// Helper that maps speech chunks back to original timeline (used by restore_speech_timestamps)
class SpeechTimestampsMap {
public:
  // chunks: vector of Chunk {start, end} in samples, sampling_rate in Hz
  SpeechTimestampsMap(const std::vector<Chunk>& chunks, int sampling_rate, int time_precision = 2);

  // map a time (seconds) relative to a chunk back to original audio timeline.
  // if chunk_index is -1, get_chunk_index will be used to find an index
  double get_original_time(double time_seconds, int chunk_index = -1, bool is_end = false) const;

  // find the chunk index that contains given time_seconds
  int get_chunk_index(double time_seconds, bool is_end = false) const;

private:
  int sampling_rate_;
  int time_precision_;
  std::vector<int> chunk_end_sample_;        // end sample index in condensed timeline
  std::vector<double> total_silence_before_; // seconds of silence before each condensed chunk
};
