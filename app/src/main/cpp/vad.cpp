#include "vad.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <iostream>

// NOTE: you must have ONNX Runtime C++ headers available and linked.
// Example include (depends on your installation path):
#include <onnxruntime_cxx_api.h>

// --------------------- SileroVADModel implementation ---------------------
struct SileroVADModel::Impl {
  Ort::Env env;
  Ort::SessionOptions session_options;
  std::unique_ptr<Ort::Session> encoder_session;
  std::unique_ptr<Ort::Session> decoder_session;
  Ort::AllocatorWithDefaultOptions allocator;

  Impl(const std::string& enc_path, const std::string& dec_path)
      : env(ORT_LOGGING_LEVEL_WARNING, "silero_vad") {
    session_options.SetIntraOpNumThreads(1);
    session_options.SetInterOpNumThreads(1);
    session_options.EnableCpuMemArena(false);
    session_options.SetLogSeverityLevel(4);

    encoder_session.reset(new Ort::Session(env, enc_path.c_str(), session_options));
    decoder_session.reset(new Ort::Session(env, dec_path.c_str(), session_options));
  }
};

SileroVADModel::SileroVADModel(const std::string& encoder_path, const std::string& decoder_path) {
  impl_ = new Impl(encoder_path, decoder_path);
}

SileroVADModel::~SileroVADModel() {
  delete impl_;
}

// Helper to run an ONNX session with a single float input (2D: batch_size x num_samples)
// and return the first output as a vector<float>. This function assumes the model input
// is a single float tensor and the output is a single float tensor.
static std::vector<float> run_onnx_session(Ort::Session& session,
                                           const std::vector<float>& input,
                                           const std::vector<int64_t>& input_shape) {
  Ort::AllocatorWithDefaultOptions allocator;
  size_t input_tensor_size = 1;
  for (auto d : input_shape) input_tensor_size *= d;

  std::vector<int64_t> input_dims = input_shape;
  // create memory info
  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  // create tensor
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(mem_info,
                                                            const_cast<float*>(input.data()),
                                                            input_tensor_size,
                                                            input_dims.data(),
                                                            input_dims.size());

  // get input name
  size_t num_input_nodes = session.GetInputCount();
  std::vector<const char*> input_node_names(num_input_nodes);
  for (size_t i = 0; i < num_input_nodes; ++i) {
    char* name = session.GetInputName(i, allocator);
    input_node_names[i] = name;
  }

  // get output names
  size_t num_output_nodes = session.GetOutputCount();
  std::vector<const char*> output_node_names(num_output_nodes);
  for (size_t i = 0; i < num_output_nodes; ++i) {
    char* name = session.GetOutputName(i, allocator);
    output_node_names[i] = name;
  }

  // run
  auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                    input_node_names.data(),
                                    &input_tensor,
                                    1,
                                    output_node_names.data(),
                                    output_node_names.size());

  // assume first output is float array
  float* out_data = output_tensors[0].GetTensorMutableData<float>();
  auto out_info = output_tensors[0].GetTensorTypeAndShapeInfo();
  auto out_shape = out_info.GetShape();
  size_t out_size = 1;
  for (auto d : out_shape) out_size *= (d > 0 ? d : 1);

  std::vector<float> out_vec(out_data, out_data + out_size);

  // free input and output name strings allocated by Ort::Allocator
  for (size_t i = 0; i < num_input_nodes; ++i) {
    allocator.Free(const_cast<char*>(input_node_names[i]));
  }
  for (size_t i = 0; i < num_output_nodes; ++i) {
    allocator.Free(const_cast<char*>(output_node_names[i]));
  }

  return out_vec;
}

std::vector<float> SileroVADModel::operator()(const std::vector<float>& audio_2d,
                                              int batch_size,
                                              int num_samples,
                                              int context_size_samples) {
  // audio_2d is batch-major flattened (batch_size * num_samples)
  if (batch_size <= 0 || num_samples <= 0) throw std::runtime_error("Invalid shape");
  std::vector<float> encoder_out;

  // run encoder on the flattened batched audio
  {
    // encoder input shape: [batch_size, num_channels?, num_samples] in original Python model
    // Silero encoder usually expects [batch_size, num_samples]. Here we pass [batch_size, num_samples]
    std::vector<int64_t> input_shape = {batch_size, num_samples};
    encoder_out = run_onnx_session(*impl_->encoder_session, audio_2d, input_shape);
  }

  // `encoder_out` shape is (batch_size, seq_len, hidden). In Python they reshape / slice.
  // For simplicity, we will feed the encoder outputs in windows to the decoder similarly:
  // We will assume the encoder output can be split per-window and decoder returns per-window prob.

  // The exact shape handling depends on your silero model layout. Here we attempt to mimic the Python path:
  // encoder_outputs were concatenated and then reshaped into (batch_size, -1, 128) in Python.
  // We don't have full shape info here — this may need adjustment depending on your Silero ONNX format.

  // For demo purpose, assume decoder accepts (batch_size, seq_len, hidden) windows (1 window at a time).
  // We will now call decoder sequentially for each window along axis 1.

  // We need info about shapes from the encoder_out tensor. For robust code, query output shape before.
  // For brevity here, assume encoder_out is already in the right flattened form and the decoder expects
  // input shape [batch_size, 1, hidden] per window. This part *may* need adaptation.

  // Very simplified: assume decoder returns shape (batch_size, windows) or (batch_size * windows).
  // We'll call the decoder once with encoder_out and return its output.
  std::vector<int64_t> decoder_input_shape = { /* depends on model */ };
  // Attempt to call decoder once with encoder_out
  std::vector<float> decoder_out = run_onnx_session(*impl_->decoder_session,
                                                    encoder_out,
      /* input shape guess */ std::vector<int64_t>{1, static_cast<int64_t>(encoder_out.size())});
  // For now return decoder_out as-is (flattened)
  return decoder_out;
}

// --------------------- get_speech_timestamps implementation ---------------------
std::vector<Chunk> get_speech_timestamps(const std::vector<float>& audio,
                                         const VadOptions& vad_options,
                                         int sampling_rate) {
  // Copy values from vad_options
  float threshold = vad_options.threshold;
  float neg_threshold = vad_options.neg_threshold.has_value()
                        ? vad_options.neg_threshold.value()
                        : std::max(threshold - 0.15f, 0.01f);
  int min_speech_duration_ms = vad_options.min_speech_duration_ms;
  double max_speech_duration_s = vad_options.max_speech_duration_s;
  int min_silence_duration_ms = vad_options.min_silence_duration_ms;
  int window_size_samples = 512;
  int speech_pad_ms = vad_options.speech_pad_ms;

  double min_speech_samples = (double)sampling_rate * min_speech_duration_ms / 1000.0;
  int speech_pad_samples = static_cast<int>(std::round(sampling_rate * speech_pad_ms / 1000.0));
  double max_speech_samples = (double)sampling_rate * max_speech_duration_s
                              - window_size_samples - 2.0 * speech_pad_samples;
  int min_silence_samples = static_cast<int>(std::round(sampling_rate * min_silence_duration_ms / 1000.0));
  int min_silence_samples_at_max_speech = static_cast<int>(std::round(sampling_rate * 98.0 / 1000.0));

  size_t audio_length_samples = audio.size();

  // Lazily constructed static Silero model; in production replace with proper singleton or injected dependency
  // You must set correct encoder/decoder paths here:
  static std::unique_ptr<SileroVADModel> model;
  static bool model_initialized = false;
  if (!model_initialized) {
    // You must change these paths to where you keep silero encoder/decoder ONNX
    const std::string encoder_path = "silero_encoder_v5.onnx";
    const std::string decoder_path = "silero_decoder_v5.onnx";
    model.reset(new SileroVADModel(encoder_path, decoder_path));
    model_initialized = true;
  }

  // Pad audio to multiple of window_size_samples
  int pad_len = (int)( ( (audio_length_samples + window_size_samples - 1) / window_size_samples) * window_size_samples );
  std::vector<float> padded_audio = audio;
  padded_audio.resize(pad_len, 0.0f);

  // Build batched input: reshape to (1, -1) windows of size 512; Silero model expects 2D (batch, samples) in wrapper
  // Python did: model(padded_audio.reshape(1, -1)).squeeze(0) => produced probabilities per window
  // We will create a batch of windows where each window is 512 samples (but Silero encoder in Python used context rolling)
  // Simpler approach: feed the whole padded_audio as single batch of length pad_len; the model wrapper should handle slicing.
  std::vector<float> batch_input = padded_audio; // as 1D representing (1, pad_len)
  // Call model: wrapper expects (batch_size, num_samples) flattened; here batch_size=1, num_samples=pad_len
  std::vector<float> speech_probs = (*model)(batch_input, 1, pad_len, 64);

  // In Python speech_probs length equals pad_len / window_size_samples
  // If decoder output is per-frame, adapt length:
  int num_frames = static_cast<int>(speech_probs.size());
  // Now we implement the same triggered logic as Python
  bool triggered = false;
  std::vector<Chunk> speeches;
  std::optional<Chunk> current_speech;
  int temp_end = 0;
  int prev_end = 0;
  int next_start = 0;

  for (int i = 0; i < num_frames; ++i) {
    float speech_prob = speech_probs[i];
    int window_sample_pos = i * window_size_samples;

    if (speech_prob >= threshold && temp_end) {
      temp_end = 0;
      if (next_start < prev_end) next_start = window_sample_pos;
    }

    if (speech_prob >= threshold && !triggered) {
      triggered = true;
      current_speech = Chunk{window_sample_pos, 0};
      continue;
    }

    if (triggered && (window_sample_pos - current_speech->start) > max_speech_samples) {
      if (prev_end) {
        current_speech->end = prev_end;
        speeches.push_back(*current_speech);
        current_speech.reset();
        if (next_start < prev_end) {
          triggered = false;
        } else {
          current_speech = Chunk{next_start, 0};
        }
        prev_end = next_start = temp_end = 0;
      } else {
        current_speech->end = window_sample_pos;
        speeches.push_back(*current_speech);
        current_speech.reset();
        prev_end = next_start = temp_end = 0;
        triggered = false;
      }
      continue;
    }

    if (speech_prob < neg_threshold && triggered) {
      if (!temp_end) temp_end = window_sample_pos;
      if ((window_sample_pos - temp_end) > min_silence_samples_at_max_speech) {
        prev_end = temp_end;
      }
      if ((window_sample_pos - temp_end) < min_silence_samples) {
        continue;
      } else {
        current_speech->end = temp_end;
        if ((current_speech->end - current_speech->start) > min_speech_samples) {
          speeches.push_back(*current_speech);
        }
        current_speech.reset();
        prev_end = next_start = temp_end = 0;
        triggered = false;
        continue;
      }
    }
  }

  // Post loop: if current_speech exists and remaining audio is long enough
  if (current_speech.has_value() && ( (int)audio_length_samples - current_speech->start) > min_speech_samples) {
    current_speech->end = static_cast<int>(audio_length_samples);
    speeches.push_back(*current_speech);
  }

  // Apply padding and merge heuristics (same as Python)
  for (size_t i = 0; i < speeches.size(); ++i) {
    if (i == 0) {
      // pad start
      int s = std::max(0, speeches[i].start - speech_pad_samples);
      speeches[i].start = s;
    }
    if (i + 1 != speeches.size()) {
      int silence_duration = speeches[i+1].start - speeches[i].end;
      if (silence_duration < 2 * speech_pad_samples) {
        // split half-half
        int add = silence_duration / 2;
        speeches[i].end += add;
        speeches[i+1].start = std::max(0, speeches[i+1].start - add);
      } else {
        speeches[i].end = std::min<int>(audio_length_samples, speeches[i].end + speech_pad_samples);
        speeches[i+1].start = std::max(0, speeches[i+1].start - speech_pad_samples);
      }
    } else {
      speeches[i].end = std::min<int>(audio_length_samples, speeches[i].end + speech_pad_samples);
    }
  }

  return speeches;
}

// --------------------- collect_chunks implementation ---------------------
std::pair<std::vector<std::vector<float>>, std::vector<ChunkMetadata>>
collect_chunks(const std::vector<float>& audio,
               const std::vector<Chunk>& chunks,
               int sampling_rate,
               double max_duration) {
  if (chunks.empty()) {
    ChunkMetadata meta;
    meta.offset = 0.0;
    meta.duration = 0.0;
    meta.segments = {};
    std::vector<float> empty_audio;
    return { {empty_audio}, {meta} };
  }

  std::vector<std::vector<float>> audio_chunks;
  std::vector<ChunkMetadata> chunks_metadata;

  std::vector<Chunk> current_segments;
  int current_duration = 0;
  double total_duration = 0.0;
  std::vector<float> current_audio;

  for (const auto& chunk : chunks) {
    int chunk_len = chunk.end - chunk.start;
    if ((current_duration + chunk_len) > static_cast<int>(max_duration * sampling_rate)) {
      audio_chunks.push_back(current_audio);
      ChunkMetadata meta;
      meta.offset = total_duration / sampling_rate;
      meta.duration = current_duration / sampling_rate;
      meta.segments = current_segments;
      total_duration += current_duration;
      chunks_metadata.push_back(meta);

      current_segments.clear();
      current_audio.clear();
      // copy new chunk audio
      current_audio.insert(current_audio.end(), audio.begin() + chunk.start, audio.begin() + chunk.end);
      current_duration = chunk_len;
      current_segments.push_back(chunk); // first in new segment
    } else {
      // append chunk into current_audio
      current_segments.push_back(chunk);
      current_audio.insert(current_audio.end(), audio.begin() + chunk.start, audio.begin() + chunk.end);
      current_duration += chunk_len;
    }
  }

  // push last
  audio_chunks.push_back(current_audio);
  ChunkMetadata meta;
  meta.offset = total_duration / sampling_rate;
  meta.duration = current_duration / sampling_rate;
  meta.segments = current_segments;
  chunks_metadata.push_back(meta);

  return {audio_chunks, chunks_metadata};
}


// --------------------- SpeechTimestampsMap implementation ---------------------
SpeechTimestampsMap::SpeechTimestampsMap(const std::vector<Chunk>& chunks, int sampling_rate, int time_precision)
    : sampling_rate_(sampling_rate), time_precision_(time_precision) {
  int previous_end = 0;
  int silent_samples = 0;
  for (const auto& chunk : chunks) {
    silent_samples += chunk.start - previous_end;
    previous_end = chunk.end;
    chunk_end_sample_.push_back(chunk.end - silent_samples);
    total_silence_before_.push_back(static_cast<double>(silent_samples) / sampling_rate_);
  }
}

double SpeechTimestampsMap::get_original_time(double time_seconds, int chunk_index, bool is_end) const {
  if (chunk_index == -1) {
    chunk_index = get_chunk_index(time_seconds, is_end);
  }
  double total_silence_before = total_silence_before_.at(chunk_index);
  double original = total_silence_before + time_seconds;
  // round to time_precision_ decimal digits
  double factor = std::pow(10.0, time_precision_);
  return std::round(original * factor) / factor;
}

int SpeechTimestampsMap::get_chunk_index(double time_seconds, bool is_end) const {
  int sample = static_cast<int>(std::round(time_seconds * sampling_rate_));
  if (is_end) {
    auto it = std::find(chunk_end_sample_.begin(), chunk_end_sample_.end(), sample);
    if (it != chunk_end_sample_.end()) return static_cast<int>(it - chunk_end_sample_.begin());
  }
  // bisect behavior: first index > sample
  auto it = std::upper_bound(chunk_end_sample_.begin(), chunk_end_sample_.end(), sample);
  int idx = static_cast<int>(it - chunk_end_sample_.begin());
  if (idx >= static_cast<int>(chunk_end_sample_.size())) idx = static_cast<int>(chunk_end_sample_.size()) - 1;
  return idx;
}
