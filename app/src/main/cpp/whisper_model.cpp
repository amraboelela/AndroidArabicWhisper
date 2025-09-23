
#include "whisper_model.h"
#include "utils.h"
#include <ctranslate2/models/whisper.h>
#include <ctranslate2/storage_view.h>
#include <string>
#include <memory>
#include <filesystem>
#include <iostream>
#include <vector>
#include <map>
#include <tuple>
#include <optional>
#include <cmath>
#include <algorithm>
#include <variant>

// Forward declarations of utility functions
std::vector<std::vector<float>> slice_features(const std::vector<std::vector<float>>& features, int start, int length);
ctranslate2::StorageView get_ctranslate2_storage(const std::vector<std::vector<float>>& features);
float get_compression_ratio(const std::string& text);
void merge_punctuations(std::vector<std::pair<std::string, float>>& alignment,
                       const std::vector<std::string>& prepend_punctuations,
                       const std::vector<std::string>& append_punctuations);
std::vector<std::vector<float>> pad_or_trim(const std::vector<std::vector<float>>& segment);
#include <stdexcept>
#include <numeric>
#include <cassert>
#include <set>
#include <zlib.h>
#include <cstring>
#include <variant>
#include <utility>
#include "tokenizer.h"
#include "audio.h"
#include "feature_extractor.h"

// Forward declarations and constants

// Logger placeholder
struct Logger {
    void debug(const char* format, ...) const {
        // Simple logging implementation
    }
};
static Logger logger;

namespace fs = std::filesystem;

WhisperModel::WhisperModel(
    const std::string &model_size_or_path,
    const std::string &device,
    const std::vector<int> &device_index,
    const std::string &compute_type,
    int cpu_threads,
    int num_workers,
    const std::string &download_root,
    bool local_files_only,
    const std::map<std::string, std::string> &files,
    const std::string &revision,
    const std::string &use_auth_token
) {
  //std::string model_path;
  //std::string preprocessor_config;

  std::string model_path;
  if (!files.empty()) {
    // If model files are already provided in memory (not implemented here)
    model_path = model_size_or_path;
  } else if (std::filesystem::is_directory(model_size_or_path)) {
    model_path = model_size_or_path;
  } else {
    // In Python: download_model(...)
    // In C++: You must implement downloading manually or assume pre-downloaded
    model_path = model_size_or_path;
  }

  ctranslate2::ReplicaPoolConfig config;
  config.num_threads_per_replica = cpu_threads;   // map your params here

  // Initialize the CTranslate2 model.
  model = std::make_shared<ctranslate2::models::Whisper>(
      model_path,
      ctranslate2::Device::CPU,
      ctranslate2::ComputeType::DEFAULT,
      device_index,
      false,
      config
  );

  // Initialize tokenizer placeholder
  hf_tokenizer = nullptr;

  // -------------------
  // Tokenizer Handling
  // -------------------
  // In Python: tokenizers.Tokenizer.from_file("tokenizer.json")
  // In C++: you must implement or use a tokenizer wrapper
  std::string tokenizer_file = model_path + "/tokenizer.json";
  if (std::filesystem::exists(tokenizer_file)) {
    std::cout << "Load tokenizer from: " << tokenizer_file << std::endl;
    // TODO: integrate Hugging Face tokenizers (Rust) or your own
  } else {
    std::cerr << "Tokenizer not found, defaulting to fallback.\n";
  }

  // Placeholder for feature_kwargs logic.
  // In a real implementation, this would parse preprocessor_config.json.
  // We assume default parameters here as in the Python `FeatureExtractor`.
  feature_extractor = FeatureExtractor();

  input_stride = 2;
  num_samples_per_token = feature_extractor.hop_length * input_stride;
  frames_per_second = feature_extractor.sampling_rate() / feature_extractor.hop_length;
  tokens_per_second = feature_extractor.sampling_rate() / num_samples_per_token;
  time_precision = 0.02;
  max_length = 448;
}

std::vector<std::string> WhisperModel::supported_languages() const {
  if (model->is_multilingual()) {
    return _LANGUAGE_CODES; // assume you have a constant vector of strings
  }
  return {"en"};
}

std::map<std::string, std::string> WhisperModel::get_feature_kwargs(
    const std::string &model_path,
    const std::optional<std::string> &preprocessor_bytes
) {
  std::map<std::string, std::string> config;
  try {
    std::string config_path = model_path + "/preprocessor_config.json";
    if (preprocessor_bytes.has_value()) {
      config = parse_json(preprocessor_bytes.value());
    } else if (std::filesystem::exists(config_path)) {
      config = parse_json_file(config_path);
    }

    // Optionally filter keys to match your FeatureExtractor constructor
    return config;
  } catch (const std::exception &e) {
    std::cerr << "Could not load preprocessor config: " << e.what() << std::endl;
  }
  return config;
}

std::tuple<std::vector<Segment>, TranscriptionInfo> WhisperModel::transcribe(
    const std::vector<float> &audio,
    const std::optional<std::string> &language,
    bool multilingual
) {
  // Detect language if multilingual
  std::string lang = language.value_or("en");
  float language_probability = 1.0;

  if (multilingual && !model->is_multilingual()) {
    std::cerr << "Model is English-only; disabling multilingual mode." << std::endl;
    multilingual = false;
  }

  // -----------------
  // Feature extraction
  // -----------------
  std::vector<float> processed_audio = audio;
  auto features = feature_extractor.extract(processed_audio);

  // -----------------
  // Tokenizer
  // -----------------
  Tokenizer tokenizer(hf_tokenizer.get(), model->is_multilingual(), std::nullopt, lang);

  // -----------------
  // Generate segments
  // -----------------
  TranscriptionOptions options;
  // fill options as needed
  std::vector<Segment> segments = generate_segments(features, tokenizer, options);

  // -----------------
  // -----------------
  // Construct TranscriptionInfo
  // -----------------
  TranscriptionInfo info;
  info.language = lang;
  info.language_probability = language_probability;
  info.duration = static_cast<float>(audio.size()) / feature_extractor.sampling_rate();
  info.transcription_options = options;

  return {segments, info};
}

std::tuple<std::vector<Segment>, int, bool> WhisperModel::split_segments_by_timestamps(
    Tokenizer &tokenizer,
    const std::vector<int> &tokens,
    float time_offset,
    int segment_size,
    float segment_duration,
    int seek
) {
  std::vector<Segment> current_segments;
  bool single_timestamp_ending = (tokens.size() >= 2 &&
                                  tokens[tokens.size() - 2] < tokenizer.get_timestamp_begin() &&
                                  tokens.back() >= tokenizer.get_timestamp_begin());

  std::vector<int> consecutive_timestamps;
  for (size_t i = 1; i < tokens.size(); ++i) {
    if (tokens[i] >= tokenizer.get_timestamp_begin() && tokens[i - 1] >= tokenizer.get_timestamp_begin()) {
      consecutive_timestamps.push_back(static_cast<int>(i));
    }
  }

  if (!consecutive_timestamps.empty()) {
    std::vector<int> slices = consecutive_timestamps;
    if (single_timestamp_ending) slices.push_back(tokens.size());

    int last_slice = 0;
    for (int current_slice: slices) {
      std::vector<int> sliced_tokens(tokens.begin() + last_slice, tokens.begin() + static_cast<std::vector<int>::difference_type>(current_slice));
      float start_time =
          time_offset + (sliced_tokens.front() - tokenizer.get_timestamp_begin()) * static_cast<float>(time_precision);
      float end_time =
          time_offset + (sliced_tokens.back() - tokenizer.get_timestamp_begin()) * static_cast<float>(time_precision);

      Segment seg;
      seg.seek = seek;
      seg.start = start_time;
      seg.end = end_time;
      seg.tokens = sliced_tokens;
      current_segments.push_back(seg);
      last_slice = current_slice;
    }

    if (single_timestamp_ending) {
      seek += segment_size;
    } else {
      int last_timestamp_position = tokens[last_slice - 1] - tokenizer.get_timestamp_begin();
      seek += static_cast<int>(last_timestamp_position) * input_stride;
    }
  } else {
    float duration = segment_duration;
    std::vector<int> timestamps;
    for (int token: tokens) if (token >= tokenizer.get_timestamp_begin()) timestamps.push_back(token);

    if (!timestamps.empty() && timestamps.back() != tokenizer.get_timestamp_begin()) {
      duration = (timestamps.back() - tokenizer.get_timestamp_begin()) * static_cast<float>(time_precision);
    }

    Segment seg;
    seg.seek = seek;
    seg.start = time_offset;
    seg.end = time_offset + duration;
    seg.tokens = tokens;
    current_segments.push_back(seg);
    seek += segment_size;
  }

  return {current_segments, seek, single_timestamp_ending};
}

std::vector<Segment> WhisperModel::generate_segments(
    const std::vector<std::vector<float>> &features,
    Tokenizer &tokenizer,
    const TranscriptionOptions &options
) {
  int content_frames = features[0].size() - 1;
  float content_duration = content_frames * feature_extractor.time_per_frame();
  std::vector<int> seek_points;
  std::vector<std::pair<int, int>> seek_clips;

  // Process clip_timestamps
  std::vector<float> timestamps;

  if (std::holds_alternative<std::vector<float>>(options.clip_timestamps)) {
    timestamps = std::get<std::vector<float>>(options.clip_timestamps);
  } else if (std::holds_alternative<std::string>(options.clip_timestamps)) {
    // Parse comma-separated string - simple implementation
    std::string ts_str = std::get<std::string>(options.clip_timestamps);
    // For now, just use empty vector if it's a string
    // In a real implementation, you'd parse the comma-separated string
  }

  for (float ts : timestamps) {
    seek_points.push_back(std::round(ts * frames_per_second));
  }
  if (seek_points.empty()) seek_points.push_back(0);
  if (seek_points.size() % 2 == 1) seek_points.push_back(content_frames);

  for (size_t i = 0; i < seek_points.size(); i += 2) {
    seek_clips.emplace_back(seek_points[i], seek_points[i + 1]);
  }

  std::vector<Segment> all_segments;
  int clip_idx = 0;
  int seek = seek_clips[clip_idx].first;

  std::vector<int> all_tokens;
  int prompt_reset_since = 0;

  // Initial prompt
  if (options.initial_prompt.has_value()) {
    std::vector<int> initial_tokens;
    const auto& prompt_variant = options.initial_prompt.value();

    if (std::holds_alternative<std::string>(prompt_variant)) {
      initial_tokens = tokenizer.encode(std::get<std::string>(prompt_variant));
    } else if (std::holds_alternative<std::vector<int>>(prompt_variant)) {
      initial_tokens = std::get<std::vector<int>>(prompt_variant);
    }

    all_tokens.insert(all_tokens.end(), initial_tokens.begin(), initial_tokens.end());
  }

  float last_speech_timestamp = 0.0;

  while (clip_idx < seek_clips.size()) {
    auto [seek_clip_start, seek_clip_end] = seek_clips[clip_idx];
    if (seek_clip_end > content_frames) seek_clip_end = content_frames;
    if (seek < seek_clip_start) seek = seek_clip_start;
    if (seek >= seek_clip_end) {
      clip_idx++;
      if (clip_idx < seek_clips.size()) seek = seek_clips[clip_idx].first;
      continue;
    }

    float time_offset = seek * feature_extractor.time_per_frame();
    int segment_size = std::min({feature_extractor.nb_max_frames(),
                                 content_frames - seek,
                                 seek_clip_end - seek});
    auto segment_features = slice_features(features, seek, segment_size);
    segment_features = pad_or_trim(segment_features);

    // Encode segment
    auto encoder_output = encode(segment_features);

    // Generate tokens
    std::vector<int> empty_prompt;
    auto [tokens, avg_logprob, temperature, compression_ratio] = generate_with_fallback(
        encoder_output, empty_prompt, tokenizer, options);

    // Split segments by timestamps
    auto [current_segments, new_seek, single_timestamp_ending] =
        split_segments_by_timestamps(tokenizer, tokens, time_offset, segment_size,
                                     segment_size * feature_extractor.time_per_frame(), seek);

    seek = new_seek;

    // Decode tokens to text
    for (auto &seg: current_segments) {
      seg.text = tokenizer.decode(seg.tokens);
      if (!seg.text.empty() && seg.start != seg.end) {
        all_segments.push_back(seg);
        all_tokens.insert(all_tokens.end(), seg.tokens.begin(), seg.tokens.end());
      }
    }

    prompt_reset_since = all_tokens.size();
  }

  return all_segments;
}

// --------------------------
// Encode features using the Whisper model
// --------------------------
ctranslate2::StorageView WhisperModel::encode(const std::vector<std::vector<float>> &features) {
  bool to_cpu = false; // Simplified for CPU-only build

  std::vector<std::vector<float>> input_features = features;

  // Expand dims if 2D -> add batch dimension
  if (input_features.size() > 0 && input_features[0].size() > 0) {
    // Assuming 2D features -> wrap in batch of 1
    if (input_features.size() == features.size() &&
        input_features[0].size() == features[0].size()) {
      input_features = {features};
    }
  }

  auto storage = get_ctranslate2_storage(input_features);
  auto future = model->encode(storage, to_cpu);
  return future.get();
}

// --------------------------
// Generate with fallback loop over temperatures
// --------------------------
std::tuple<std::vector<int>, float, float, float>
WhisperModel::generate_with_fallback(
    const ctranslate2::StorageView &encoder_output,
    const std::vector<int> &prompt,
    Tokenizer &tokenizer,
    const TranscriptionOptions &options
) {
  ctranslate2::models::WhisperGenerationResult best_result;
  float best_avg_logprob = -std::numeric_limits<float>::infinity();
  float best_temperature = 0.0f;
  float best_compression_ratio = 1.0f;

  std::vector<std::tuple<std::vector<int>, float, float, float>> all_results;
  std::vector<std::tuple<std::vector<int>, float, float, float>> below_cr_threshold_results;

  int max_initial_timestamp_index = static_cast<int>(
      std::round(options.max_initial_timestamp / time_precision)
  );

  int max_length = options.max_new_tokens.has_value() ? prompt.size() +
                                                        options.max_new_tokens.value()
                                                      : this->max_length;
  if (max_length > this->max_length) {
    throw std::runtime_error("Prompt + max_new_tokens exceeds Whisper max_length");
  }

  for (float temperature: options.temperatures) {
    std::map<std::string, float> kwargs;

    if (temperature > 0) {
      kwargs["beam_size"] = 1;
      kwargs["num_hypotheses"] = options.best_of;
      kwargs["sampling_topk"] = 0;
      kwargs["sampling_temperature"] = temperature;
    } else {
      kwargs["beam_size"] = options.beam_size;
      kwargs["patience"] = options.patience;
    }

    // Create WhisperOptions for CTranslate2
    ctranslate2::models::WhisperOptions whisper_options;

    // Convert vector<int> to vector<size_t> for CTranslate2 API
    std::vector<size_t> prompt_size_t(prompt.begin(), prompt.end());
    std::vector<std::vector<size_t>> prompts = {prompt_size_t};

    auto result_futures = model->generate(encoder_output, prompts, whisper_options);

    // Get the result from the future
    auto result = result_futures[0].get();

    // Get the first sequence from the results
    std::vector<int> tokens;
    if (!result.sequences_ids.empty() && !result.sequences_ids[0].empty()) {
      const auto &tokens_size_t = result.sequences_ids[0];
      tokens.assign(tokens_size_t.begin(), tokens_size_t.end());
    }
    int seq_len = tokens.size();
    float cum_logprob = result.scores[0] * std::pow(seq_len, options.length_penalty);
    float avg_logprob = cum_logprob / (seq_len + 1);

    std::string text = tokenizer.decode(tokens);
    float compression_ratio = get_compression_ratio(text);

    auto current_result = std::make_tuple(tokens, avg_logprob, temperature, compression_ratio);
    all_results.push_back(current_result);

    bool needs_fallback = false;

    if (options.compression_ratio_threshold.has_value() &&
        compression_ratio > options.compression_ratio_threshold.value()) {
      needs_fallback = true;
      logger.debug("Compression ratio threshold not met at temperature %.1f (%.3f > %.3f)",
                   temperature, compression_ratio, options.compression_ratio_threshold.value());
    } else {
      below_cr_threshold_results.push_back(current_result);
    }

    if (options.log_prob_threshold.has_value() &&
        avg_logprob < options.log_prob_threshold.value()) {
      needs_fallback = true;
      logger.debug("Log probability threshold not met at temperature %.1f (%.3f < %.3f)",
                   temperature, avg_logprob, options.log_prob_threshold.value());
    }

    if (options.no_speech_threshold.has_value() &&
        result.no_speech_prob > options.no_speech_threshold.value() &&
        options.log_prob_threshold.has_value() &&
        avg_logprob < options.log_prob_threshold.value()) {
      needs_fallback = false;
    }

    if (!needs_fallback) {
      return current_result;
    }

    // Update best result
    if (avg_logprob > best_avg_logprob) {
      best_result = result;
      best_avg_logprob = avg_logprob;
      best_temperature = temperature;
      best_compression_ratio = compression_ratio;
    }
  }

  // All temperatures failed: return best result
  if (below_cr_threshold_results.empty()) {
    std::vector<int> tokens;
    if (!best_result.sequences_ids.empty() && !best_result.sequences_ids[0].empty()) {
      const auto &tokens_size_t = best_result.sequences_ids[0];
      tokens.assign(tokens_size_t.begin(), tokens_size_t.end());
    }
    return std::make_tuple(tokens, best_avg_logprob, best_temperature, best_compression_ratio);
  }

  auto best_it = std::max_element(
      below_cr_threshold_results.begin(), below_cr_threshold_results.end(),
      [](const auto &a, const auto &b) { return std::get<1>(a) < std::get<1>(b); }
  );
  return *best_it;
}

std::vector<int> WhisperModel::get_prompt(
    Tokenizer &tokenizer,
    const std::vector<int> &previous_tokens,
    bool without_timestamps,
    std::optional<std::string> prefix,
    std::optional<std::string> hotwords
) {
  std::vector<int> prompt;

  if (!previous_tokens.empty() || (hotwords.has_value() && !prefix.has_value())) {
    prompt.push_back(tokenizer.get_sot_prev());

    if (hotwords.has_value() && !prefix.has_value()) {
      std::string hw = " " + hotwords.value();
      std::vector<int> hotwords_tokens = tokenizer.encode(hw);
      if (hotwords_tokens.size() >= max_length / 2) {
        hotwords_tokens.resize(max_length / 2 - 1);
      }
      prompt.insert(prompt.end(), hotwords_tokens.begin(), hotwords_tokens.end());
    }

    if (!previous_tokens.empty()) {
      size_t start_idx = std::max(0, static_cast<int>(previous_tokens.size()) - max_length / 2 + 1);
      prompt.insert(prompt.end(), previous_tokens.begin() + start_idx, previous_tokens.end());
    }
  }

  prompt.insert(prompt.end(), tokenizer.get_sot_sequence().begin(), tokenizer.get_sot_sequence().end());

  if (without_timestamps) {
    prompt.push_back(tokenizer.get_no_timestamps());
  }

  if (prefix.has_value()) {
    std::string pre = " " + prefix.value();
    std::vector<int> prefix_tokens = tokenizer.encode(pre);
    if (prefix_tokens.size() >= max_length / 2) {
      prefix_tokens.resize(max_length / 2 - 1);
    }
    if (!without_timestamps) {
      prompt.push_back(tokenizer.get_timestamp_begin());
    }
    prompt.insert(prompt.end(), prefix_tokens.begin(), prefix_tokens.end());
  }

  return prompt;
}

float WhisperModel::add_word_timestamps(
    std::vector<std::vector<std::map<std::string, std::any>>> &segments,
    Tokenizer &tokenizer,
    const ctranslate2::StorageView &encoder_output,
    int num_frames,
    const std::string &prepend_punctuations,
    const std::string &append_punctuations,
    float last_speech_timestamp
) {
  if (segments.empty()) return last_speech_timestamp;

  std::vector<std::vector<int>> text_tokens;
  std::vector<std::vector<std::vector<int>>> text_tokens_per_segment;

  for (auto &segment: segments) {
    std::vector<std::vector<int>> segment_tokens;
    for (auto &subsegment: segment) {
      std::vector<int> filtered_tokens;
      auto tokens = std::any_cast<std::vector<int>>(subsegment["tokens"]);
      std::copy_if(tokens.begin(), tokens.end(), std::back_inserter(filtered_tokens),
                   [&](int t) { return t < tokenizer.get_eot(); });
      segment_tokens.push_back(filtered_tokens);
    }
    std::vector<int> flattened;
    for (auto &tvec: segment_tokens)
      flattened.insert(flattened.end(), tvec.begin(), tvec.end());
    text_tokens.push_back(flattened);
    text_tokens_per_segment.push_back(segment_tokens);
  }

  auto alignments = find_alignment(tokenizer, text_tokens, encoder_output, num_frames);

  std::vector<std::pair<float, float>> median_max_durations;
  for (auto &alignment: alignments) {
    std::vector<float> word_durations;
    for (auto &word: alignment) {
      float duration =
          std::any_cast<float>(word.at("end")) - std::any_cast<float>(word.at("start"));
      if (duration > 0) word_durations.push_back(duration);
    }

    float median_duration = 0.0f;
    if (!word_durations.empty()) {
      size_t mid = word_durations.size() / 2;
      std::nth_element(word_durations.begin(), word_durations.begin() + mid, word_durations.end());
      median_duration = word_durations[mid];
    }
    median_duration = std::min(0.7f, median_duration);
    float max_duration = median_duration * 2.0f;
    median_max_durations.push_back({median_duration, max_duration});

    // merge_punctuations(alignment, prepend_punctuations, append_punctuations);
    // TODO: Fix type mismatch - alignment is vector<map<string,any>> but function expects vector<pair<string,float>>
  }

  for (size_t segment_idx = 0; segment_idx < segments.size(); ++segment_idx) {
    auto &segment = segments[segment_idx];
    size_t word_index = 0;
    float time_offset = std::any_cast<int>(segment[0]["seek"]) / frames_per_second;
    auto [median_duration, max_duration] = median_max_durations[segment_idx];

    for (size_t subsegment_idx = 0; subsegment_idx < segment.size(); ++subsegment_idx) {
      auto &subsegment = segment[subsegment_idx];
      int saved_tokens = 0;
      std::vector<std::map<std::string, std::any>> words;

      while (word_index < alignments[segment_idx].size() &&
             saved_tokens < text_tokens_per_segment[segment_idx][subsegment_idx].size()) {
        auto &timing = alignments[segment_idx][word_index];
        if (timing.count("word") && !std::any_cast<std::string>(timing["word"]).empty()) {
          words.push_back({
                              {"word",        timing["word"]},
                              {"start",       std::round(
                                  (time_offset + std::any_cast<float>(timing["start"])) * 100) /
                                              100},
                              {"end",         std::round(
                                  (time_offset + std::any_cast<float>(timing["end"])) * 100) / 100},
                              {"probability", timing["probability"]}
                          });
        }
        auto tokens = std::any_cast<std::vector<int>>(timing["tokens"]);
        saved_tokens += static_cast<int>(tokens.size());
        word_index++;
      }
      subsegment["words"] = words;
      if (!words.empty()) last_speech_timestamp = std::any_cast<float>(words.back().at("end"));
    }
  }

  return last_speech_timestamp;
}

std::vector<std::vector<std::map<std::string, std::any>>>
WhisperModel::find_alignment(
    Tokenizer &tokenizer,
    const std::vector<std::vector<int>> &text_tokens,
    const ctranslate2::StorageView &encoder_output,
    int num_frames,
    int median_filter_width
) {
  std::vector<std::vector<std::map<std::string, std::any>>> return_list;
  if (text_tokens.empty()) return return_list;

  // Convert vector<int> to vector<size_t> for CTranslate2 API
  auto sot_sequence_int = tokenizer.get_sot_sequence();
  std::vector<size_t> sot_sequence(sot_sequence_int.begin(), sot_sequence_int.end());

  // Convert text_tokens from vector<vector<int>> to vector<vector<size_t>>
  std::vector<std::vector<size_t>> text_tokens_size_t;
  for (const auto& token_vec : text_tokens) {
    std::vector<size_t> converted_tokens(token_vec.begin(), token_vec.end());
    text_tokens_size_t.push_back(converted_tokens);
  }

  // Create num_frames vector - one entry per text sequence
  std::vector<size_t> num_frames_vec(text_tokens_size_t.size(), static_cast<size_t>(num_frames));

  auto results = model->align(encoder_output, sot_sequence, text_tokens_size_t, num_frames_vec,
                             median_filter_width);

  for (size_t i = 0; i < results.size(); ++i) {
    const auto &result = results[i];
    const auto &tokens = text_tokens[i];
    auto [words, word_tokens] = tokenizer.split_to_word_tokens(tokens);
    if (word_tokens.size() <= 1) {
      return_list.push_back({});
      continue;
    }

    // Construct alignment
    std::vector<std::map<std::string, std::any>> alignment_list;
    for (size_t j = 0; j < words.size(); ++j) {
      alignment_list.push_back({
                                   {"word",        words[j]},
                                   {"tokens",      word_tokens[j]},
                                   {"start",       0.0f},  // placeholder, compute from result
                                   {"end",         0.0f},    // placeholder, compute from result
                                   {"probability", 1.0f}  // placeholder
                               });
    }
    return_list.push_back(alignment_list);
  }

  return return_list;
}

std::tuple<std::string, float, std::vector<std::pair<std::string, float>>>
WhisperModel::detect_language(
    const std::vector<float> *audio,
    const std::vector<std::vector<float>> *features,
    int language_detection_segments,
    float language_detection_threshold
) {
  assert(audio != nullptr || features != nullptr);

  std::vector<std::vector<float>> input_features;

  if (audio != nullptr) {
    std::vector<float> processed_audio = *audio;

    size_t n_samples = feature_extractor.n_samples;
    if (processed_audio.size() > static_cast<size_t>(language_detection_segments * n_samples)) {
      processed_audio.resize(language_detection_segments * n_samples);
    }

    input_features = feature_extractor.extract(processed_audio);
  } else if (features != nullptr) {
    input_features = *features;
  }

  size_t max_frames = feature_extractor.nb_max_frames();
  if (input_features[0].size() > static_cast<size_t>(language_detection_segments * max_frames)) {
    for (auto &row: input_features)
      row.resize(language_detection_segments * max_frames);
  }

  std::map<std::string, std::vector<float>> detected_language_info;
  std::vector<std::pair<std::string, float>> all_language_probs;
  std::string language;
  float language_probability = 0.0f;

  for (size_t i = 0; i < input_features[0].size(); i += max_frames) {
    std::vector<std::vector<float>> segment_features;
    size_t end_idx = std::min(i + max_frames, input_features[0].size());

    for (auto &row: input_features) {
      std::vector<float> segment_row(row.begin() + i, row.begin() + end_idx);
      segment_features.push_back(segment_row);
    }

    auto encoder_output = encode(pad_or_trim(segment_features));
    auto future_results = model->detect_language(encoder_output);
    auto results = future_results[0].get(); // Get result from future

    // strip markers from token
    all_language_probs.clear();
    for (auto &token_prob: results) {
      std::string token = token_prob.first;
      float prob = token_prob.second;
      if (token.size() > 4) // remove first 2 and last 2 chars
        token = token.substr(2, token.size() - 4);
      all_language_probs.emplace_back(token, prob);
    }

    if (!all_language_probs.empty()) {
      language = all_language_probs[0].first;
      language_probability = all_language_probs[0].second;
      if (language_probability > language_detection_threshold) break;
      detected_language_info[language].push_back(language_probability);
    }
  }

  if (language_probability <= language_detection_threshold && !detected_language_info.empty()) {
    // majority vote
    size_t max_count = 0;
    for (auto &kv: detected_language_info) {
      if (kv.second.size() > max_count) {
        max_count = kv.second.size();
        language = kv.first;
        language_probability = *std::max_element(kv.second.begin(), kv.second.end());
      }
    }
  }

  return {language, language_probability, all_language_probs};
}

// Helper function implementations

std::vector<std::vector<float>>
slice_features(const std::vector<std::vector<float>> &features, int start, int length) {
  // TODO: implement feature slicing
  return {};
}

std::vector<std::vector<float>>
pad_or_trim(const std::vector<std::vector<float>> &segment) {
  // TODO: implement padding/trimming
  return segment;
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
  std::vector<unsigned char> compressed(text.size() * 2);
  uLongf compressed_size = compressed.size();
  int res = compress(compressed.data(), &compressed_size,
                     reinterpret_cast<const unsigned char *>(text.data()), text.size());
  if (res != Z_OK) return 1.0f;
  return static_cast<float>(text.size()) / static_cast<float>(compressed_size);
}

// SpeechTimestampsMap helper class
class SpeechTimestampsMap {
public:
  SpeechTimestampsMap(const std::vector<std::map<std::string, float>> &speech_chunks,
                      int sampling_rate) {
    // TODO: implement constructor
  }

  int get_chunk_index(float t) const {
    // TODO: return chunk index containing time t
    return 0;
  }

  float get_original_time(float t, int chunk_index = -1, bool is_end = false) const {
    // TODO: map to original audio time
    return t;
  }
};

std::vector<Segment> restore_speech_timestamps(
    std::vector<Segment> segments,
    const std::vector<std::map<std::string, float>> &speech_chunks,
    int sampling_rate
) {
  SpeechTimestampsMap ts_map(speech_chunks, sampling_rate);

  for (auto &segment: segments) {
    if (segment.words.has_value() && !segment.words.value().empty()) {
      std::vector<Word> words;
      for (auto &word: segment.words.value()) {
        float middle = (word.start + word.end) / 2.0f;
        int chunk_index = ts_map.get_chunk_index(middle);
        word.start = ts_map.get_original_time(word.start, chunk_index);
        word.end = ts_map.get_original_time(word.end, chunk_index);
        words.push_back(word);
      }
      segment.start = words.front().start;
      segment.end = words.back().end;
      segment.words = words;
    } else {
      segment.start = ts_map.get_original_time(segment.start);
      segment.end = ts_map.get_original_time(segment.end, -1, true);
    }
  }
  return segments;
}
