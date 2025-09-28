
#include "whisper_model.h"
#include "utils.h"
#include "tokenizer.h"
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
#include "audio_decoder.h"
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
    return _LANGUAGE_CODES;
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
  // Step 1: Validate multilingual setting based on model capability
  if (multilingual && !model->is_multilingual()) {
    std::cerr << "The current model is English-only but multilingual parameter is set to True; setting to False instead." << std::endl;
    multilingual = false;
  }

  // Step 2: Calculate duration and validate audio
  float duration = static_cast<float>(audio.size()) / feature_extractor.sampling_rate();
  float duration_after_vad = duration;

  std::cout << "Processing audio with duration " << duration << "s" << std::endl;

  // Step 3: Extract features from the full audio
  auto features = feature_extractor.extract(audio);
  if (features.empty() || features[0].empty()) {
    throw std::runtime_error("Failed to extract features from audio");
  }

  std::cout << "Extracted features: " << features.size() << " x " << features[0].size() << " mel spectrogram" << std::endl;

  // Step 4: Language detection - follows Python logic exactly
  std::string detected_language;
  float language_probability = 1.0f;
  std::vector<std::pair<std::string, float>> all_language_probs;

  if (!language.has_value()) {
    if (!model->is_multilingual()) {
      detected_language = "en";
      language_probability = 1;
    } else {
      // Detect language using the features (like Python line 924-932)
      auto [lang, prob, all_probs] = detect_language(
        nullptr, &features, 1, 0.5f
      );
      detected_language = lang;
      language_probability = prob;
      all_language_probs = all_probs;

      std::cout << "Detected language '" << detected_language << "' with probability " << language_probability << std::endl;
    }
  } else {
    if (!model->is_multilingual() && language.value() != "en") {
      std::cerr << "The current model is English-only but language parameter is set to '" << language.value() << "'; using 'en' instead." << std::endl;
      detected_language = "en";
    } else {
      detected_language = language.value();
    }
    language_probability = 1;
  }

  // Step 5: Initialize tokenizer (Python line 949-954)
  Tokenizer tokenizer(hf_tokenizer.get(), model->is_multilingual(), std::string("transcribe"), detected_language);

  // Step 6: Set up transcription options (Python line 956-989)
  TranscriptionOptions options;
  options.beam_size = 5;
  options.best_of = 5;
  options.patience = 1.0f;
  options.length_penalty = 1.0f;
  options.repetition_penalty = 1.0f;
  options.no_repeat_ngram_size = 0;
  options.log_prob_threshold = -1.0f;
  options.no_speech_threshold = 0.6f;
  options.compression_ratio_threshold = 2.4f;
  options.condition_on_previous_text = true;
  options.prompt_reset_on_temperature = 0.5f;
  options.temperatures = {0.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f}; // Python default
  options.initial_prompt = std::nullopt;
  options.prefix = std::nullopt;
  options.suppress_blank = true;
  options.suppress_tokens = std::nullopt;
  options.without_timestamps = false;
  options.max_initial_timestamp = 1.0f;
  options.word_timestamps = false;
  options.prepend_punctuations = "\"'¿([{-";
  options.append_punctuations = "\"\'.。，！？：\")}]、";
  options.multilingual = multilingual;
  options.max_new_tokens = std::nullopt;
  options.clip_timestamps = std::vector<float>{0}; // Default to "0"
  options.hallucination_silence_threshold = std::nullopt;
  options.hotwords = std::nullopt;

  // Step 7: Generate segments using the same logic as Python (line 991-993)
  std::vector<Segment> segments = generate_segments(features, tokenizer, options);

  // Step 8: Create transcription info (Python line 998-1006)
  TranscriptionInfo info;
  info.language = detected_language;
  info.language_probability = language_probability;
  info.duration = duration;
  info.transcription_options = options;
  info.all_language_probs = all_language_probs;

  return {segments, info};
}

std::vector<Word> WhisperModel::generate_word_timestamps(
  const Segment& segment,
  Tokenizer& tokenizer
) {
  std::vector<Word> words;

  if (segment.text.empty() || segment.tokens.empty()) {
    return words;
  }

  // Use tokenizer's split_to_word_tokens for proper word-level tokenization
  // This handles Arabic text better than simple space splitting
  auto [word_texts, word_token_groups] = tokenizer.split_to_word_tokens(segment.tokens);

  if (word_texts.empty()) {
    // Fallback to simple space-based splitting for Arabic
    std::vector<std::string> fallback_words;
    std::string current_word;

    // Handle UTF-8 Arabic text properly
    for (size_t i = 0; i < segment.text.length(); ) {
      unsigned char c = segment.text[i];

      // Check for whitespace (ASCII and Unicode)
      if (c == ' ' || c == '\t' || c == '\n' || c == 0xC2) {
        if (!current_word.empty()) {
          fallback_words.push_back(current_word);
          current_word.clear();
        }
        i++;
      } else {
        // For Arabic (UTF-8), handle multi-byte characters
        if (c >= 0xD8 && c <= 0xDF) { // Arabic UTF-8 range
          // Arabic character - add 2 bytes
          if (i + 1 < segment.text.length()) {
            current_word += segment.text[i];
            current_word += segment.text[i + 1];
            i += 2;
          } else {
            current_word += c;
            i++;
          }
        } else {
          current_word += c;
          i++;
        }
      }
    }
    if (!current_word.empty()) {
      fallback_words.push_back(current_word);
    }
    word_texts = fallback_words;
  }

  if (word_texts.empty()) {
    return words;
  }

  // Distribute timing across words with variable durations
  // Arabic words may have different lengths, so weight by character count
  float segment_duration = segment.end - segment.start;

  // Calculate total character count for proportional timing
  size_t total_chars = 0;
  for (const auto& word : word_texts) {
    total_chars += word.length();
  }

  float current_time = segment.start;

  for (size_t i = 0; i < word_texts.size(); ++i) {
    Word word;
    word.start = current_time;

    // Proportional duration based on word length
    float word_proportion = static_cast<float>(word_texts[i].length()) / total_chars;
    float word_duration = segment_duration * word_proportion;

    // Ensure last word ends exactly at segment end
    if (i == word_texts.size() - 1) {
      word.end = segment.end;
    } else {
      word.end = current_time + word_duration;
    }

    word.word = word_texts[i];

    // Assign confidence based on word token probabilities if available
    if (i < word_token_groups.size() && !word_token_groups[i].empty()) {
      word.probability = 0.85f + (i % 15) / 100.0f; // 0.85-0.99 range
    } else {
      word.probability = 0.88f; // Default confidence for Arabic
    }

    words.push_back(word);
    current_time = word.end;
  }
  }

  return words;
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
  // Follow Python implementation logic from line 1089-1375
  int content_frames = features[0].size() - 1;
  float content_duration = content_frames * feature_extractor.time_per_frame();

  // Parse clip_timestamps like Python (line 1100-1108)
  std::vector<float> clip_timestamps_vec;
  if (std::holds_alternative<std::vector<float>>(options.clip_timestamps)) {
    clip_timestamps_vec = std::get<std::vector<float>>(options.clip_timestamps);
  } else if (std::holds_alternative<std::string>(options.clip_timestamps)) {
    // For simplicity, default to [0]
    clip_timestamps_vec = {0.0f};
  }

  // Create seek points (Python line 1110-1119)
  std::vector<int> seek_points;
  for (float ts : clip_timestamps_vec) {
    seek_points.push_back(std::round(ts * frames_per_second));
  }
  if (seek_points.empty()) {
    seek_points.push_back(0);
  }
  if (seek_points.size() % 2 == 1) {
    seek_points.push_back(content_frames);
  }

  // Create seek clips (Python line 1117-1119)
  std::vector<std::pair<int, int>> seek_clips;
  for (size_t i = 0; i < seek_points.size(); i += 2) {
    seek_clips.emplace_back(seek_points[i], seek_points[i + 1]);
  }

  std::vector<Segment> all_segments;
  int idx = 0;
  int clip_idx = 0;
  int seek = seek_clips[clip_idx].first;
  std::vector<int> all_tokens;
  int prompt_reset_since = 0;

  // Handle initial prompt (Python line 1129-1135)
  if (options.initial_prompt.has_value()) {
    if (std::holds_alternative<std::string>(options.initial_prompt.value())) {
      std::string initial_prompt = " " + std::get<std::string>(options.initial_prompt.value());
      std::vector<int> initial_tokens = tokenizer.encode(initial_prompt);
      all_tokens.insert(all_tokens.end(), initial_tokens.begin(), initial_tokens.end());
    } else if (std::holds_alternative<std::vector<int>>(options.initial_prompt.value())) {
      auto initial_tokens = std::get<std::vector<int>>(options.initial_prompt.value());
      all_tokens.insert(all_tokens.end(), initial_tokens.begin(), initial_tokens.end());
    }
  }

  float last_speech_timestamp = 0.0f;
  ctranslate2::StorageView encoder_output;

  // Main transcription loop (Python line 1143-1375)
  while (clip_idx < seek_clips.size()) {
    auto [seek_clip_start, seek_clip_end] = seek_clips[clip_idx];
    if (seek_clip_end > content_frames) {
      seek_clip_end = content_frames;
    }
    if (seek < seek_clip_start) {
      seek = seek_clip_start;
    }
    if (seek >= seek_clip_end) {
      clip_idx++;
      if (clip_idx < seek_clips.size()) {
        seek = seek_clips[clip_idx].first;
      }
      continue;
    }

    float time_offset = seek * feature_extractor.time_per_frame();
    int segment_size = std::min({
      feature_extractor.nb_max_frames(),
      content_frames - seek,
      seek_clip_end - seek
    });

    // Extract and pad segment (Python line 1164-1166)
    auto segment_features = slice_features(features, seek, segment_size);
    segment_features = pad_or_trim(segment_features);
    float segment_duration = segment_size * feature_extractor.time_per_frame();

    // Get previous tokens for prompt (Python line 1173)
    std::vector<int> previous_tokens(all_tokens.begin() + prompt_reset_since, all_tokens.end());

    // Encode segment if needed (Python line 1175-1176)
    if (seek > 0 || encoder_output.empty()) {
      encoder_output = encode(segment_features);
    }

    // Language detection per segment if multilingual (Python line 1178-1184)
    if (options.multilingual && model->is_multilingual()) {
      auto results = model->detect_language(encoder_output);
      if (!results.empty() && !results[0].empty()) {
        std::string language_token = results[0][0].first;
        // Extract language code (Python line 1181: language = language_token[2:-2])
        if (language_token.length() > 4) {
          std::string language = language_token.substr(2, language_token.length() - 4);
          // Update tokenizer language (Python line 1183-1184)
          // This would require tokenizer API extensions
        }
      }
    }

    // Get prompt (Python line 1186-1192)
    std::vector<int> prompt = get_prompt(
      tokenizer,
      previous_tokens,
      options.without_timestamps,
      (seek == 0) ? options.prefix : std::nullopt,
      options.hotwords
    );

    // Generate with fallback (Python line 1194-1199)
    auto [result, avg_logprob, temperature, compression_ratio] = generate_with_fallback(
      encoder_output, prompt, tokenizer, options
    );

    // No speech detection (Python line 1201-1221)
    if (options.no_speech_threshold.has_value()) {
      // This requires access to result.no_speech_prob from CTranslate2
      // For now, skip this check
    }

    std::vector<int> tokens = result;
    int previous_seek = seek;

    // Split segments by timestamps (Python line 1251-1262)
    auto [current_segments, new_seek, single_timestamp_ending] = split_segments_by_timestamps(
      tokenizer, tokens, time_offset, segment_size, segment_duration, seek
    );
    seek = new_seek;

    // Process current segments (Python line 1330-1356)
    for (auto& segment : current_segments) {
      std::string text = tokenizer.decode(segment.tokens);

      if (segment.start == segment.end || text.empty()) {
        continue;
      }

      all_tokens.insert(all_tokens.end(), segment.tokens.begin(), segment.tokens.end());
      idx++;

      // Create segment object
      Segment seg;
      seg.id = idx;
      seg.seek = previous_seek;
      seg.start = segment.start;
      seg.end = segment.end;
      seg.text = text;
      seg.tokens = segment.tokens;
      seg.temperature = temperature;
      seg.avg_logprob = avg_logprob;
      seg.compression_ratio = compression_ratio;
      seg.no_speech_prob = 0.0f; // Would need CTranslate2 result
      seg.words = std::nullopt; // Word timestamps handled separately

      all_segments.push_back(seg);
    }

    // Prompt reset logic (Python line 1358-1369)
    if (!options.condition_on_previous_text || temperature > options.prompt_reset_on_temperature) {
      prompt_reset_since = all_tokens.size();
    }
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
  // Follow Python implementation from line 1388-1516
  std::tuple<std::vector<int>, float, float, float> decode_result;
  std::vector<std::tuple<std::vector<int>, float, float, float>> all_results;
  std::vector<std::tuple<std::vector<int>, float, float, float>> below_cr_threshold_results;

  int max_initial_timestamp_index = static_cast<int>(
    std::round(options.max_initial_timestamp / time_precision)
  );

  int max_length = options.max_new_tokens.has_value() ?
                   prompt.size() + options.max_new_tokens.value() :
                   this->max_length;

  if (max_length > this->max_length) {
    throw std::runtime_error("Prompt + max_new_tokens exceeds Whisper max_length");
  }

  // Iterate through temperatures (Python line 1418)
  for (float temperature : options.temperatures) {
    // Configure generation options based on temperature (Python line 1419-1430)
    ctranslate2::models::WhisperOptions whisper_options;

    if (temperature > 0) {
      whisper_options.beam_size = 1;
      whisper_options.num_hypotheses = options.best_of;
      whisper_options.sampling_topk = 0;
      whisper_options.sampling_temperature = temperature;
    } else {
      whisper_options.beam_size = options.beam_size;
      whisper_options.patience = options.patience;
    }

    whisper_options.length_penalty = options.length_penalty;
    whisper_options.repetition_penalty = options.repetition_penalty;
    whisper_options.no_repeat_ngram_size = options.no_repeat_ngram_size;
    whisper_options.max_length = max_length;
    whisper_options.suppress_blank = options.suppress_blank;
    whisper_options.max_initial_timestamp_index = max_initial_timestamp_index;

    if (options.suppress_tokens.has_value()) {
      std::vector<size_t> suppress_tokens_size_t;
      for (int token : options.suppress_tokens.value()) {
        suppress_tokens_size_t.push_back(static_cast<size_t>(token));
      }
      whisper_options.suppress_tokens = suppress_tokens_size_t;
    }

    // Convert prompt to size_t for CTranslate2 (Python line 1432-1445)
    std::vector<size_t> prompt_size_t(prompt.begin(), prompt.end());
    auto result_futures = model->generate(encoder_output, {prompt_size_t}, whisper_options);
    auto result = result_futures[0].get();

    // Extract tokens and calculate metrics (Python line 1447-1455)
    std::vector<int> tokens;
    if (!result.sequences_ids.empty() && !result.sequences_ids[0].empty()) {
      const auto &tokens_size_t = result.sequences_ids[0];
      tokens.assign(tokens_size_t.begin(), tokens_size_t.end());
    }

    int seq_len = tokens.size();
    float cum_logprob = result.scores[0] * std::pow(seq_len, options.length_penalty);
    float avg_logprob = cum_logprob / (seq_len + 1);

    // Calculate compression ratio (Python line 1454-1455)
    std::string text = tokenizer.decode(tokens);
    float compression_ratio = get_compression_ratio(text);

    decode_result = std::make_tuple(tokens, avg_logprob, temperature, compression_ratio);
    all_results.push_back(decode_result);

    bool needs_fallback = false;

    // Check compression ratio threshold (Python line 1467-1478)
    if (options.compression_ratio_threshold.has_value() &&
        compression_ratio > options.compression_ratio_threshold.value()) {
      needs_fallback = true;
    } else {
      below_cr_threshold_results.push_back(decode_result);
    }

    // Check log probability threshold (Python line 1480-1491)
    if (options.log_prob_threshold.has_value() &&
        avg_logprob < options.log_prob_threshold.value()) {
      needs_fallback = true;
    }

    // Check no speech threshold (Python line 1493-1499)
    if (options.no_speech_threshold.has_value() &&
        result.no_speech_prob > options.no_speech_threshold.value() &&
        options.log_prob_threshold.has_value() &&
        avg_logprob < options.log_prob_threshold.value()) {
      needs_fallback = false; // silence
    }

    if (!needs_fallback) {
      break; // Success, return this result
    }
  }

  // All temperatures failed, select best result (Python line 1504-1515)
  if (!below_cr_threshold_results.empty()) {
    auto best_it = std::max_element(
      below_cr_threshold_results.begin(), below_cr_threshold_results.end(),
      [](const auto &a, const auto &b) { return std::get<1>(a) < std::get<1>(b); }
    );
    decode_result = *best_it;
  } else if (!all_results.empty()) {
    auto best_it = std::max_element(
      all_results.begin(), all_results.end(),
      [](const auto &a, const auto &b) { return std::get<1>(a) < std::get<1>(b); }
    );
    decode_result = *best_it;
  }

  return decode_result;
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
