
#include "transcribe.h"
#include "utils.h"
#include "tokenizer.h"
#include <ctranslate2/models/whisper.h>
#include <ctranslate2/storage_view.h>
#include <string>
#include <memory>
#include <filesystem>
#include <iostream>
#include <fstream>  // Add missing fstream header
#include <optional>
#include <vector>
#include <map>
#include <tuple>
#include <optional>
#include <cmath>
#include <algorithm>
#include <variant>

// Forward declarations of utility functions
std::vector<std::vector<float>> slice_features(const std::vector<std::vector<float>>& features, int start, int length);
ctranslate2::StorageView get_ctranslate2_storage_3d(const std::vector<std::vector<float>>& features);
float get_compression_ratio(const std::string& text);
std::vector<std::vector<float>> pad_or_trim(const std::vector<std::vector<float>>& segment);
#include <stdexcept>
#include <numeric>
#include <cassert>
#include <set>
#include <zlib.h>
#include <cstring>
#include <variant>
#include <utility>
#include <chrono>
#include "audio.h"
#include "feature_extractor.h"
#ifdef ANDROID
#include <android/log.h>
#else
#include <iostream>
// Define Android logging macros for non-Android builds
#define __android_log_print(level, tag, ...) printf(__VA_ARGS__); printf("\n")
#define ANDROID_LOG_DEBUG 0
#define ANDROID_LOG_ERROR 0
#endif

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

  // All branches lead to the same result, so we can simplify
  std::string model_path = model_size_or_path;
  model_path_ = model_path;  // Store in member variable for later use

  ctranslate2::ReplicaPoolConfig config;
  config.num_threads_per_replica = cpu_threads;   // map your params here

  // Initialize the CTranslate2 model with fallback compute types for Android
  std::vector<ctranslate2::ComputeType> compute_types = {
    ctranslate2::ComputeType::INT8,     // Try INT8 first (lowest memory, no SGEMM)
    ctranslate2::ComputeType::INT16,    // Try INT16 next
    ctranslate2::ComputeType::FLOAT32   // Fallback to FLOAT32
  };

  std::shared_ptr<ctranslate2::models::Whisper> created_model = nullptr;
  std::string last_error;

  for (auto compute_type : compute_types) {
    try {
      // std::cout << "Trying to initialize Whisper model with compute type: "
      //           << (int)compute_type << std::endl;

      created_model = std::make_shared<ctranslate2::models::Whisper>(
        model_path,
        ctranslate2::Device::CPU,
        compute_type,
        device_index,
        false,
        config
      );

      // std::cout << "Successfully initialized Whisper model with compute type: "
      //           << (int)compute_type << std::endl;
      break;

    } catch (const std::exception& e) {
      last_error = e.what();
      // std::cerr << "Failed to initialize with compute type " << (int)compute_type
      //           << ": " << e.what() << std::endl;
    }
  }

  if (!created_model) {
    throw std::runtime_error("Failed to initialize Whisper model with any compute type. Last error: " + last_error);
  }

  model = created_model;

  // Initialize tokenizer placeholder
  hf_tokenizer = nullptr;

  // -------------------
  // Tokenizer Handling
  // -------------------
  // In Python: tokenizers.Tokenizer.from_file("tokenizer.json")
  // In C++: you must implement or use a tokenizer wrapper
  std::string tokenizer_file = model_path + "/tokenizer.json";
  if (std::filesystem::exists(tokenizer_file)) {
    // Check vocabulary file and log token count
    std::string vocab_file = model_path + "/vocabulary.json";
    std::ifstream vocab_stream(vocab_file);
    if (vocab_stream.is_open()) {
      ctranslate2::Vocabulary temp_vocabulary = ctranslate2::Vocabulary::from_json_file(vocab_stream);
      vocab_stream.close();
      std::cout << "Loading HuggingFace tokenizer format with " << temp_vocabulary.size() << " tokens" << std::endl;
      std::cout << "Loaded " << temp_vocabulary.size() << " tokens from vocabulary file" << std::endl;
    }
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
  return {"ar"};
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

  // Log feature statistics
  if (!features.empty() && !features[0].empty()) {
    float min_val = features[0][0], max_val = features[0][0];
    double sum = 0.0;
    size_t count = 0;

    for (const auto& row : features) {
      for (float val : row) {
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
        sum += val;
        count++;
      }
    }

    float mean = sum / count;
    double sq_sum = 0.0;
    for (const auto& row : features) {
      for (float val : row) {
        sq_sum += (val - mean) * (val - mean);
      }
    }
    float std_dev = std::sqrt(sq_sum / count);

    std::cout << "Features stats: min=" << min_val << ", max=" << max_val
              << ", mean=" << mean << ", std=" << std_dev << std::endl;

    // Print first 5x5 of features
    std::cout << "First 5x5 of features:" << std::endl;
    for (size_t i = 0; i < std::min(size_t(5), features.size()); ++i) {
      for (size_t j = 0; j < std::min(size_t(5), features[i].size()); ++j) {
        std::cout << features[i][j] << " ";
      }
      std::cout << std::endl;
    }
  }

  // Step 4: Language detection - follows Python logic exactly
  std::string detected_language;
  float language_probability = 1.0f;
  std::vector<std::pair<std::string, float>> all_language_probs;

  if (!language.has_value()) {
    if (!model->is_multilingual()) {
      detected_language = "ar";
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
    if (!model->is_multilingual() && language.value() != "ar") {
      std::cerr << "The current model is English-only but language parameter is set to '" << language.value() << "'; using 'en' instead." << std::endl;
      detected_language = "en";
    } else {
      detected_language = language.value();
    }
    language_probability = 1;
  }

  // Step 5: Initialize tokenizer with vocabulary from assets (Python line 949-954)
  // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "🔧 TOKENIZER INITIALIZATION DEBUG");
  // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Creating tokenizer with multilingual=%d, task=transcribe, language=%s",
  //                     model->is_multilingual(), detected_language.c_str());
  // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Model path received: '%s'", model_path_.c_str());

  // Load vocabulary from copied assets in internal storage
  std::string vocab_path = model_path_ + "/vocabulary.json";
  // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "📁 Constructed vocabulary path: '%s'", vocab_path.c_str());

  // Check if vocabulary file exists before trying to load it
  std::ifstream vocab_check(vocab_path);
  if (vocab_check.good()) {
    vocab_check.seekg(0, std::ios::end);
    size_t file_size = vocab_check.tellg();
    vocab_check.close();
    // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "✅ Vocabulary file found! Size: %zu bytes", file_size);
  } else {
    __android_log_print(ANDROID_LOG_ERROR, "#transcribe", "❌ Vocabulary file NOT FOUND at: %s", vocab_path.c_str());
    // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "📂 Trying to debug what files exist in model directory: %s", model_path_.c_str());

    // Try to check if model directory exists
    std::ifstream model_dir_test(model_path_ + "/config.json");
    if (model_dir_test.good()) {
      // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "✅ Model directory exists (found config.json)");
      model_dir_test.close();
    } else {
      __android_log_print(ANDROID_LOG_ERROR, "#transcribe", "❌ Model directory might not exist or be accessible");
    }

    // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "🔍 Model directory existence check completed");
  }

  // Load vocabulary from the model directory
  std::string vocab_file = model_path_ + "/vocabulary.json";

  std::ifstream vocab_stream(vocab_file);
  if (!vocab_stream.is_open()) {
    throw std::runtime_error("Failed to open vocabulary file: " + vocab_file);
  }

  ctranslate2::Vocabulary vocabulary = ctranslate2::Vocabulary::from_json_file(vocab_stream);
  vocab_stream.close();

  // Use the CTranslate2 vocabulary to create the tokenizer
  Tokenizer tokenizer(vocabulary, model->is_multilingual(), std::string("transcribe"), detected_language);

  // Debug tokenizer initialization
  // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "🧪 Testing tokenizer methods...");

  try {
    int sot = tokenizer.get_sot();
    int sot_prev = tokenizer.get_sot_prev();
    int eot = tokenizer.get_eot();
    int transcribe_token = tokenizer.get_transcribe();
    auto sot_seq = tokenizer.get_sot_sequence();

    // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "SOT token: %d", sot);
    // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "SOT_PREV token: %d", sot_prev);
    // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "EOT token: %d", eot);
    // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "TRANSCRIBE token: %d", transcribe_token);
    // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "SOT sequence length: %zu", sot_seq.size());

    // std::string sot_seq_str = "SOT sequence: ";
    // for (size_t i = 0; i < sot_seq.size(); ++i) {
    //   sot_seq_str += std::to_string(sot_seq[i]);
    //   if (i < sot_seq.size() - 1) sot_seq_str += ", ";
    // }
    // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "%s", sot_seq_str.c_str());
  } catch (const std::exception& e) {
    // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Error testing tokenizer methods: %s", e.what());
  }

  // Step 6: Set up transcription options (Python line 956-989)
  TranscriptionOptions options;
  options.beam_size = 5;
  options.best_of = 5;
  options.patience = 1.0f;
  options.length_penalty = 1.0f;
  options.repetition_penalty = 1.1f;
  options.no_repeat_ngram_size = 3;
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
  options.word_timestamps = true;
  options.prepend_punctuations = "\"'¿([{-";
  options.append_punctuations = "\"\'.。，！？：\")}]、";
  options.multilingual = multilingual;
  options.max_new_tokens = std::nullopt;
  options.clip_timestamps = std::vector<float>{0}; // Default to "0"
  options.hallucination_silence_threshold = std::nullopt;
  options.hotwords = std::nullopt;

  // Step 7: Generate segments using the same logic as Python (line 991-993)
  std::vector<Segment> segments = generate_segments(features, tokenizer, options);

  // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "transcribe() received %zu segments from generate_segments", segments.size());
  // for (size_t i = 0; i < segments.size(); ++i) {
  //   __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Transcribe segment %zu: '%s'", i, segments[i].text.c_str());
  // }

  // Step 8: Create transcription info (Python line 998-1006)
  TranscriptionInfo info;
  info.language = detected_language;
  info.language_probability = language_probability;
  info.duration = duration;
  info.transcription_options = options;
  info.all_language_probs = all_language_probs;

  // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "🎯 TRANSCRIBE FUNCTION ABOUT TO RETURN!");
  // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Returning %zu segments, language: %s", segments.size(), info.language.c_str());

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
  // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "🔍 === ENTERING split_segments_by_timestamps ===");
  // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Input tokens count: %zu", tokens.size());
  // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "time_offset: %.2f, segment_size: %d, segment_duration: %.2f, seek: %d",
  //                    time_offset, segment_size, segment_duration, seek);

  // // Log first 20 tokens for debugging
  // std::string tokens_debug = "First 20 tokens: ";
  // for (size_t i = 0; i < std::min(tokens.size(), size_t(20)); ++i) {
  //   tokens_debug += std::to_string(tokens[i]) + " ";
  // }
  // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "%s", tokens_debug.c_str());

  // // Log tokenizer constants for reference
  // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Tokenizer constants - timestamp_begin: %d, eot: %d, sot: %d",
  //                    tokenizer.get_timestamp_begin(), tokenizer.get_eot(), tokenizer.get_sot());

  std::vector<Segment> current_segments;
  bool single_timestamp_ending = (tokens.size() >= 2 &&
              tokens[tokens.size() - 2] < tokenizer.get_timestamp_begin() &&
              tokens.back() >= tokenizer.get_timestamp_begin());

  // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "single_timestamp_ending: %s", single_timestamp_ending ? "true" : "false");
  // if (tokens.size() >= 2) {
  //   __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Last two tokens: [%d, %d]",
  //                      tokens[tokens.size() - 2], tokens[tokens.size() - 1]);
  // }

  std::vector<int> consecutive_timestamps;
  for (size_t i = 1; i < tokens.size(); ++i) {
  if (tokens[i] >= tokenizer.get_timestamp_begin() && tokens[i - 1] >= tokenizer.get_timestamp_begin()) {
    consecutive_timestamps.push_back(static_cast<int>(i));
    // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Found consecutive timestamp at position %zu: [%d, %d]",
    //                    i, tokens[i-1], tokens[i]);
  }
  }

  // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "consecutive_timestamps count: %zu", consecutive_timestamps.size());

  if (!consecutive_timestamps.empty()) {
  // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Processing consecutive timestamps path...");
  std::vector<int> slices = consecutive_timestamps;
  if (single_timestamp_ending) slices.push_back(tokens.size());

  // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Slices count: %zu", slices.size());

  int last_slice = 0;
  for (int current_slice: slices) {
    // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Processing slice from %d to %d", last_slice, current_slice);
    std::vector<int> sliced_tokens(tokens.begin() + last_slice, tokens.begin() + static_cast<std::vector<int>::difference_type>(current_slice));

    // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Sliced tokens count: %zu", sliced_tokens.size());
    // if (!sliced_tokens.empty()) {
    //   std::string sliced_debug = "Sliced tokens: ";
    //   for (size_t i = 0; i < std::min(sliced_tokens.size(), size_t(10)); ++i) {
    //     sliced_debug += std::to_string(sliced_tokens[i]) + " ";
    //   }
    //   __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "%s", sliced_debug.c_str());
    // }

    float start_time =
    time_offset + (sliced_tokens.front() - tokenizer.get_timestamp_begin()) * static_cast<float>(time_precision);
    float end_time =
    time_offset + (sliced_tokens.back() - tokenizer.get_timestamp_begin()) * static_cast<float>(time_precision);

    // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Calculated times - start: %.2f, end: %.2f", start_time, end_time);

    Segment seg;
    seg.seek = seek;
    seg.start = start_time;
    seg.end = end_time;
    seg.tokens = sliced_tokens;
    current_segments.push_back(seg);
    // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Added segment with %zu tokens", sliced_tokens.size());
    last_slice = current_slice;
  }

  if (single_timestamp_ending) {
    seek += segment_size;
    // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Updated seek (single_timestamp_ending): %d", seek);
  } else {
    int last_timestamp_position = tokens[last_slice - 1] - tokenizer.get_timestamp_begin();
    seek += static_cast<int>(last_timestamp_position) * input_stride;
    // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Updated seek (normal): %d", seek);
  }
  } else {
  // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Processing no consecutive timestamps path...");
  float duration = segment_duration;
  std::vector<int> timestamps;
  for (int token: tokens) if (token >= tokenizer.get_timestamp_begin()) timestamps.push_back(token);

  // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Found %zu timestamp tokens", timestamps.size());
  // if (!timestamps.empty()) {
  //   std::string timestamp_debug = "Timestamp tokens: ";
  //   for (size_t i = 0; i < std::min(timestamps.size(), size_t(10)); ++i) {
  //     timestamp_debug += std::to_string(timestamps[i]) + " ";
  //   }
  //   __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "%s", timestamp_debug.c_str());
  // }

  if (!timestamps.empty() && timestamps.back() != tokenizer.get_timestamp_begin()) {
    duration = (timestamps.back() - tokenizer.get_timestamp_begin()) * static_cast<float>(time_precision);
    // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Calculated duration from timestamps: %.2f", duration);
  } else {
    // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Using segment_duration: %.2f", duration);
  }

  Segment seg;
  seg.seek = seek;
  seg.start = time_offset;
  seg.end = time_offset + duration;
  seg.tokens = tokens;
  current_segments.push_back(seg);
  // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Added single segment with %zu tokens, start: %.2f, end: %.2f",
  //                    tokens.size(), seg.start, seg.end);
  seek += segment_size;
  // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Updated seek: %d", seek);
  }

  // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "🎯 === EXITING split_segments_by_timestamps ===");
  // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Returning %zu segments, seek: %d, single_timestamp_ending: %s",
  //                    current_segments.size(), seek, single_timestamp_ending ? "true" : "false");

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
    // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "previous_tokens.size(): %zu", previous_tokens.size());

    // Encode segment if needed (Python line 1175-1176)
    // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Checking if encoding needed: seek=%d, encoder_output.empty()=%d",
    //                     seek, encoder_output.empty());
    if (seek > 0 || encoder_output.empty()) {
      // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Encoding segment features...");
      encoder_output = encode(segment_features);
      // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Segment encoding completed");
    } else {
      // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Reusing existing encoder_output");
    }

    // Language detection per segment if multilingual (Python line 1178-1184)
    if (options.multilingual && model->is_multilingual()) {
      auto results_future = model->detect_language(encoder_output);
      auto results = results_future[0].get(); // Get result from first future in vector
      if (!results.empty()) {
        std::string language_token = results[0].first;
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

    // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "get_prompt returned prompt.size(): %zu", prompt.size());

    // Generate with fallback (Python line 1194-1199)
    // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "About to call generate_with_fallback");
    // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Prompt length: %zu", prompt.size());

    // std::string prompt_str = "Full prompt tokens: ";
    // for (size_t i = 0; i < prompt.size(); ++i) {
    //   prompt_str += std::to_string(prompt[i]);
    //   if (i < prompt.size() - 1) prompt_str += ", ";
    // }
    // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "%s", prompt_str.c_str());

    // Debug: Try to decode the prompt using our tokenizer to see what it looks like
    // std::string decoded_prompt = tokenizer.decode(prompt);
    // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Decoded prompt: '%s'", decoded_prompt.c_str());

    // Debug: Check specific token decoding
    // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Token 50257 decodes to: '%s'", tokenizer.decode({50257}).c_str());

    auto [result, avg_logprob, temperature, compression_ratio] = generate_with_fallback(
      encoder_output, prompt, tokenizer, options
    );

    // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "generate_with_fallback completed successfully");
    // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Generated %zu tokens", result.size());

    // Debug: Show first few generated tokens
    // if (!result.empty()) {
    //   std::string tokens_str = "Generated tokens: ";
    //   for (size_t i = 0; i < std::min(result.size(), size_t(10)); ++i) {
    //     tokens_str += std::to_string(result[i]) + " ";
    //   }
    //   __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "%s", tokens_str.c_str());

    //   // Try to decode the result
    //   std::string decoded_result = tokenizer.decode(result);
    //   __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Decoded result: '%s'", decoded_result.c_str());
    // }

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
    // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Processing %zu segments", current_segments.size());
    for (auto& segment : current_segments) {
      // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "About to decode segment with %zu tokens...", segment.tokens.size());

      // Log the tokens before decoding
      // std::string tokens_debug = "Segment tokens: ";
      // for (size_t i = 0; i < std::min(segment.tokens.size(), size_t(20)); ++i) {
      //   tokens_debug += std::to_string(segment.tokens[i]) + " ";
      // }
      // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "%s", tokens_debug.c_str());

      std::string text = tokenizer.decode(segment.tokens);
      // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "✅ Segment decode completed! Text: '%s' (tokens: %zu)", text.c_str(), segment.tokens.size());

      // Print Arabic text to console for visibility
      if (!text.empty()) {
        std::cout << "🎯 GENERATED ARABIC TEXT: \"" << text << "\"" << std::endl;
        std::cout << "   Segment timing: " << segment.start << "s -> " << segment.end << "s" << std::endl;
        std::cout << "   Token count: " << segment.tokens.size() << std::endl;
      }

      if (segment.start == segment.end || text.empty()) {
        // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Skipping empty segment");
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

  // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "generate_segments completed with %zu segments", all_segments.size());
  // for (size_t i = 0; i < all_segments.size(); ++i) {
  //   __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Final segment %zu: '%s'", i, all_segments[i].text.c_str());
  // }

  return all_segments;
}

// --------------------------
// Encode features using the Whisper model
// --------------------------
ctranslate2::StorageView WhisperModel::encode(const std::vector<std::vector<float>> &features) {
  // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "=== ENTERING encode() ===");
  // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Features dimensions: %zu x %zu", features.size(),
  //                     features.empty() ? 0 : features[0].size());

  bool to_cpu = false; // Simplified for CPU-only build

  // CTranslate2 Whisper model expects 3D input: [batch_size, n_mels, n_frames]
  // Input features are 2D: [n_mels, n_frames], so we need to add batch dimension

  if (features.empty() || features[0].empty()) {
    __android_log_print(ANDROID_LOG_ERROR, "#transcribe", "encode() called with empty features!");
    throw std::runtime_error("Cannot encode empty features");
  }

  // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Creating 3D storage tensor...");
  // Create 3D tensor by adding batch dimension
  auto storage = get_ctranslate2_storage_3d(features);
  // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Storage created with shape: [%lld, %lld, %lld]",
  //                     (long long)storage.shape()[0], (long long)storage.shape()[1], (long long)storage.shape()[2]);

  // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Calling model->encode()...");
  try {
    auto future = model->encode(storage, to_cpu);
    // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "model->encode() returned future, getting result...");

    auto result = future.get();
    // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "encode() completed successfully!");
    // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "=== EXITING encode() ===");
    return result;

  } catch (const std::exception& e) {
    __android_log_print(ANDROID_LOG_ERROR, "#transcribe", "EXCEPTION in model->encode(): %s", e.what());
    throw;
  }
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
  // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "=== ENTERING generate_with_fallback ===");
  // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Encoder output shape: [%lld, %lld, %lld]",
  //                     (long long)encoder_output.shape()[0], (long long)encoder_output.shape()[1], (long long)encoder_output.shape()[2]);
  // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Prompt size: %zu", prompt.size());
  // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Temperature options count: %zu", options.temperatures.size());

  // Follow Python implementation from line 1388-1516
  std::tuple<std::vector<int>, float, float, float> decode_result;
  std::vector<std::tuple<std::vector<int>, float, float, float>> all_results;
  std::vector<std::tuple<std::vector<int>, float, float, float>> below_cr_threshold_results;

  // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Calculating max_initial_timestamp_index...");
  int max_initial_timestamp_index = static_cast<int>(
    std::round(options.max_initial_timestamp / time_precision)
  );
  // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "max_initial_timestamp_index: %d", max_initial_timestamp_index);

  // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Calculating max_length...");
  // OPTIMIZATION: Reduce max_length for faster generation
  int max_length = options.max_new_tokens.has_value() ?
                   prompt.size() + options.max_new_tokens.value() :
                   256; // Reduced from 448 to 256 for Arabic speech
  // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "max_length: %d (OPTIMIZED), this->max_length: %d", max_length, this->max_length);

  if (max_length > this->max_length) {
    throw std::runtime_error("Prompt + max_new_tokens exceeds Whisper max_length");
  }

  // Iterate through temperatures (Python line 1418)
  // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Starting temperature loop...");

  for (size_t temp_idx = 0; temp_idx < options.temperatures.size(); ++temp_idx) {
    float temperature = options.temperatures[temp_idx];
    // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "=== Temperature iteration %zu/%zu: %.2f ===",
    //                     temp_idx + 1, options.temperatures.size(), temperature);

    // Configure generation options based on temperature (Python line 1419-1430)
    // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Configuring whisper_options...");
    ctranslate2::models::WhisperOptions whisper_options;

    // Use proper beam search like Python faster-whisper
    whisper_options.beam_size = options.beam_size;  // Use configured beam size (5)
    whisper_options.num_hypotheses = 1;  // Single best hypothesis
    if (temperature == 0.0f) {
      // Greedy search - no sampling
      whisper_options.sampling_topk = 1;  // Greedy
      whisper_options.sampling_temperature = 1.0f;  // No effect in greedy
    } else {
      // Sampling with temperature
      whisper_options.sampling_topk = 0;  // No top-k restriction
      whisper_options.sampling_temperature = temperature;  // Use sampling temperature
    }

    whisper_options.length_penalty = options.length_penalty;
    whisper_options.repetition_penalty = options.repetition_penalty;
    whisper_options.no_repeat_ngram_size = options.no_repeat_ngram_size;
    whisper_options.max_length = max_length;
    whisper_options.suppress_blank = options.suppress_blank;
    whisper_options.max_initial_timestamp_index = max_initial_timestamp_index;

    if (options.suppress_tokens.has_value()) {
      // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Setting suppress_tokens...");
      std::vector<int> suppress_tokens_int;
      for (int token : options.suppress_tokens.value()) {
        suppress_tokens_int.push_back(token);
      }
      whisper_options.suppress_tokens = suppress_tokens_int;
      // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "suppress_tokens set with %zu tokens", suppress_tokens_int.size());
    }

    // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Converting prompt to size_t...");
    // Convert prompt to size_t for CTranslate2 (Python line 1432-1445)
    std::vector<size_t> prompt_size_t(prompt.begin(), prompt.end());
    // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Prompt converted: %zu tokens", prompt_size_t.size());

    // CRITICAL DEBUG: Log the actual prompt being sent to the model
    // std::string full_prompt_debug = "🚨 FULL PROMPT being sent to model: ";
    // for (size_t i = 0; i < prompt_size_t.size(); ++i) {
    //   full_prompt_debug += std::to_string(prompt_size_t[i]) + " ";
    // }
    // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "%s", full_prompt_debug.c_str());

    // Verify Arabic SOT sequence
    // if (prompt_size_t.size() >= 3) {
    //   __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "🔍 Checking Arabic SOT sequence:");
    //   __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "   Token 0 (should be SOT 50258): %zu", prompt_size_t[0]);
    //   __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "   Token 1 (should be Arabic lang 50272): %zu", prompt_size_t[1]);
    //   __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "   Token 2 (should be transcribe 50359): %zu", prompt_size_t[2]);
    // }

    // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "About to call model->generate() - THIS IS THE CRITICAL CALL");
    // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "WhisperOptions configured: beam_size=%zu, max_length=%zu, temperature=%.2f",
    //                     (size_t)whisper_options.beam_size, (size_t)whisper_options.max_length, temperature);

    try {
      // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Calling model->generate()...");
      auto result_futures = model->generate(encoder_output, {prompt_size_t}, whisper_options);
      // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "model->generate() returned futures, getting result...");

      // Add timeout logging to track how long result.get() takes
      // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "About to call result_futures[0].get() - monitoring for hang...");

      auto start_time = std::chrono::steady_clock::now();
      auto result = result_futures[0].get();
      auto end_time = std::chrono::steady_clock::now();

      auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
      // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "result.get() completed in %lld seconds!", (long long)duration.count());

      // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Result sequences count: %zu", result.sequences_ids.size());
      // if (!result.sequences_ids.empty()) {
      //   __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "First sequence length: %zu", result.sequences_ids[0].size());
      // }
      // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Result scores count: %zu", result.scores.size());

      // Extract tokens and calculate metrics (Python line 1447-1455)
      // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Extracting tokens from result...");
      std::vector<int> tokens;
      if (!result.sequences_ids.empty() && !result.sequences_ids[0].empty()) {
        const auto &tokens_size_t = result.sequences_ids[0];
        tokens.assign(tokens_size_t.begin(), tokens_size_t.end());
        // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Extracted %zu tokens", tokens.size());
      } else {
        // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "No tokens in result sequences!");
      }
      // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Before seq_len = tokens.size()");
      int seq_len = tokens.size();
      // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "After seq_len = tokens.size()");

      // Check if scores array is available
      float cum_logprob = 0.0f;
      float avg_logprob = 0.0f;

      if (!result.scores.empty()) {
        // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Calculating scores - result.scores[0]: %.4f", result.scores[0]);
        cum_logprob = result.scores[0] * std::pow(seq_len, options.length_penalty);
        // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "After options.length_penalty calculation");
        avg_logprob = cum_logprob / (seq_len + 1);
        // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Calculated avg_logprob: %.4f", avg_logprob);
      } else {
        // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "⚠️ result.scores is EMPTY! Using default values: cum_logprob=0.0, avg_logprob=0.0");
        // Use default values when scores are not available
        cum_logprob = 0.0f;
        avg_logprob = 0.0f;
      }

      // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "About to call tokenizer.decode() - THIS IS THE LIKELY BOTTLENECK");
      // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Decoding %zu tokens...", tokens.size());

      // Calculate compression ratio (Python line 1454-1455)
      // CRITICAL: This decode() call is likely where we're getting stuck
      std::string text = tokenizer.decode(tokens);

      // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "✅ tokenizer.decode() COMPLETED!");
      // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Generated text: '%s'", text.c_str());

      // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Calculating compression ratio...");
      float compression_ratio = get_compression_ratio(text);
      // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "✅ Compression ratio calculated: %.2f, avg_logprob: %.4f", compression_ratio, avg_logprob);

      decode_result = std::make_tuple(tokens, avg_logprob, temperature, compression_ratio);
      all_results.push_back(decode_result);

      bool needs_fallback = false;

      // Check compression ratio threshold (Python line 1467-1478)
      if (options.compression_ratio_threshold.has_value() &&
          compression_ratio > options.compression_ratio_threshold.value()) {
        // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Compression ratio %.2f > threshold %.2f, needs fallback",
        //                     compression_ratio, options.compression_ratio_threshold.value());
        needs_fallback = true;
      } else {
        below_cr_threshold_results.push_back(decode_result);
      }

      // Check log probability threshold (Python line 1480-1491)
      if (options.log_prob_threshold.has_value() &&
          avg_logprob < options.log_prob_threshold.value()) {
        // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "avg_logprob %.4f < threshold %.4f, needs fallback",
        //                     avg_logprob, options.log_prob_threshold.value());
        needs_fallback = true;
      }

      // Check no speech threshold (Python line 1493-1499)
      if (options.no_speech_threshold.has_value() &&
          result.no_speech_prob > options.no_speech_threshold.value() &&
          options.log_prob_threshold.has_value() &&
          avg_logprob < options.log_prob_threshold.value()) {
        // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "No speech detected, silence");
        needs_fallback = false; // silence
      }

      if (!needs_fallback) {
        // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Temperature %.2f successful, breaking loop", temperature);
        break; // Success, return this result
      } else {
        // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Temperature %.2f failed, trying next", temperature);
      }

    } catch (const std::exception& e) {
      __android_log_print(ANDROID_LOG_ERROR, "#transcribe", "EXCEPTION in model->generate(): %s", e.what());
      throw;
    }
  }

  // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Temperature loop completed");

  // All temperatures failed, select best result (Python line 1504-1515)
  // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Selecting best result from %zu below_cr_threshold and %zu all_results",
  //                     below_cr_threshold_results.size(), all_results.size());

  if (!below_cr_threshold_results.empty()) {
    // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Using best from below_cr_threshold_results");
    auto best_it = std::max_element(
      below_cr_threshold_results.begin(), below_cr_threshold_results.end(),
      [](const auto &a, const auto &b) { return std::get<1>(a) < std::get<1>(b); }
    );
    decode_result = *best_it;
  } else if (!all_results.empty()) {
    // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Using best from all_results");
    auto best_it = std::max_element(
      all_results.begin(), all_results.end(),
      [](const auto &a, const auto &b) { return std::get<1>(a) < std::get<1>(b); }
    );
    decode_result = *best_it;
  } else {
    // __android_log_print(ANDROID_LOG_ERROR, "#transcribe", "No results available! This should not happen");
  }

  // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "=== EXITING generate_with_fallback successfully ===");
  return decode_result;
}

std::vector<int> WhisperModel::get_prompt(
  Tokenizer &tokenizer,
  const std::vector<int> &previous_tokens,
  bool without_timestamps,
  std::optional<std::string> prefix,
  std::optional<std::string> hotwords
) {
  // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "get_prompt called with previous_tokens.size()=%zu, without_timestamps=%d",
  //                     previous_tokens.size(), without_timestamps);

  std::vector<int> prompt;

  if (!previous_tokens.empty() || (hotwords.has_value() && !prefix.has_value())) {
  // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Adding SOT_PREV token");
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

  // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Before adding SOT sequence, prompt.size()=%zu", prompt.size());

  auto sot_sequence = tokenizer.get_sot_sequence();
  // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "SOT sequence size: %zu", sot_sequence.size());

  prompt.insert(prompt.end(), sot_sequence.begin(), sot_sequence.end());

  // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "After adding SOT sequence, prompt.size()=%zu", prompt.size());

  // Debug: Log the prompt tokens to help diagnose the issue
  // std::string prompt_debug = "Generated prompt tokens: ";
  // for (size_t i = 0; i < std::min(prompt.size(), size_t(10)); ++i) {
  //   prompt_debug += std::to_string(prompt[i]) + " ";
  // }
  // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "%s", prompt_debug.c_str());

  // Debug: Check if SOT token is in the prompt (reuse existing sot_sequence)
  // std::string sot_debug = "SOT sequence: ";
  // for (int token : sot_sequence) {
  //   sot_debug += std::to_string(token) + " ";
  // }
  // __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "%s", sot_debug.c_str());

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
      auto timing_tokens = std::any_cast<std::vector<int>>(timing["tokens"]);
      saved_tokens += static_cast<int>(timing_tokens.size());
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
  std::vector<size_t> sot_sequence_size_t(sot_sequence_int.begin(), sot_sequence_int.end());

  // Convert text_tokens from vector<vector<int>> to vector<vector<size_t>>
  std::vector<std::vector<size_t>> text_tokens_size_t;
  for (const auto& token_vec : text_tokens) {
  std::vector<size_t> converted_tokens(token_vec.begin(), token_vec.end());
  text_tokens_size_t.push_back(converted_tokens);
  }

  // Create num_frames vector - one entry per text sequence
  std::vector<size_t> num_frames_vec(text_tokens_size_t.size(), static_cast<size_t>(num_frames));

  auto results_future = model->align(encoder_output, sot_sequence_size_t, text_tokens_size_t, num_frames_vec,
         static_cast<long>(median_filter_width));

  // Process each future in the vector
  std::vector<ctranslate2::models::WhisperAlignmentResult> results;
  for (auto& future : results_future) {
    results.push_back(future.get());
  }

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

ctranslate2::StorageView get_ctranslate2_storage_3d(const std::vector<std::vector<float>> &features) {
  // Create 3D tensor with batch dimension: [batch_size=1, n_mels, n_frames]
  // Input features are 2D: [n_mels, n_frames]

  if (features.empty() || features[0].empty()) {
    throw std::runtime_error("Cannot create storage from empty features");
  }

  size_t n_mels = features.size();
  size_t n_frames = features[0].size();
  size_t batch_size = 1;

  // Flatten into contiguous memory with batch dimension
  std::vector<float> contiguous;
  contiguous.reserve(batch_size * n_mels * n_frames);

  // Add batch dimension by copying features once
  for (const auto &row : features) {
    contiguous.insert(contiguous.end(), row.begin(), row.end());
  }

  // Create 3D shape: [batch_size, n_mels, n_frames]
  ctranslate2::Shape shape = {
    static_cast<long>(batch_size),
    static_cast<long>(n_mels),
    static_cast<long>(n_frames)
  };

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
