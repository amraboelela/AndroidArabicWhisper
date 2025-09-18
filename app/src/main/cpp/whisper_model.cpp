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
#include <stdexcept>
#include <numeric>
#include <cassert>
#include <set>
#include <zlib.h>        // for compression
#include <cstring>       // for memcpy
#include <variant>
#include <utility> // for std::pair

// ---------------- Word ----------------
struct Word {
  float start;
  float end;
  std::string word;
  float probability;

  // For compatibility with Python's asdict (optional)
  std::string to_string() const {
    return "{start: " + std::to_string(start) +
           ", end: " + std::to_string(end) +
           ", word: \"" + word + "\"" +
           ", probability: " + std::to_string(probability) + "}";
  }
};

// ---------------- Segment ----------------
struct Segment {
  int id;
  int seek;
  float start;
  float end;
  std::string text;
  std::vector<int> tokens;
  float avg_logprob;
  float compression_ratio;
  float no_speech_prob;
  std::optional<std::vector<Word>> words;  // optional field
  std::optional<float> temperature;        // optional field

  std::string to_string() const {
    std::string words_str = "[";
    if (words.has_value()) {
      for (const auto& w : words.value()) {
        words_str += w.to_string() + ", ";
      }
      if (!words.value().empty()) words_str.pop_back(), words_str.pop_back();
    }
    words_str += "]";

    return "{id: " + std::to_string(id) +
           ", seek: " + std::to_string(seek) +
           ", start: " + std::to_string(start) +
           ", end: " + std::to_string(end) +
           ", text: \"" + text + "\"" +
           ", avg_logprob: " + std::to_string(avg_logprob) +
           ", compression_ratio: " + std::to_string(compression_ratio) +
           ", no_speech_prob: " + std::to_string(no_speech_prob) +
           ", words: " + words_str +
           ", temperature: " + (temperature.has_value() ? std::to_string(temperature.value()) : "null") +
           "}";
  }
};

// Forward declaration of VadOptions
struct VadOptions;

// ---------------- TranscriptionOptions ----------------
struct TranscriptionOptions {
  int beam_size;
  int best_of;
  float patience;
  float length_penalty;
  float repetition_penalty;
  int no_repeat_ngram_size;

  std::optional<float> log_prob_threshold;
  std::optional<float> no_speech_threshold;
  std::optional<float> compression_ratio_threshold;

  bool condition_on_previous_text;
  float prompt_reset_on_temperature;
  std::vector<float> temperatures;

  // initial_prompt can be either string or vector<int>
  std::optional<std::variant<std::string, std::vector<int>>> initial_prompt;

  std::optional<std::string> prefix;
  bool suppress_blank;
  std::optional<std::vector<int>> suppress_tokens;
  bool without_timestamps;
  float max_initial_timestamp;
  bool word_timestamps;
  std::string prepend_punctuations;
  std::string append_punctuations;
  bool multilingual;
  std::optional<int> max_new_tokens;

  // clip_timestamps can be string (comma-separated) or vector<float>
  std::variant<std::string, std::vector<float>> clip_timestamps;

  std::optional<float> hallucination_silence_threshold;
  std::optional<std::string> hotwords;
};

// ---------------- TranscriptionInfo ----------------
struct TranscriptionInfo {
  std::string language;
  float language_probability;
  float duration;
  float duration_after_vad;
  std::optional<std::vector<std::pair<std::string, float>>> all_language_probs;
  TranscriptionOptions transcription_options;
  VadOptions vad_options;  // should be defined elsewhere
};

class WhisperModel {
public:
  WhisperModel(
      const std::string &model_size_or_path,
      const std::string &device = "auto",
      const std::vector<int> &device_index = {0},
      const std::string &compute_type = "default",
      int cpu_threads = 0,
      int num_workers = 1,
      const std::string &download_root = "",
      bool local_files_only = false,
      const std::map<std::string, std::string> &files = {},
      const std::string &revision = "",
      const std::string &use_auth_token = ""
  ) {
    // -------------------
    // Model Path Handling
    // -------------------
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

    // -------------------
    // Load Whisper Model
    // -------------------
    ctranslate2::models::Whisper::Options options;
    options.device = device;
    options.device_indices = device_index;
    options.compute_type = compute_type;
    options.intra_threads = cpu_threads;
    options.inter_threads = num_workers;

    model = std::make_unique<ctranslate2::models::Whisper>(model_path, options);

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

    // -------------------
    // Feature extractor
    // -------------------
    // In Python: FeatureExtractor(**kwargs)
    // In C++: You need to reimplement feature extractor logic
    input_stride = 2;
    hop_length = 160; // typical Whisper hop length
    sampling_rate = 16000;
    num_samples_per_token = hop_length * input_stride;
    frames_per_second = sampling_rate / hop_length;
    tokens_per_second = sampling_rate / num_samples_per_token;
    time_precision = 0.02f;
    max_length = 448;
  }

  // -----------------
  // Supported Languages
  // -----------------
  std::vector<std::string> supported_languages() const {
    if (model->is_multilingual()) {
      return LANGUAGE_CODES; // assume you have a constant vector of strings
    }
    return {"en"};
  }

  // -----------------
  // Load feature extractor config
  // -----------------
  std::map<std::string, std::string> get_feature_kwargs(
      const std::string &model_path,
      const std::optional<std::string> &preprocessor_bytes = std::nullopt
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

  // -----------------
  // Transcribe
  // -----------------
  std::tuple<std::vector<Segment>, TranscriptionInfo> transcribe(
      const std::vector<float> &audio,
      const std::optional<std::string> &language = std::nullopt,
      bool multilingual = false,
      bool vad_filter = false
  ) {
    // Detect language if multilingual
    std::string lang = language.value_or("en");
    float language_probability = 1.0;

    if (multilingual && !model->is_multilingual()) {
      std::cerr << "Model is English-only; disabling multilingual mode." << std::endl;
      multilingual = false;
    }

    // -----------------
    // Run VAD if requested
    // -----------------
    std::vector<std::map<std::string, float>> speech_chunks;
    std::vector<float> processed_audio = audio;

    if (vad_filter) {
      speech_chunks = get_speech_timestamps(processed_audio);
      processed_audio = collect_chunks(processed_audio, speech_chunks);
    }

    // -----------------
    // Feature extraction
    // -----------------
    auto features = feature_extractor->extract(processed_audio);

    // -----------------
    // Tokenizer
    // -----------------
    Tokenizer tokenizer(hf_tokenizer, model->is_multilingual(), lang);

    // -----------------
    // Generate segments
    // -----------------
    TranscriptionOptions options;
    // fill options as needed
    std::vector<Segment> segments = generate_segments(features, tokenizer, options);

    // -----------------
    // Restore timestamps if VAD was applied
    // -----------------
    if (!speech_chunks.empty()) {
      segments = restore_speech_timestamps(segments, speech_chunks,
                                           feature_extractor->sampling_rate());
    }

    // -----------------
    // Construct TranscriptionInfo
    // -----------------
    TranscriptionInfo info;
    info.language = lang;
    info.language_probability = language_probability;
    info.duration = static_cast<float>(audio.size()) / feature_extractor->sampling_rate();
    info.duration_after_vad =
        static_cast<float>(processed_audio.size()) / feature_extractor->sampling_rate();
    info.transcription_options = options;

    return {segments, info};
  }

  // --------------------------
  // Split segments by timestamps
  // --------------------------
  std::tuple<std::vector<Segment>, int, bool> split_segments_by_timestamps(
      Tokenizer& tokenizer,
      const std::vector<int>& tokens,
      float time_offset,
      int segment_size,
      float segment_duration,
      int seek
  ) {
    std::vector<Segment> current_segments;
    bool single_timestamp_ending = (tokens.size() >= 2 &&
                                    tokens[tokens.size()-2] < tokenizer.timestamp_begin &&
                                    tokens.back() >= tokenizer.timestamp_begin);

    std::vector<int> consecutive_timestamps;
    for (size_t i = 1; i < tokens.size(); ++i) {
      if (tokens[i] >= tokenizer.timestamp_begin && tokens[i-1] >= tokenizer.timestamp_begin) {
        consecutive_timestamps.push_back(i);
      }
    }

    if (!consecutive_timestamps.empty()) {
      std::vector<int> slices = consecutive_timestamps;
      if (single_timestamp_ending) slices.push_back(tokens.size());

      int last_slice = 0;
      for (int current_slice : slices) {
        std::vector<int> sliced_tokens(tokens.begin() + last_slice, tokens.begin() + current_slice);
        float start_time = time_offset + (sliced_tokens.front() - tokenizer.timestamp_begin) * time_precision;
        float end_time = time_offset + (sliced_tokens.back() - tokenizer.timestamp_begin) * time_precision;

        current_segments.push_back(Segment{seek, start_time, end_time, sliced_tokens});
        last_slice = current_slice;
      }

      if (single_timestamp_ending) {
        seek += segment_size;
      } else {
        int last_timestamp_position = tokens[last_slice-1] - tokenizer.timestamp_begin;
        seek += last_timestamp_position * input_stride;
      }
    } else {
      float duration = segment_duration;
      std::vector<int> timestamps;
      for (int token : tokens) if (token >= tokenizer.timestamp_begin) timestamps.push_back(token);

      if (!timestamps.empty() && timestamps.back() != tokenizer.timestamp_begin) {
        duration = (timestamps.back() - tokenizer.timestamp_begin) * time_precision;
      }

      current_segments.push_back(Segment{seek, time_offset, time_offset + duration, tokens});
      seek += segment_size;
    }

    return {current_segments, seek, single_timestamp_ending};
  }

  // --------------------------
  // Generate segments
  // --------------------------
  std::vector<Segment> generate_segments(
      const std::vector<std::vector<float>>& features,
      Tokenizer& tokenizer,
      const TranscriptionOptions& options
  ) {
    int content_frames = features[0].size() - 1;
    float content_duration = content_frames * feature_extractor->time_per_frame;
    std::vector<int> seek_points;
    std::vector<std::pair<int,int>> seek_clips;

    // Process clip_timestamps
    for (float ts : options.clip_timestamps) {
      seek_points.push_back(std::round(ts * frames_per_second));
    }
    if (seek_points.empty()) seek_points.push_back(0);
    if (seek_points.size() % 2 == 1) seek_points.push_back(content_frames);

    for (size_t i=0; i<seek_points.size(); i+=2) {
      seek_clips.emplace_back(seek_points[i], seek_points[i+1]);
    }

    std::vector<Segment> all_segments;
    int clip_idx = 0;
    int seek = seek_clips[clip_idx].first;

    std::vector<int> all_tokens;
    int prompt_reset_since = 0;

    // Initial prompt
    if (options.initial_prompt.has_value()) {
      std::vector<int> initial_tokens = tokenizer.encode(options.initial_prompt.value());
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

      float time_offset = seek * feature_extractor->time_per_frame;
      int segment_size = std::min({feature_extractor->nb_max_frames,
                                   content_frames - seek,
                                   seek_clip_end - seek});
      auto segment_features = slice_features(features, seek, segment_size);
      segment_features = pad_or_trim(segment_features);

      // Encode segment
      auto encoder_output = encode(segment_features);

      // Generate tokens
      auto [tokens, avg_logprob, temperature, compression_ratio] = generate_with_fallback(encoder_output, tokenizer, options);

      // Split segments by timestamps
      auto [current_segments, new_seek, single_timestamp_ending] =
          split_segments_by_timestamps(tokenizer, tokens, time_offset, segment_size, segment_size * feature_extractor->time_per_frame, seek);

      seek = new_seek;

      // Decode tokens to text
      for (auto& seg : current_segments) {
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
  ctranslate2::StorageView encode(const std::vector<std::vector<float>>& features) {
    bool to_cpu = (model.device == "cuda" && model.device_index.size() > 1);

    std::vector<std::vector<float>> input_features = features;

    // Expand dims if 2D -> add batch dimension
    if (input_features.size() > 0 && input_features[0].size() > 0) {
      // Assuming 2D features -> wrap in batch of 1
      if (input_features.size() == features.size() && input_features[0].size() == features[0].size()) {
        input_features = {features};
      }
    }

    auto storage = get_ctranslate2_storage(input_features);
    return model.encode(storage, to_cpu);
  }

  // --------------------------
  // Generate with fallback loop over temperatures
  // --------------------------
  std::tuple<ctranslate2::models::WhisperGenerationResult, float, float, float>
  generate_with_fallback(
      const ctranslate2::StorageView& encoder_output,
      const std::vector<int>& prompt,
      Tokenizer& tokenizer,
      const TranscriptionOptions& options
  ) {
    ctranslate2::models::WhisperGenerationResult decode_result;
    std::vector<std::tuple<ctranslate2::models::WhisperGenerationResult, float, float, float>> all_results;
    std::vector<std::tuple<ctranslate2::models::WhisperGenerationResult, float, float, float>> below_cr_threshold_results;

    int max_initial_timestamp_index = static_cast<int>(
        std::round(options.max_initial_timestamp / time_precision)
    );

    int max_length = options.max_new_tokens.has_value() ? prompt.size() + options.max_new_tokens.value() : this->max_length;
    if (max_length > this->max_length) {
      throw std::runtime_error("Prompt + max_new_tokens exceeds Whisper max_length");
    }

    for (float temperature : options.temperatures) {
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

      auto result = model.generate(
          encoder_output,
          {prompt},
          options.length_penalty,
          options.repetition_penalty,
          options.no_repeat_ngram_size,
          max_length,
          /*return_scores=*/true,
          /*return_no_speech_prob=*/true,
          options.suppress_blank,
          options.suppress_tokens,
          max_initial_timestamp_index,
          kwargs
      )[0];  // assuming first element as Python does

      const auto& tokens = result.sequences_ids[0];
      int seq_len = tokens.size();
      float cum_logprob = result.scores[0] * std::pow(seq_len, options.length_penalty);
      float avg_logprob = cum_logprob / (seq_len + 1);

      std::string text = tokenizer.decode(tokens);
      float compression_ratio = get_compression_ratio(text);

      decode_result = std::make_tuple(result, avg_logprob, temperature, compression_ratio);
      all_results.push_back(decode_result);

      bool needs_fallback = false;

      if (options.compression_ratio_threshold.has_value() && compression_ratio > options.compression_ratio_threshold.value()) {
        needs_fallback = true;
        logger.debug("Compression ratio threshold not met at temperature %.1f (%.3f > %.3f)",
                     temperature, compression_ratio, options.compression_ratio_threshold.value());
      } else {
        below_cr_threshold_results.push_back(decode_result);
      }

      if (options.log_prob_threshold.has_value() && avg_logprob < options.log_prob_threshold.value()) {
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

      if (!needs_fallback) break;
    }

    // All temperatures failed: pick best average logprob
    if (decode_result.empty()) {
      auto best_it = std::max_element(
          below_cr_threshold_results.begin(), below_cr_threshold_results.end(),
          [](auto& a, auto& b) { return std::get<1>(a) < std::get<1>(b); }
      );
      decode_result = *best_it;
    }

    return decode_result;
  }

  // --------------------------
  // Generate prompt for Whisper
  // --------------------------
  std::vector<int> get_prompt(
      Tokenizer& tokenizer,
      const std::vector<int>& previous_tokens,
      bool without_timestamps = false,
      std::optional<std::string> prefix = std::nullopt,
      std::optional<std::string> hotwords = std::nullopt
  ) {
    std::vector<int> prompt;

    if (!previous_tokens.empty() || (hotwords.has_value() && !prefix.has_value())) {
      prompt.push_back(tokenizer.sot_prev);

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

    prompt.insert(prompt.end(), tokenizer.sot_sequence.begin(), tokenizer.sot_sequence.end());

    if (without_timestamps) {
      prompt.push_back(tokenizer.no_timestamps);
    }

    if (prefix.has_value()) {
      std::string pre = " " + prefix.value();
      std::vector<int> prefix_tokens = tokenizer.encode(pre);
      if (prefix_tokens.size() >= max_length / 2) {
        prefix_tokens.resize(max_length / 2 - 1);
      }
      if (!without_timestamps) {
        prompt.push_back(tokenizer.timestamp_begin);
      }
      prompt.insert(prompt.end(), prefix_tokens.begin(), prefix_tokens.end());
    }

    return prompt;
  }

  // --------------------------
  // Add word timestamps to segments
  // --------------------------
  float add_word_timestamps(
      std::vector<std::vector<std::map<std::string, std::any>>>& segments,
      Tokenizer& tokenizer,
      const ctranslate2::StorageView& encoder_output,
      int num_frames,
      const std::string& prepend_punctuations,
      const std::string& append_punctuations,
      float last_speech_timestamp
  ) {
    if (segments.empty()) return last_speech_timestamp;

    std::vector<std::vector<int>> text_tokens;
    std::vector<std::vector<std::vector<int>>> text_tokens_per_segment;

    for (auto& segment : segments) {
      std::vector<std::vector<int>> segment_tokens;
      for (auto& subsegment : segment) {
        std::vector<int> filtered_tokens;
        auto tokens = std::any_cast<std::vector<int>>(subsegment["tokens"]);
        std::copy_if(tokens.begin(), tokens.end(), std::back_inserter(filtered_tokens),
                     [&](int t) { return t < tokenizer.eot; });
        segment_tokens.push_back(filtered_tokens);
      }
      std::vector<int> flattened;
      for (auto& tvec : segment_tokens)
        flattened.insert(flattened.end(), tvec.begin(), tvec.end());
      text_tokens.push_back(flattened);
      text_tokens_per_segment.push_back(segment_tokens);
    }

    auto alignments = find_alignment(tokenizer, text_tokens, encoder_output, num_frames);

    std::vector<std::pair<float, float>> median_max_durations;
    for (auto& alignment : alignments) {
      std::vector<float> word_durations;
      for (auto& word : alignment) {
        float duration = std::any_cast<float>(word.at("end")) - std::any_cast<float>(word.at("start"));
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

      merge_punctuations(alignment, prepend_punctuations, append_punctuations);
    }

    for (size_t segment_idx = 0; segment_idx < segments.size(); ++segment_idx) {
      auto& segment = segments[segment_idx];
      size_t word_index = 0;
      float time_offset = std::any_cast<int>(segment[0]["seek"]) / frames_per_second;
      auto [median_duration, max_duration] = median_max_durations[segment_idx];

      for (size_t subsegment_idx = 0; subsegment_idx < segment.size(); ++subsegment_idx) {
        auto& subsegment = segment[subsegment_idx];
        int saved_tokens = 0;
        std::vector<std::map<std::string, std::any>> words;

        while (word_index < alignments[segment_idx].size() &&
               saved_tokens < text_tokens_per_segment[segment_idx][subsegment_idx].size()) {
          auto& timing = alignments[segment_idx][word_index];
          if (timing.count("word") && !std::any_cast<std::string>(timing["word"]).empty()) {
            words.push_back({
                                {"word", timing["word"]},
                                {"start", std::round((time_offset + std::any_cast<float>(timing["start"])) * 100) / 100},
                                {"end", std::round((time_offset + std::any_cast<float>(timing["end"])) * 100) / 100},
                                {"probability", timing["probability"]}
                            });
          }
          saved_tokens += std::any_cast<int>(timing["tokens"].size());
          word_index++;
        }
        subsegment["words"] = words;
        if (!words.empty()) last_speech_timestamp = std::any_cast<float>(words.back().at("end"));
      }
    }

    return last_speech_timestamp;
  }

  // --------------------------
  // Alignment of tokens to frames
  // --------------------------
  std::vector<std::vector<std::map<std::string, std::any>>>
  find_alignment(
      Tokenizer& tokenizer,
      const std::vector<std::vector<int>>& text_tokens,
      const ctranslate2::StorageView& encoder_output,
      int num_frames,
      int median_filter_width = 7
  ) {
    std::vector<std::vector<std::map<std::string, std::any>>> return_list;
    if (text_tokens.empty()) return return_list;

    auto results = model.align(encoder_output, tokenizer.sot_sequence, text_tokens, num_frames, median_filter_width);

    for (size_t i = 0; i < results.size(); ++i) {
      const auto& result = results[i];
      const auto& tokens = text_tokens[i];
      auto [words, word_tokens] = tokenizer.split_to_word_tokens(tokens);
      if (word_tokens.size() <= 1) {
        return_list.push_back({});
        continue;
      }

      // Construct alignment
      std::vector<std::map<std::string, std::any>> alignment_list;
      for (size_t j = 0; j < words.size(); ++j) {
        alignment_list.push_back({
                                     {"word", words[j]},
                                     {"tokens", word_tokens[j]},
                                     {"start", 0.0f},  // placeholder, compute from result
                                     {"end", 0.0f},    // placeholder, compute from result
                                     {"probability", 1.0f}  // placeholder
                                 });
      }
      return_list.push_back(alignment_list);
    }

    return return_list;
  }

  // --------------------------
  // Detect language of audio/features
  // --------------------------
  std::tuple<std::string, float, std::vector<std::pair<std::string, float>>>
  detect_language(
      const std::vector<float>* audio = nullptr,
      const std::vector<std::vector<float>>* features = nullptr,
      bool vad_filter = false,
      const std::map<std::string, float>& vad_parameters = {},
      int language_detection_segments = 1,
      float language_detection_threshold = 0.5f
  ) {
    assert(audio != nullptr || features != nullptr);

    std::vector<std::vector<float>> input_features;

    if (audio != nullptr) {
      std::vector<float> processed_audio = *audio;

      if (vad_filter) {
        auto speech_chunks = get_speech_timestamps(processed_audio, vad_parameters);
        auto [audio_chunks, chunks_metadata] = collect_chunks(processed_audio, speech_chunks);
        processed_audio.clear();
        for (auto& chunk : audio_chunks)
          processed_audio.insert(processed_audio.end(), chunk.begin(), chunk.end());
      }

      size_t n_samples = feature_extractor.n_samples;
      if (processed_audio.size() > static_cast<size_t>(language_detection_segments * n_samples)) {
        processed_audio.resize(language_detection_segments * n_samples);
      }

      input_features = feature_extractor.compute_features(processed_audio);
    } else if (features != nullptr) {
      input_features = *features;
    }

    size_t max_frames = feature_extractor.nb_max_frames;
    if (input_features[0].size() > static_cast<size_t>(language_detection_segments * max_frames)) {
      for (auto& row : input_features)
        row.resize(language_detection_segments * max_frames);
    }

    std::map<std::string, std::vector<float>> detected_language_info;
    std::vector<std::pair<std::string, float>> all_language_probs;
    std::string language;
    float language_probability = 0.0f;

    for (size_t i = 0; i < input_features[0].size(); i += max_frames) {
      std::vector<std::vector<float>> segment_features;
      size_t end_idx = std::min(i + max_frames, input_features[0].size());

      for (auto& row : input_features) {
        std::vector<float> segment_row(row.begin() + i, row.begin() + end_idx);
        segment_features.push_back(segment_row);
      }

      auto encoder_output = encode(pad_or_trim(segment_features));
      auto results = model.detect_language(encoder_output)[0]; // returns vector<pair<string,float>>

      // strip markers from token
      all_language_probs.clear();
      for (auto& token_prob : results) {
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
      for (auto& kv : detected_language_info) {
        if (kv.second.size() > max_count) {
          max_count = kv.second.size();
          language = kv.first;
          language_probability = *std::max_element(kv.second.begin(), kv.second.end());
        }
      }
    }

    return {language, language_probability, all_language_probs};
  }

private:
  std::unique_ptr<ctranslate2::models::Whisper> model;
  int input_stride;
  int hop_length;
  int sampling_rate;
  int num_samples_per_token;
  int frames_per_second;
  int tokens_per_second;
  float time_precision;
  int max_length;

  // You need to implement these helpers:
  std::vector<std::pair<std::vector<size_t>, float>> get_speech_timestamps(
      const std::vector<float>& audio,
      const std::map<std::string, float>& vad_parameters
  );
  std::tuple<std::vector<std::vector<float>>, std::vector<std::map<std::string,float>>> collect_chunks(
      const std::vector<float>& audio,
      const std::vector<std::pair<std::vector<size_t>, float>>& speech_chunks
  );

  std::unique_ptr<FeatureExtractor> feature_extractor;
  Tokenizer hf_tokenizer;

  const std::vector<std::string> LANGUAGE_CODES = {
      "en", "fr", "de", "es", "it", "ja", "ko", "zh" // etc.
  };

  // Placeholder for JSON parsing in C++
  std::map<std::string, std::string> parse_json(const std::string &json_str) { /*...*/ }

  std::map<std::string, std::string> parse_json_file(const std::string &path) { /*...*/ }

  std::vector<Segment> generate_segments(
      const std::vector<std::vector<float>> &features,
      Tokenizer &tokenizer,
      const TranscriptionOptions &options
  ) {
    // TODO: call ctranslate2::models::Whisper::generate or decoding loop
    return {};
  }

  std::vector<Segment> restore_speech_timestamps(
      const std::vector<Segment> &segments,
      const std::vector<std::map<std::string, float>> &speech_chunks,
      int sampling_rate
  ) {
    // TODO: implement as in your Python WhisperModel
    return segments;
  }

  // Placeholder functions
  std::vector<std::vector<float>> slice_features(const std::vector<std::vector<float>>& features, int start, int length) { /*...*/ }
  std::vector<std::vector<float>> pad_or_trim(const std::vector<std::vector<float>>& segment) { /*...*/ }
  Logger logger;

  // Placeholder helpers
  ctranslate2::StorageView get_ctranslate2_storage(const std::vector<std::vector<float>>& features) { /*...*/ }
  float get_compression_ratio(const std::string& text) { /*...*/ return 1.0f; }

};


// Placeholder for SpeechTimestampsMap C++ class
class SpeechTimestampsMap {
public:
  SpeechTimestampsMap(const std::vector<std::map<std::string,float>>& speech_chunks,
                      int sampling_rate) {
    // implement constructor
  }

  int get_chunk_index(float t) const {
    // return chunk index containing time t
    return 0;
  }

  float get_original_time(float t, int chunk_index = -1, bool is_end = false) const {
    // map to original audio time
    return t;
  }
};

// ------------------- restore_speech_timestamps -------------------
std::vector<Segment> restore_speech_timestamps(
    std::vector<Segment> segments,
    const std::vector<std::map<std::string,float>>& speech_chunks,
    int sampling_rate
) {
  SpeechTimestampsMap ts_map(speech_chunks, sampling_rate);

  for (auto& segment : segments) {
    if (!segment.words.empty()) {
      std::vector<Word> words;
      for (auto& word : segment.words) {
        float middle = (word.start + word.end) / 2.0f;
        int chunk_index = ts_map.get_chunk_index(middle);
        word.start = ts_map.get_original_time(word.start, chunk_index);
        word.end   = ts_map.get_original_time(word.end, chunk_index);
        words.push_back(word);
      }
      segment.start = words.front().start;
      segment.end   = words.back().end;
      segment.words = words;
    } else {
      segment.start = ts_map.get_original_time(segment.start);
      segment.end   = ts_map.get_original_time(segment.end, -1, true);
    }
  }
  return segments;
}

// ------------------- get_ctranslate2_storage -------------------
// Placeholder for ctranslate2::StorageView
ctranslate2::StorageView get_ctranslate2_storage(const std::vector<std::vector<float>>& segment) {
  // Flatten 2D vector into contiguous memory and wrap in StorageView
  std::vector<float> contiguous;
  for (const auto& row : segment)
    contiguous.insert(contiguous.end(), row.begin(), row.end());
  return ctranslate2::StorageView::from_array(contiguous.data(), segment.size(), segment[0].size());
}

// ------------------- get_compression_ratio -------------------
float get_compression_ratio(const std::string& text) {
  std::vector<unsigned char> compressed(text.size() * 2); // rough estimate
  uLongf compressed_size = compressed.size();
  int res = compress(compressed.data(), &compressed_size,
                     reinterpret_cast<const unsigned char*>(text.data()), text.size());
  if (res != Z_OK) return 1.0f;
  return static_cast<float>(text.size()) / static_cast<float>(compressed_size);
}

// ------------------- get_suppressed_tokens -------------------
std::vector<int> get_suppressed_tokens(
    const Tokenizer& tokenizer,
    const std::vector<int>& suppress_tokens_in
) {
  std::vector<int> suppress_tokens = suppress_tokens_in;

  if (std::find(suppress_tokens.begin(), suppress_tokens.end(), -1) != suppress_tokens.end()) {
    suppress_tokens.erase(std::remove(suppress_tokens.begin(), suppress_tokens.end(), -1), suppress_tokens.end());
    suppress_tokens.insert(suppress_tokens.end(), tokenizer.non_speech_tokens.begin(), tokenizer.non_speech_tokens.end());
  } else if (suppress_tokens.empty()) {
    // interpret empty as empty
  }

  suppress_tokens.push_back(tokenizer.transcribe);
  suppress_tokens.push_back(tokenizer.translate);
  suppress_tokens.push_back(tokenizer.sot);
  suppress_tokens.push_back(tokenizer.sot_prev);
  suppress_tokens.push_back(tokenizer.sot_lm);

  // remove duplicates and sort
  std::set<int> unique_tokens(suppress_tokens.begin(), suppress_tokens.end());
  suppress_tokens.assign(unique_tokens.begin(), unique_tokens.end());
  return suppress_tokens;
}

// ------------------- merge_punctuations -------------------
void merge_punctuations(std::vector<std::map<std::string,std::vector<std::string>>>& alignment,
                        const std::string& prepended,
                        const std::string& appended) {
  // merge prepended punctuations
  int i = alignment.size() - 2;
  int j = alignment.size() - 1;
  while (i >= 0) {
    auto& previous = alignment[i];
    auto& following = alignment[j];
    std::string prev_word = previous["word"][0];
    if (!prev_word.empty() && prev_word[0] == ' ' && prepended.find(prev_word[0]) != std::string::npos) {
      following["word"].insert(following["word"].begin(), previous["word"].begin(), previous["word"].end());
      following["tokens"].insert(following["tokens"].begin(), previous["tokens"].begin(), previous["tokens"].end());
      previous["word"].clear();
      previous["tokens"].clear();
    } else {
      j = i;
    }
    i--;
  }

  // merge appended punctuations
  i = 0; j = 1;
  while (j < alignment.size()) {
    auto& previous = alignment[i];
    auto& following = alignment[j];
    std::string follow_word = following["word"][0];
    std::string prev_word = previous["word"][0];
    if (!prev_word.empty() && prev_word.back() != ' ' && appended.find(follow_word[0]) != std::string::npos) {
      previous["word"].push_back(follow_word);
      previous["tokens"].insert(previous["tokens"].end(), following["tokens"].begin(), following["tokens"].end());
      following["word"].clear();
      following["tokens"].clear();
    } else {
      i = j;
    }
    j++;
  }
}
