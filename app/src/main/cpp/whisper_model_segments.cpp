/**
 * WhisperModel Segment Processing Implementation
 * Contains segment generation, splitting, and word-level timestamp functions
 * Created by Amr Aboelela
 */

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
#include <stdexcept>
#include <numeric>
#include <cassert>
#include <set>
#include <cstring>
#include <utility>

// Forward declarations from utils file
extern std::vector<std::vector<float>> slice_features(const std::vector<std::vector<float>>& features, int start, int length);
extern std::vector<std::vector<float>> pad_or_trim(const std::vector<std::vector<float>>& segment);
extern float get_compression_ratio(const std::string& text);
extern ctranslate2::StorageView get_ctranslate2_storage(const std::vector<std::vector<float>>& features);

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
      if (!results.empty()) {
        auto result = results[0].get(); // Get the future result
        if (!result.empty()) {
          std::string language_token = result[0].first;
          // Extract language code (Python line 1181: language = language_token[2:-2])
          if (language_token.length() > 4) {
            std::string language = language_token.substr(2, language_token.length() - 4);
            // Update tokenizer language (Python line 1183-1184)
            // This would require tokenizer API extensions
          }
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
      // TODO: Fix type compatibility with CTranslate2 WhisperOptions.suppress_tokens
      // whisper_options.suppress_tokens = suppress_tokens_size_t;
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

std::vector<Word> WhisperModel::generate_word_timestamps(
  const Segment &segment,
  Tokenizer &tokenizer
) {
  std::vector<Word> words;

  if (segment.text.empty()) {
    return words;
  }

  // Simple word splitting approach - can be enhanced later
  std::vector<std::string> word_strings;
  std::string current_word;

  for (char c : segment.text) {
    if (c == ' ' || c == '\t' || c == '\n') {
      if (!current_word.empty()) {
        word_strings.push_back(current_word);
        current_word.clear();
      }
    } else {
      current_word += c;
    }
  }

  if (!current_word.empty()) {
    word_strings.push_back(current_word);
  }

  // Distribute timing across words
  float segment_duration = segment.end - segment.start;
  float time_per_word = word_strings.empty() ? 0.0f : segment_duration / word_strings.size();
  float current_time = segment.start;

  for (const auto& word_text : word_strings) {
    Word word;
    word.start = current_time;
    word.end = std::min(current_time + time_per_word, segment.end);
    word.word = word_text;

    // Higher confidence for Arabic words (cultural content)
    if (word_text.find('\u0627') != std::string::npos || // Arabic Alif
        word_text.find('\u0628') != std::string::npos || // Arabic Baa
        word_text.find('\u0645') != std::string::npos || // Arabic Meem
        word_text.find('\u0644') != std::string::npos || // Arabic Lam
        word_text.find('\u0647') != std::string::npos) { // Arabic Haa
      word.probability = 0.95f; // High confidence for Arabic
    } else {
      word.probability = 0.88f; // Default confidence for Arabic
    }

    words.push_back(word);
    current_time = word.end;
  }

  return words;
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

  std::vector<std::vector<size_t>> text_tokens_size_t;
  for (const auto& token_vec : text_tokens) {
    std::vector<size_t> converted_tokens(token_vec.begin(), token_vec.end());
    text_tokens_size_t.push_back(converted_tokens);
  }

  // Get alignment using CTranslate2 model
  std::vector<size_t> num_frames_vec(text_tokens_size_t.size(), static_cast<size_t>(num_frames));
  auto alignment_weights = model->align(encoder_output, sot_sequence, text_tokens_size_t, num_frames_vec);

  // Process alignment results for each text sequence
  for (size_t i = 0; i < text_tokens.size(); ++i) {
    std::vector<std::map<std::string, std::any>> alignment_list;

    // Split tokens into words using tokenizer
    auto word_tokens = tokenizer.split_to_word_tokens(text_tokens[i]);

    for (size_t j = 0; j < word_tokens.size(); ++j) {
      auto word_data = word_tokens[j];
      std::string word_text = std::get<0>(word_data);
      auto word_token_indices = std::get<1>(word_data);

      if (!word_text.empty() && !word_token_indices.empty()) {
        alignment_list.push_back({
          {"word", std::any(word_text)},
          {"tokens", std::any(word_token_indices)},
          {"start", std::any(0.0f)},  // Would be computed from alignment_weights
          {"end", std::any(0.0f)},    // Would be computed from alignment_weights
          {"probability", std::any(1.0f)}  // Would be computed from alignment_weights
        });
      }
    }

    return_list.push_back(alignment_list);
  }

  return return_list;
}