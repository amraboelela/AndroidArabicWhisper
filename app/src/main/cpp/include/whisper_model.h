#ifndef WHISPER_MODEL_H
#define WHISPER_MODEL_H

#include "feature_extractor.h"

#include <ctranslate2/models/whisper.h>
#include "tokenizer.h"
#include <string>
#include <vector>
#include <map>
#include <optional>
#include <memory>
#include "vad.h"

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
  std::optional<std::vector<Word>> words;
  std::optional<float> temperature;

  std::string to_string() const {
    std::string words_str = "[";
    if (words.has_value()) {
      for (const auto &w: words.value()) {
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
           ", temperature: " +
           (temperature.has_value() ? std::to_string(temperature.value()) : "null") +
           "}";
  }
};

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
  // C++ constructor to match the Python `__init__`.
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
  );
  std::vector<std::string> supported_languages() const;
  std::map<std::string, std::string> get_feature_kwargs(
      const std::string &model_path,
      const std::optional<std::string> &preprocessor_bytes = std::nullopt
  );
  std::tuple<std::vector<Segment>, TranscriptionInfo> transcribe(
      const std::vector<float> &audio,
      const std::optional<std::string> &language = std::nullopt,
      bool multilingual = false,
      bool vad_filter = false
  );
  std::tuple<std::vector<Segment>, int, bool> split_segments_by_timestamps(
      Tokenizer &tokenizer,
      const std::vector<int> &tokens,
      float time_offset,
      int segment_size,
      float segment_duration,
      int seek
  );
  std::vector<Segment> generate_segments(
      const std::vector<std::vector<float>> &features,
      Tokenizer &tokenizer,
      const TranscriptionOptions &options
  );
  ctranslate2::StorageView encode(const std::vector<std::vector<float>> &features);
  std::tuple<ctranslate2::models::WhisperGenerationResult, float, float, float>
  generate_with_fallback(
      const ctranslate2::StorageView &encoder_output,
      const std::vector<int> &prompt,
      Tokenizer &tokenizer,
      const TranscriptionOptions &options
  );
  std::vector<int> get_prompt(
      Tokenizer &tokenizer,
      const std::vector<int> &previous_tokens,
      bool without_timestamps = false,
      std::optional<std::string> prefix = std::nullopt,
      std::optional<std::string> hotwords = std::nullopt
  );
  float add_word_timestamps(
      std::vector<std::vector<std::map<std::string, std::any>>> &segments,
      Tokenizer &tokenizer,
      const ctranslate2::StorageView &encoder_output,
      int num_frames,
      const std::string &prepend_punctuations,
      const std::string &append_punctuations,
      float last_speech_timestamp
  );
  std::vector<std::vector<std::map<std::string, std::any>>> find_alignment(
      Tokenizer &tokenizer,
      const std::vector<std::vector<int>> &text_tokens,
      const ctranslate2::StorageView &encoder_output,
      int num_frames,
      int median_filter_width = 7
  );
  std::tuple<std::string, float, std::vector<std::pair<std::string, float>>> detect_language(
      const std::vector<float> *audio = nullptr,
      const std::vector<std::vector<float>> *features = nullptr,
      bool vad_filter = false,
      const VadOptions& vad_parameters = VadOptions(),
      int language_detection_segments = 1,
      float language_detection_threshold = 0.5f
  );

private:
  std::shared_ptr<ctranslate2::models::Whisper> model;
  std::shared_ptr<tokenizers::Tokenizer> hf_tokenizer;
  FeatureExtractor feature_extractor;
  int input_stride;
  int num_samples_per_token;
  int frames_per_second;
  int tokens_per_second;
  double time_precision;
  int max_length;
};

// --- Conceptual helper functions (replace with actual implementations) ---

// Conceptual function to simulate Python's `download_model`.
// In a real application, this would use a library like libcurl to download files.
//std::string download_model(
//    const std::string& model_size_or_path,
//    bool local_files_only,
//    const std::optional<std::string>& cache_dir,
//    const std::optional<std::string>& revision,
//    const std::optional<bool>& use_auth_token
//);

#endif