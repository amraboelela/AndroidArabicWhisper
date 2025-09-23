#include "tokenizer.h"
#include "whisper_tokenizer.h"

#include <iostream>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <set>
#include <string_view>

// Use whisper tokenizer for the underlying implementation
namespace tokenizers {
  class Tokenizer {
  private:
  std::unique_ptr<whisper::WhisperTokenizer> whisper_tokenizer_;

  public:
  Tokenizer() : whisper_tokenizer_(std::make_unique<whisper::WhisperTokenizer>()) {}

  int token_to_id(const std::string& token) {
    return whisper_tokenizer_->token_to_id(token);
  }

  std::vector<int> encode(const std::string& text, bool add_special_tokens) {
    return whisper_tokenizer_->encode(text, add_special_tokens);
  }

  std::string decode(const std::vector<int>& tokens) {
    return whisper_tokenizer_->decode(tokens, true);
  }
  };
} // namespace tokenizers

// Global constant definitions, equivalent to the Python tuples.
const std::unordered_set<std::string> _TASKS = {
  "transcribe", "translate"
};

const std::vector<std::string> _LANGUAGE_CODES = {
  "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo", "br", "bs", "ca",
  "cs", "cy", "da", "de", "el", "en", "es", "et", "eu", "fa", "fi", "fo", "fr",
  "gl", "gu", "ha", "haw", "he", "hi", "hr", "ht", "hu", "hy", "id", "is", "it",
  "ja", "jw", "ka", "kk", "km", "kn", "ko", "la", "lb", "ln", "lo", "lt", "lv",
  "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt", "my", "ne", "nl", "nn", "no",
  "oc", "pa", "pl", "ps", "pt", "ro", "ru", "sa", "sd", "si", "sk", "sl", "sn",
  "so", "sq", "sr", "su", "sv", "sw", "ta", "te", "tg", "th", "tk", "tl", "tr",
  "tt", "uk", "ur", "uz", "vi", "yi", "yo", "zh", "yue"
};

// --- Tokenizer Class Implementation ---

Tokenizer::Tokenizer(
  tokenizers::Tokenizer* tokenizer,
  bool multilingual,
  std::optional<std::string> task,
  std::optional<std::string> language
) : _tokenizer(tokenizer), _multilingual(multilingual) {

  // Create whisper tokenizer wrapper for enhanced functionality
  whisper_wrapper_ = std::make_unique<whisper::TokenizerWrapper>(
    multilingual,
    language.value_or("en"),
    task.value_or("transcribe")
  );

  if (multilingual) {
  if (task && _TASKS.find(task.value()) == _TASKS.end()) {
    throw std::invalid_argument("'" + task.value() + "' is not a valid task.");
  }
  if (language && std::find(_LANGUAGE_CODES.begin(), _LANGUAGE_CODES.end(), language.value()) == _LANGUAGE_CODES.end()) {
    throw std::invalid_argument("'" + language.value() + "' is not a valid language code.");
  }

  _task = whisper_wrapper_->get_transcribe();
  if (task.value_or("") == "translate") {
    _task = whisper_wrapper_->get_translate();
  }

  // Use whisper tokenizer to get language token
  auto whisper_tok = whisper::WhisperTokenizer();
  _language = whisper_tok.get_language_token(language.value_or("en"));
  _language_code = language.value_or("en");
  } else {
  _task = std::nullopt;
  _language = std::nullopt;
  _language_code = "en";
  }
}

int Tokenizer::get_transcribe() {
  return whisper_wrapper_->get_transcribe();
}

int Tokenizer::get_translate() {
  return whisper_wrapper_->get_translate();
}

int Tokenizer::get_sot() {
  return whisper_wrapper_->get_sot();
}

int Tokenizer::get_sot_lm() {
  return whisper_wrapper_->get_sot_lm();
}

int Tokenizer::get_sot_prev() {
  return whisper_wrapper_->get_sot_prev();
}

int Tokenizer::get_eot() {
  return whisper_wrapper_->get_eot();
}

int Tokenizer::get_no_timestamps() {
  return whisper_wrapper_->get_no_timestamps();
}

std::vector<int> Tokenizer::get_non_speech_tokens() {
  return whisper_wrapper_->get_non_speech_tokens();
}

int Tokenizer::get_timestamp_begin() {
  return whisper_wrapper_->get_timestamp_begin();
}

std::vector<int> Tokenizer::get_sot_sequence() {
  return whisper_wrapper_->get_sot_sequence();
}

std::vector<int> Tokenizer::encode(const std::string& text) {
  return whisper_wrapper_->encode(text);
}

std::string Tokenizer::decode(const std::vector<int>& tokens) {
  return whisper_wrapper_->decode(tokens);
}

std::string Tokenizer::decode_with_timestamps(const std::vector<int>& tokens) {
  std::string result;
  std::vector<std::vector<int>> outputs = {{}};

  for (int token : tokens) {
  if (token >= get_timestamp_begin()) {
    char buffer[50];
    double timestamp_sec = (token - get_timestamp_begin()) * 0.02;
    snprintf(buffer, sizeof(buffer), "<|%.2f|>", timestamp_sec);
    result += std::string(buffer);
    outputs.push_back({});
  } else {
    outputs.back().push_back(token);
  }
  }

  for (const auto& output_tokens : outputs) {
  result += whisper_wrapper_->decode(output_tokens);
  }

  return result;
}

std::pair<std::vector<std::string>, std::vector<std::vector<int>>>
Tokenizer::split_to_word_tokens(const std::vector<int>& tokens) {
  return whisper_wrapper_->split_to_word_tokens(tokens);
}

std::pair<std::vector<std::string>, std::vector<std::vector<int>>>
Tokenizer::split_tokens_on_unicode(const std::vector<int>& tokens) {
  // Use whisper tokenizer's implementation
  return whisper_wrapper_->split_to_word_tokens(tokens);
}

std::pair<std::vector<std::string>, std::vector<std::vector<int>>>
Tokenizer::split_tokens_on_spaces(const std::vector<int>& tokens) {
  // Use whisper tokenizer's implementation
  return whisper_wrapper_->split_to_word_tokens(tokens);
}
