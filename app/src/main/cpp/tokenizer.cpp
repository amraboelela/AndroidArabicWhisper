#include "Tokenizer.h"

#include <iostream>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <set>
#include <string_view>

// A mock implementation of the `tokenizers::Tokenizer` class.
// In a real application, you would link to the actual tokenizer library.
namespace tokenizers {
  class Tokenizer {
  public:
    int token_to_id(const std::string& token) {
      // A simple hardcoded mapping for demonstration.
      if (token == "<|transcribe|>") return 50359;
      if (token == "<|translate|>") return 50358;
      if (token == "<|startoftranscript|>") return 50257;
      if (token == "<|startoflm|>") return 50361;
      if (token == "<|startofprev|>") return 50362;
      if (token == "<|endoftext|>") return 50256;
      if (token == "<|notimestamps|>") return 50363;
      if (token == " -") return 11;
      if (token == " '") return 12;
      return -1; // Placeholder for unknown tokens.
    }

    std::vector<int> encode(const std::string& text, bool add_special_tokens) {
      // A very basic mock encoding for demonstration.
      std::vector<int> tokens;
      for (char c : text) {
        tokens.push_back(static_cast<int>(c));
      }
      return tokens;
    }

    std::string decode(const std::vector<int>& tokens) {
      // A basic mock decoding.
      std::string text;
      for (int token : tokens) {
        text += static_cast<char>(token);
      }
      return text;
    }
  };
} // namespace tokenizers

// Global constant definitions, equivalent to the Python tuples.
const std::unordered_set<std::string> _TASKS = {
    "transcribe", "translate"
};

const std::unordered_set<std::string> _LANGUAGE_CODES = {
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
  if (multilingual) {
    if (task && _TASKS.find(task.value()) == _TASKS.end()) {
      throw std::invalid_argument("'" + task.value() + "' is not a valid task.");
    }
    if (language && _LANGUAGE_CODES.find(language.value()) == _LANGUAGE_CODES.end()) {
      throw std::invalid_argument("'" + language.value() + "' is not a valid language code.");
    }
    _task = _tokenizer->token_to_id("<|" + task.value() + "|>");
    _language = _tokenizer->token_to_id("<|" + language.value() + "|>");
    _language_code = language.value();
  } else {
    _task = std::nullopt;
    _language = std::nullopt;
    _language_code = "en";
  }
}

int Tokenizer::get_transcribe() {
  if (!_transcribe) {
    _transcribe = _tokenizer->token_to_id("<|transcribe|>");
  }
  return _transcribe.value();
}

int Tokenizer::get_translate() {
  if (!_translate) {
    _translate = _tokenizer->token_to_id("<|translate|>");
  }
  return _translate.value();
}

int Tokenizer::get_sot() {
  if (!_sot) {
    _sot = _tokenizer->token_to_id("<|startoftranscript|>");
  }
  return _sot.value();
}

int Tokenizer::get_sot_lm() {
  if (!_sot_lm) {
    _sot_lm = _tokenizer->token_to_id("<|startoflm|>");
  }
  return _sot_lm.value();
}

int Tokenizer::get_sot_prev() {
  if (!_sot_prev) {
    _sot_prev = _tokenizer->token_to_id("<|startofprev|>");
  }
  return _sot_prev.value();
}

int Tokenizer::get_eot() {
  if (!_eot) {
    _eot = _tokenizer->token_to_id("<|endoftext|>");
  }
  return _eot.value();
}

int Tokenizer::get_no_timestamps() {
  if (!_no_timestamps) {
    _no_timestamps = _tokenizer->token_to_id("<|notimestamps|>");
  }
  return _no_timestamps.value();
}

std::vector<int> Tokenizer::get_non_speech_tokens() {
  if (!_non_speech_tokens) {
    std::set<int> result;

    std::string symbols = R"("_#()*+/:;<=>@[\\]^_`{|}~「」『』)";
    std::string miscellaneous = "♩♪♫♬♭♮♯";

    // Handle " -" and " '" separately
    result.insert(encode(" -")[0]);
    result.insert(encode(" '")[0]);

    for (char symbol : symbols) {
      std::string s(1, symbol);
      std::vector<int> tokens_with_space = encode(" " + s);
      if (!tokens_with_space.empty()) {
        result.insert(tokens_with_space[0]);
      }
      std::vector<int> tokens_without_space = encode(s);
      if (!tokens_without_space.empty()) {
        result.insert(tokens_without_space[0]);
      }
    }

    for (char symbol : miscellaneous) {
      std::string s(1, symbol);
      std::vector<int> tokens_with_space = encode(" " + s);
      if (!tokens_with_space.empty()) {
        result.insert(tokens_with_space[0]);
      }
      std::vector<int> tokens_without_space = encode(s);
      if (!tokens_without_space.empty()) {
        result.insert(tokens_without_space[0]);
      }
    }

    std::vector<int> non_speech_tokens_vec(result.begin(), result.end());
    _non_speech_tokens = non_speech_tokens_vec;
  }
  return _non_speech_tokens.value();
}

int Tokenizer::get_timestamp_begin() {
  return get_no_timestamps() + 1;
}

std::vector<int> Tokenizer::get_sot_sequence() {
  std::vector<int> sequence = {get_sot()};

  if (_language) {
    sequence.push_back(_language.value());
  }

  if (_task) {
    sequence.push_back(_task.value());
  }

  return sequence;
}

std::vector<int> Tokenizer::encode(const std::string& text) {
  return _tokenizer->encode(text, false);
}

std::string Tokenizer::decode(const std::vector<int>& tokens) {
  std::vector<int> text_tokens;
  for (int token : tokens) {
    if (token < get_eot()) {
      text_tokens.push_back(token);
    }
  }
  return _tokenizer->decode(text_tokens);
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
    result += _tokenizer->decode(output_tokens);
  }

  return result;
}

std::pair<std::vector<std::string>, std::vector<std::vector<int>>>
Tokenizer::split_to_word_tokens(const std::vector<int>& tokens) {
  if (_language_code == "zh" || _language_code == "ja" || _language_code == "th" ||
      _language_code == "lo" || _language_code == "my" || _language_code == "yue") {
    return split_tokens_on_unicode(tokens);
  }
  return split_tokens_on_spaces(tokens);
}

std::pair<std::vector<std::string>, std::vector<std::vector<int>>>
Tokenizer::split_tokens_on_unicode(const std::vector<int>& tokens) {
  // This is a simplified C++ implementation of the complex logic.
  // A full implementation would require more advanced unicode handling.
  return {{"mock_word"}, {{1}}};
}

std::pair<std::vector<std::string>, std::vector<std::vector<int>>>
Tokenizer::split_tokens_on_spaces(const std::vector<int>& tokens) {
  // This is a simplified C++ implementation of the complex logic.
  // A full implementation would require more advanced unicode handling.
  return {{"mock_word_with_spaces"}, {{2}}};
}
