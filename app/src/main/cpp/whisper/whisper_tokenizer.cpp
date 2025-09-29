#include "whisper_tokenizer.h"

#ifdef ANDROID

#include <android/log.h>

#else
#include <iostream>
// Define Android logging macros for non-Android builds
#define __android_log_print(level, tag, ...) printf(__VA_ARGS__); printf("\n")
#define ANDROID_LOG_DEBUG 0
#define ANDROID_LOG_ERROR 0
#endif

#include <fstream>
#include <sstream>
#include <algorithm>
#include <regex>
#include <cctype>
#include <iostream>

namespace whisper {

// Language codes to token ID mapping (matching whisper.cpp)
  static const std::unordered_map<std::string, int> LANGUAGE_TO_TOKEN = {
      {"en",  50259},
      {"zh",  50260},
      {"de",  50261},
      {"es",  50262},
      {"ru",  50263},
      {"ko",  50264},
      {"fr",  50265},
      {"ja",  50266},
      {"pt",  50267},
      {"tr",  50268},
      {"pl",  50269},
      {"ca",  50270},
      {"nl",  50271},
      {"ar",  50272},
      {"sv",  50273},
      {"it",  50274},
      {"id",  50275},
      {"hi",  50276},
      {"fi",  50277},
      {"vi",  50278},
      {"he",  50279},
      {"uk",  50280},
      {"el",  50281},
      {"ms",  50282},
      {"cs",  50283},
      {"ro",  50284},
      {"da",  50285},
      {"hu",  50286},
      {"ta",  50287},
      {"no",  50288},
      {"th",  50289},
      {"ur",  50290},
      {"hr",  50291},
      {"bg",  50292},
      {"lt",  50293},
      {"la",  50294},
      {"mi",  50295},
      {"ml",  50296},
      {"cy",  50297},
      {"sk",  50298},
      {"te",  50299},
      {"fa",  50300},
      {"lv",  50301},
      {"bn",  50302},
      {"sr",  50303},
      {"az",  50304},
      {"sl",  50305},
      {"kn",  50306},
      {"et",  50307},
      {"mk",  50308},
      {"br",  50309},
      {"eu",  50310},
      {"is",  50311},
      {"hy",  50312},
      {"ne",  50313},
      {"mn",  50314},
      {"bs",  50315},
      {"kk",  50316},
      {"sq",  50317},
      {"sw",  50318},
      {"gl",  50319},
      {"mr",  50320},
      {"pa",  50321},
      {"si",  50322},
      {"km",  50323},
      {"sn",  50324},
      {"yo",  50325},
      {"so",  50326},
      {"af",  50327},
      {"oc",  50328},
      {"ka",  50329},
      {"be",  50330},
      {"tg",  50331},
      {"sd",  50332},
      {"gu",  50333},
      {"am",  50334},
      {"yi",  50335},
      {"lo",  50336},
      {"uz",  50337},
      {"fo",  50338},
      {"ht",  50339},
      {"ps",  50340},
      {"tk",  50341},
      {"nn",  50342},
      {"mt",  50343},
      {"sa",  50344},
      {"lb",  50345},
      {"my",  50346},
      {"bo",  50347},
      {"tl",  50348},
      {"mg",  50349},
      {"as",  50350},
      {"tt",  50351},
      {"haw", 50352},
      {"ln",  50353},
      {"ha",  50354},
      {"ba",  50355},
      {"jw",  50356},
      {"su",  50357}
  };

  WhisperTokenizer::WhisperTokenizer(const std::string &vocab_file, bool multilingual)
      : multilingual_(multilingual) {

    __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "🔧 WhisperTokenizer constructor called");
    __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "   vocab_file: '%s'",
                        vocab_file.c_str());
    __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "   multilingual: %d", multilingual);

    if (!vocab_file.empty()) {
      __android_log_print(ANDROID_LOG_DEBUG, "#transcribe",
                          "📂 Attempting to load vocabulary from file...");
      bool load_success = load_vocab_from_file(vocab_file);
      __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "📂 Vocabulary loading result: %s",
                          load_success ? "SUCCESS" : "FAILED");

      if (!load_success) {
        __android_log_print(ANDROID_LOG_ERROR, "#transcribe",
                            "❌ Failed to load vocabulary from file, using built-in vocab");
        initialize_builtin_vocab();
      }
    } else {
      __android_log_print(ANDROID_LOG_DEBUG, "#transcribe",
                          "📂 No vocab_file provided, using built-in vocabulary");
      initialize_builtin_vocab();
    }

    __android_log_print(ANDROID_LOG_DEBUG, "#transcribe",
                        "📊 Final vocabulary size after constructor: %zu", vocab_to_id_.size());

    initialize_special_tokens();
    initialize_language_tokens();

    __android_log_print(ANDROID_LOG_DEBUG, "#transcribe",
                        "✅ WhisperTokenizer constructor completed");
  }

#ifndef NO_CTRANSLATE2

  WhisperTokenizer::WhisperTokenizer(const ctranslate2::Vocabulary &vocabulary, bool multilingual)
      : multilingual_(multilingual) {

    load_vocab_from_ctranslate2(vocabulary);
    initialize_special_tokens();
    initialize_language_tokens();
  }

  void WhisperTokenizer::load_vocab_from_ctranslate2(const ctranslate2::Vocabulary &vocabulary) {
    __android_log_print(ANDROID_LOG_DEBUG, "#transcribe",
                        "Loading vocabulary from CTranslate2 model...");

    vocab_to_id_.clear();
    id_to_vocab_.clear();

    // Load all tokens from the CTranslate2 vocabulary
    size_t vocab_size = vocabulary.size();
    __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "CTranslate2 vocabulary size: %zu",
                        vocab_size);

    for (size_t i = 0; i < vocab_size; ++i) {
      const std::string &token = vocabulary.to_token(i);
      vocab_to_id_[token] = static_cast<int>(i);
      id_to_vocab_[static_cast<int>(i)] = token;
    }

    __android_log_print(ANDROID_LOG_DEBUG, "#transcribe",
                        "✅ Loaded %zu tokens from CTranslate2 vocabulary", vocab_size);
  }

#endif // NO_CTRANSLATE2

  void WhisperTokenizer::initialize_builtin_vocab() {
    // Initialize with basic GPT-2 style vocabulary
    // This is a simplified version - in production, you'd load the full GPT-2 vocab

    // Add basic ASCII characters
    for (int i = 0; i < 256; ++i) {
      std::string token(1, static_cast<char>(i));
      vocab_to_id_[token] = i;
      id_to_vocab_[i] = token;
    }

    // Add common English words and subwords (simplified)
    std::vector<std::string> common_tokens = {
        " the", " and", " to", " of", " a", " in", " is", " it", " you", " that",
        " he", " was", " for", " on", " are", " as", " with", " his", " they",
        " I", " at", " be", " this", " have", " from", " or", " one", " had",
        " by", " word", " but", " not", " what", " all", " were", " we", " when",
        " your", " can", " said", " there", " each", " which", " she", " do",
        " how", " their", " if", " will", " up", " other", " about", " out",
        " many", " then", " them", " these", " so", " some", " her", " would",
        " make", " like", " into", " him", " has", " two", " more", " very",
        " after", " words", " first", " where", " much", " through", " back",
        " years", " work", " came", " right", " still", " children", " left"
    };

    int token_id = 256;
    for (const auto &token: common_tokens) {
      vocab_to_id_[token] = token_id;
      id_to_vocab_[token_id] = token;
      ++token_id;
    }

    // Add Arabic tokens (basic set)
    std::vector<std::string> arabic_tokens = {
        " في", " من", " إلى", " على", " أن", " هذا", " هذه", " التي", " الذي",
        " كان", " كانت", " يكون", " تكون", " هو", " هي", " لا", " ما", " قد",
        " أو", " أم", " إذا", " حتى", " عند", " بعد", " قبل", " أثناء",
        " خلال", " أمام", " وراء", " تحت", " فوق", " بين", " ضد", " مع"
    };

    for (const auto &token: arabic_tokens) {
      vocab_to_id_[token] = token_id;
      id_to_vocab_[token_id] = token;
      ++token_id;
    }
  }

  bool WhisperTokenizer::load_vocab_from_file(const std::string &vocab_file) {
    __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Loading vocabulary from file: %s",
                        vocab_file.c_str());

    vocab_to_id_.clear();
    id_to_vocab_.clear();

    if (vocab_file.empty()) {
      __android_log_print(ANDROID_LOG_DEBUG, "#transcribe",
                          "No vocabulary file specified, using built-in");
      return false;
    }

    // Try different paths for Android assets
    std::vector<std::string> paths_to_try = {
        vocab_file,
        "/android_asset/" + vocab_file,
        "assets/" + vocab_file
    };

    std::ifstream file;
    std::string successful_path;

    for (const auto &path: paths_to_try) {
      __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Trying path: %s", path.c_str());
      file.open(path);
      if (file.is_open()) {
        successful_path = path;
        __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Successfully opened: %s",
                            path.c_str());
        break;
      }
      file.clear(); // Reset error flags
    }

    if (!file.is_open()) {
      __android_log_print(ANDROID_LOG_ERROR, "#transcribe",
                          "Failed to open vocabulary file at any path");
      return false;
    }

    __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Reading vocabulary file...");

    // Simple line-based parsing (assuming each line is a token)
    std::string line;
    int token_id = 0;
    bool is_json_format = false;

    // Check if first line starts with '[' (JSON format)
    std::getline(file, line);
    if (!line.empty() && line[0] == '[') {
      is_json_format = true;
      __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Detected JSON format vocabulary");
    } else {
      // Treat first line as first token
      if (!line.empty()) {
        vocab_to_id_[line] = token_id;
        id_to_vocab_[token_id] = line;
        token_id++;
      }
    }

    if (is_json_format) {
      // JSON format: parse tokens from JSON array
      std::string content;
      file.seekg(0);
      content.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

      __android_log_print(ANDROID_LOG_DEBUG, "#transcribe",
                          "Loaded %zu characters, parsing JSON...", content.size());

      // Simple JSON parsing: find tokens between quotes
      size_t pos = 0;
      token_id = 0;

      while (pos < content.length()) {
        // Find opening quote
        pos = content.find('"', pos);
        if (pos == std::string::npos) break;
        pos++; // Skip opening quote

        // Find closing quote, handling escape sequences
        size_t end_pos = pos;
        while (end_pos < content.length()) {
          if (content[end_pos] == '"') {
            break;
          } else if (content[end_pos] == '\\' && end_pos + 1 < content.length()) {
            end_pos += 2; // Skip escape sequence
          } else {
            end_pos++;
          }
        }

        if (end_pos < content.length()) {
          std::string token = content.substr(pos, end_pos - pos);

          // Basic unescape for common sequences
          std::string unescaped_token;
          for (size_t i = 0; i < token.length(); i++) {
            if (token[i] == '\\' && i + 1 < token.length()) {
              char next = token[i + 1];
              switch (next) {
                case 'n':
                  unescaped_token += '\n';
                  i++;
                  break;
                case 't':
                  unescaped_token += '\t';
                  i++;
                  break;
                case 'r':
                  unescaped_token += '\r';
                  i++;
                  break;
                case '\\':
                  unescaped_token += '\\';
                  i++;
                  break;
                case '"':
                  unescaped_token += '"';
                  i++;
                  break;
                case 'u':
                  // Parse Unicode escape sequence \uXXXX
                  if (i + 5 < token.length()) {
                    std::string hex_str = token.substr(i + 2, 4);
                    try {
                      // Convert hex string to Unicode codepoint
                      unsigned int codepoint = std::stoul(hex_str, nullptr, 16);

                      // Convert Unicode codepoint to UTF-8 properly
                      if (codepoint <= 0x7F) {
                        // 1-byte UTF-8 sequence (ASCII)
                        unescaped_token += static_cast<char>(codepoint);
                      } else if (codepoint <= 0x7FF) {
                        // 2-byte UTF-8 sequence
                        unescaped_token += static_cast<char>(0xC0 | (codepoint >> 6));
                        unescaped_token += static_cast<char>(0x80 | (codepoint & 0x3F));
                      } else if (codepoint <= 0xFFFF) {
                        // 3-byte UTF-8 sequence
                        unescaped_token += static_cast<char>(0xE0 | (codepoint >> 12));
                        unescaped_token += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
                        unescaped_token += static_cast<char>(0x80 | (codepoint & 0x3F));
                      } else {
                        // 4-byte UTF-8 sequence (rare Unicode planes)
                        unescaped_token += static_cast<char>(0xF0 | (codepoint >> 18));
                        unescaped_token += static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F));
                        unescaped_token += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
                        unescaped_token += static_cast<char>(0x80 | (codepoint & 0x3F));
                      }
                      i += 5; // Skip \uXXXX
                    } catch (...) {
                      // If parsing fails, keep the escape sequence as-is
                      unescaped_token += token.substr(i, 6);
                      i += 5;
                    }
                  } else {
                    // Not enough characters for full escape sequence
                    unescaped_token += token[i];
                  }
                  break;
                default:
                  unescaped_token += token[i];
                  break;
              }
            } else {
              unescaped_token += token[i];
            }
          }

          vocab_to_id_[unescaped_token] = token_id;
          id_to_vocab_[token_id] = unescaped_token;
          token_id++;

          pos = end_pos + 1;
        } else {
          break;
        }
      }
    } else {
      // Line-based format: each line is a token
      while (std::getline(file, line)) {
        if (!line.empty()) {
          vocab_to_id_[line] = token_id;
          id_to_vocab_[token_id] = line;
          token_id++;
        }
      }
    }

    file.close();

    __android_log_print(ANDROID_LOG_DEBUG, "#transcribe",
                        "✅ Successfully loaded %d tokens from vocabulary file", token_id);

    // Add a verification log for the problematic token
    if (token_id > 28814) {
      auto it = id_to_vocab_.find(28814);
      if (it != id_to_vocab_.end()) {
        __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "✅ Verification: Token 28814 = '%s'",
                            it->second.c_str());
      }
    }

    return token_id > 0;
  }

  void WhisperTokenizer::initialize_special_tokens() {
    // Add special tokens to vocabulary if not present (matching actual model vocabulary)
    std::unordered_map<std::string, int> special_tokens = {
        {"<|endoftext|>",         EOT_TOKEN},           // 50257
        {"<|startoftranscript|>", SOT_TOKEN},   // 50258 - CORRECTED!
        {"<|transcribe|>",        TRANSCRIBE_TOKEN},   // 50359
        {"<|translate|>",         TRANSLATE_TOKEN},     // 50358
        {"<|notimestamps|>",      NO_TIMESTAMPS_TOKEN}, // 50363
        {"<|startofprev|>",       SOT_PREV_TOKEN},    // 50361 - CORRECTED!
        {"<|startoflm|>",         SOT_LM_TOKEN}         // 50360 - CORRECTED!
    };

    for (const auto &[token, id]: special_tokens) {
      vocab_to_id_[token] = id;
      id_to_vocab_[id] = token;
    }

    // Add timestamp tokens
    for (int i = 0; i < 1500; ++i) { // 30 seconds * 50 tokens per second
      int token_id = TIMESTAMP_BEGIN + i;
      float seconds = i * 0.02f; // 20ms per token

      char buffer[32];
      snprintf(buffer, sizeof(buffer), "<|%.2f|>", seconds);
      std::string token_str(buffer);

      vocab_to_id_[token_str] = token_id;
      id_to_vocab_[token_id] = token_str;
    }
  }

  void WhisperTokenizer::initialize_language_tokens() {
    for (const auto &[lang_code, token_id]: LANGUAGE_TO_TOKEN) {
      std::string token_str = "<|" + lang_code + "|>";
      vocab_to_id_[token_str] = token_id;
      id_to_vocab_[token_id] = token_str;
      language_tokens_[lang_code] = token_id;
    }
  }

  std::vector<int>
  WhisperTokenizer::encode(const std::string &text, bool add_special_tokens) const {
    if (text.empty()) {
      return {};
    }

    // Normalize text
    std::string normalized = normalize_text(text);

    // Tokenize into subwords
    auto tokens = tokenize_text(normalized);

    // Convert to IDs
    std::vector<int> token_ids;
    for (const auto &token: tokens) {
      auto it = vocab_to_id_.find(token);
      if (it != vocab_to_id_.end()) {
        token_ids.push_back(it->second);
      } else {
        // Handle unknown tokens - split into characters
        for (char c: token) {
          auto char_it = vocab_to_id_.find(std::string(1, c));
          if (char_it != vocab_to_id_.end()) {
            token_ids.push_back(char_it->second);
          }
        }
      }
    }

    return token_ids;
  }

  std::string
  WhisperTokenizer::decode(const std::vector<int> &tokens, bool skip_special_tokens) const {
    __android_log_print(ANDROID_LOG_DEBUG, "#transcribe",
                        "🔍 WhisperTokenizer::decode() called with %zu tokens, skip_special=%d",
                        tokens.size(), skip_special_tokens);

    // First pass: collect all raw BPE tokens
    std::string raw_bpe;

    for (size_t i = 0; i < tokens.size(); ++i) {
      int token_id = tokens[i];
      __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Processing token %zu/%zu: ID=%d",
                          i + 1, tokens.size(), token_id);

      auto it = id_to_vocab_.find(token_id);
      if (it != id_to_vocab_.end()) {
        const std::string &token = it->second;
        __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Token %d -> '%s' (length: %zu)",
                            token_id, token.c_str(), token.length());

        // Skip special tokens if requested
        if (skip_special_tokens) {
          __android_log_print(ANDROID_LOG_DEBUG, "#transcribe",
                              "Checking if token '%s' is special...", token.c_str());

          // SAFER special token detection - avoid substr on very long strings
          if (token.length() >= 4 && token[0] == '<' && token[1] == '|' &&
              token[token.length() - 2] == '|' && token[token.length() - 1] == '>') {
            __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Skipping special token: '%s'",
                                token.c_str());
            continue;
          }
        }

        raw_bpe += token;
        __android_log_print(ANDROID_LOG_DEBUG, "#transcribe",
                            "✅ Token %zu added to raw BPE, length now: %zu", i + 1,
                            raw_bpe.length());
      } else {
        __android_log_print(ANDROID_LOG_DEBUG, "#transcribe",
                            "⚠️ Token ID %d not found in vocabulary!", token_id);
      }
    }

    // Second pass: decode BPE to proper text
    std::string result = decode_bpe(raw_bpe);

    __android_log_print(ANDROID_LOG_DEBUG, "#transcribe",
                        "🎯 WhisperTokenizer::decode() COMPLETED! Final result: '%s'",
                        result.c_str());
    return result;
  }

  std::string WhisperTokenizer::decode_bpe(const std::string &raw_bpe) const {
    __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "🔧 decode_bpe() called with length: %zu",
                        raw_bpe.length());

    // For now, implement the basic approach similar to faster-whisper:
    // Replace Ġ (U+0120) with spaces and return the result
    std::string result;
    result.reserve(raw_bpe.length());

    for (size_t i = 0; i < raw_bpe.length();) {
      // Handle BPE space prefix 'Ġ' (U+0120 encoded as UTF-8: 0xC4 0xA0)
      if (i + 1 < raw_bpe.length() &&
          (unsigned char) raw_bpe[i] == 0xC4 &&
          (unsigned char) raw_bpe[i + 1] == 0xA0) {
        result += ' ';  // Replace Ġ with space
        i += 2;
        __android_log_print(ANDROID_LOG_DEBUG, "#transcribe",
                            "🔧 Converted Ġ to space at position %zu", i - 2);
      } else {
        // Copy character as-is - let UTF-8 handle the rest
        result += raw_bpe[i];
        i++;
      }
    }

    __android_log_print(ANDROID_LOG_DEBUG, "#transcribe",
                        "🔧 decode_bpe() result: '%s' (length: %zu)", result.c_str(),
                        result.length());

    // Log some bytes for debugging
    if (result.length() > 0) {
      std::string bytes_debug = "First 20 result bytes: ";
      for (size_t i = 0; i < std::min(result.length(), size_t(20)); ++i) {
        char byte_str[10];
        sprintf(byte_str, "\\x%02x", (unsigned char) result[i]);
        bytes_debug += byte_str;
      }
      __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "%s", bytes_debug.c_str());
    }

    return result;
  }

  int WhisperTokenizer::token_to_id(const std::string &token) const {
    auto it = vocab_to_id_.find(token);
    return (it != vocab_to_id_.end()) ? it->second : -1;
  }

  std::string WhisperTokenizer::id_to_token(int id) const {
    auto it = id_to_vocab_.find(id);
    return (it != id_to_vocab_.end()) ? it->second : "";
  }

  int WhisperTokenizer::get_language_token(const std::string &language_code) const {
    auto it = language_tokens_.find(language_code);
    return (it != language_tokens_.end()) ? it->second : -1;
  }

  std::vector<int> WhisperTokenizer::get_sot_sequence(const std::string &language_code,
                                                      const std::string &task) const {
    __android_log_print(ANDROID_LOG_DEBUG, "#transcribe",
                        "WhisperTokenizer::get_sot_sequence called");
    __android_log_print(ANDROID_LOG_DEBUG, "#transcribe",
                        "WhisperTokenizer params - language_code='%s', task='%s', multilingual_=%d",
                        language_code.c_str(), task.c_str(), multilingual_);

    std::vector<int> sequence = {SOT_TOKEN};
    __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "WhisperTokenizer - Added SOT_TOKEN: %d",
                        SOT_TOKEN);

    if (multilingual_ && !language_code.empty()) {
      int lang_token = get_language_token(language_code);
      __android_log_print(ANDROID_LOG_DEBUG, "#transcribe",
                          "WhisperTokenizer - Language token for '%s': %d",
                          language_code.c_str(), lang_token);
      if (lang_token != -1) {
        sequence.push_back(lang_token);
        __android_log_print(ANDROID_LOG_DEBUG, "#transcribe",
                            "WhisperTokenizer - Added language token: %d", lang_token);
      }
    }

    if (task == "transcribe") {
      sequence.push_back(TRANSCRIBE_TOKEN);
      __android_log_print(ANDROID_LOG_DEBUG, "#transcribe",
                          "WhisperTokenizer - Added TRANSCRIBE_TOKEN: %d", TRANSCRIBE_TOKEN);
    } else if (task == "translate") {
      sequence.push_back(TRANSLATE_TOKEN);
      __android_log_print(ANDROID_LOG_DEBUG, "#transcribe",
                          "WhisperTokenizer - Added TRANSLATE_TOKEN: %d", TRANSLATE_TOKEN);
    }

    __android_log_print(ANDROID_LOG_DEBUG, "#transcribe",
                        "WhisperTokenizer::get_sot_sequence final sequence length: %zu",
                        sequence.size());

    std::string seq_str = "WhisperTokenizer::get_sot_sequence final sequence: ";
    for (size_t i = 0; i < sequence.size(); ++i) {
      seq_str += std::to_string(sequence[i]);
      if (i < sequence.size() - 1) seq_str += ", ";
    }
    __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "%s", seq_str.c_str());

    return sequence;
  }

  std::vector<int> WhisperTokenizer::get_non_speech_tokens() const {
    if (!non_speech_tokens_cache_.has_value()) {
      std::unordered_set<int> tokens;

      // Punctuation and symbols
      std::string symbols = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
      for (char c: symbols) {
        std::string token(1, c);
        int id = token_to_id(token);
        if (id != -1) tokens.insert(id);

        std::string spaced_token = " " + token;
        id = token_to_id(spaced_token);
        if (id != -1) tokens.insert(id);
      }

      // Musical symbols
      std::vector<std::string> musical = {"♩", "♪", "♫", "♬", "♭", "♮", "♯"};
      for (const auto &symbol: musical) {
        int id = token_to_id(symbol);
        if (id != -1) tokens.insert(id);

        std::string spaced_symbol = " " + symbol;
        id = token_to_id(spaced_symbol);
        if (id != -1) tokens.insert(id);
      }

      non_speech_tokens_cache_ = std::vector<int>(tokens.begin(), tokens.end());
    }

    return non_speech_tokens_cache_.value();
  }

  bool WhisperTokenizer::is_timestamp_token(int token_id) const {
    return token_id >= TIMESTAMP_BEGIN && token_id < TIMESTAMP_BEGIN + 1500;
  }

  float WhisperTokenizer::timestamp_to_seconds(int token_id) const {
    if (!is_timestamp_token(token_id)) {
      return -1.0f;
    }
    return (token_id - TIMESTAMP_BEGIN) * 0.02f;
  }

  int WhisperTokenizer::seconds_to_timestamp(float seconds) const {
    int offset = static_cast<int>(seconds / 0.02f);
    return TIMESTAMP_BEGIN + offset;
  }

  std::pair<std::vector<std::string>, std::vector<std::vector<int>>>
  WhisperTokenizer::split_to_word_tokens(const std::vector<int> &tokens) const {
    std::vector<std::string> words;
    std::vector<std::vector<int>> word_tokens;

    std::vector<int> current_word_tokens;
    std::string current_word;

    for (int token_id: tokens) {
      if (token_id >= EOT_TOKEN) {
        // Special token - finish current word if any
        if (!current_word_tokens.empty()) {
          words.push_back(current_word);
          word_tokens.push_back(current_word_tokens);
          current_word.clear();
          current_word_tokens.clear();
        }
        continue;
      }

      std::string token_str = id_to_token(token_id);
      if (token_str.empty()) continue;

      current_word_tokens.push_back(token_id);
      current_word += token_str;

      // Check if this completes a word (ends with space or punctuation)
      if (!token_str.empty() && (token_str.back() == ' ' ||
                                 std::ispunct(static_cast<unsigned char>(token_str.back())))) {
        words.push_back(current_word);
        word_tokens.push_back(current_word_tokens);
        current_word.clear();
        current_word_tokens.clear();
      }
    }

    // Add final word if any
    if (!current_word_tokens.empty()) {
      words.push_back(current_word);
      word_tokens.push_back(current_word_tokens);
    }

    return {words, word_tokens};
  }

  std::string WhisperTokenizer::normalize_text(const std::string &text) const {
    std::string normalized = text;

    // Basic normalization - convert to lowercase and trim
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), ::tolower);

    // Remove extra whitespace
    normalized = std::regex_replace(normalized, std::regex("\\s+"), " ");

    // Trim leading/trailing whitespace
    normalized.erase(0, normalized.find_first_not_of(" \\t\\n\\r"));
    normalized.erase(normalized.find_last_not_of(" \\t\\n\\r") + 1);

    return normalized;
  }

  std::vector<std::string> WhisperTokenizer::tokenize_text(const std::string &text) const {
    // Simple whitespace tokenization as fallback
    // In production, you'd use proper BPE tokenization
    std::vector<std::string> tokens;
    std::stringstream ss(text);
    std::string token;

    while (ss >> token) {
      // Add space prefix except for first token
      if (!tokens.empty()) {
        tokens.push_back(" " + token);
      } else {
        tokens.push_back(token);
      }
    }

    return tokens;
  }

// TokenizerWrapper implementation
  TokenizerWrapper::TokenizerWrapper(bool multilingual, const std::string &language,
                                     const std::string &task, const std::string &vocab_path)
      : tokenizer_(std::make_unique<WhisperTokenizer>(vocab_path, multilingual)),
        language_(language), task_(task) {

    __android_log_print(ANDROID_LOG_DEBUG, "#transcribe",
                        "❌ FILE-BASED TokenizerWrapper constructor called - THIS IS THE WRONG ONE!");
    __android_log_print(ANDROID_LOG_DEBUG, "#transcribe",
                        "TokenizerWrapper params - multilingual: %d, language: %s, task: %s, vocab_path: %s",
                        multilingual, language.c_str(), task.c_str(), vocab_path.c_str());
    __android_log_print(ANDROID_LOG_DEBUG, "#transcribe",
                        "WhisperTokenizer created in TokenizerWrapper");

    // Test basic token retrieval
    try {
      int sot = tokenizer_->get_sot_token();
      __android_log_print(ANDROID_LOG_DEBUG, "#transcribe",
                          "TokenizerWrapper - SOT token from WhisperTokenizer: %d", sot);
    } catch (const std::exception &e) {
      __android_log_print(ANDROID_LOG_DEBUG, "#transcribe",
                          "TokenizerWrapper - Error getting SOT token: %s", e.what());
    }
  }

#ifndef NO_CTRANSLATE2

  TokenizerWrapper::TokenizerWrapper(const ctranslate2::Vocabulary &vocabulary, bool multilingual,
                                     const std::string &language, const std::string &task)
      : tokenizer_(std::make_unique<WhisperTokenizer>(vocabulary, multilingual)),
        language_(language), task_(task) {

    __android_log_print(ANDROID_LOG_DEBUG, "#transcribe",
                        "🔥 CTRANSLATE2 TokenizerWrapper constructor called - THIS IS THE CORRECT ONE!");
    __android_log_print(ANDROID_LOG_DEBUG, "#transcribe",
                        "TokenizerWrapper params - multilingual: %d, language: %s, task: %s",
                        multilingual, language.c_str(), task.c_str());
    __android_log_print(ANDROID_LOG_DEBUG, "#transcribe",
                        "WhisperTokenizer created with CTranslate2 vocabulary");

    // Test basic token retrieval
    try {
      int sot = tokenizer_->get_sot_token();
      __android_log_print(ANDROID_LOG_DEBUG, "#transcribe",
                          "TokenizerWrapper - SOT token from WhisperTokenizer: %d", sot);
    } catch (const std::exception &e) {
      __android_log_print(ANDROID_LOG_DEBUG, "#transcribe",
                          "TokenizerWrapper - Error getting SOT token: %s", e.what());
    }

    __android_log_print(ANDROID_LOG_DEBUG, "#transcribe",
                        "TokenizerWrapper (CTranslate2) created successfully");
  }

#endif // NO_CTRANSLATE2

  int TokenizerWrapper::get_transcribe() const {
    return tokenizer_->get_transcribe_token();
  }

  int TokenizerWrapper::get_translate() const {
    return tokenizer_->get_translate_token();
  }

  int TokenizerWrapper::get_sot() const {
    return tokenizer_->get_sot_token();
  }

  int TokenizerWrapper::get_sot_lm() const {
    return tokenizer_->get_sot_lm_token();
  }

  int TokenizerWrapper::get_sot_prev() const {
    return tokenizer_->get_sot_prev_token();
  }

  int TokenizerWrapper::get_eot() const {
    return tokenizer_->get_eot_token();
  }

  int TokenizerWrapper::get_no_timestamps() const {
    return tokenizer_->get_no_timestamps_token();
  }

  int TokenizerWrapper::get_timestamp_begin() const {
    return tokenizer_->get_timestamp_begin();
  }

  std::vector<int> TokenizerWrapper::get_sot_sequence() const {
    __android_log_print(ANDROID_LOG_DEBUG, "#transcribe",
                        "TokenizerWrapper::get_sot_sequence called");
    __android_log_print(ANDROID_LOG_DEBUG, "#transcribe",
                        "Calling tokenizer_->get_sot_sequence with language='%s', task='%s'",
                        language_.c_str(), task_.c_str());

    auto result = tokenizer_->get_sot_sequence(language_, task_);

    __android_log_print(ANDROID_LOG_DEBUG, "#transcribe",
                        "TokenizerWrapper::get_sot_sequence result length: %zu", result.size());

    std::string result_str = "TokenizerWrapper::get_sot_sequence result: ";
    for (size_t i = 0; i < result.size(); ++i) {
      result_str += std::to_string(result[i]);
      if (i < result.size() - 1) result_str += ", ";
    }
    __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "%s", result_str.c_str());

    return result;
  }

  std::vector<int> TokenizerWrapper::get_non_speech_tokens() const {
    return tokenizer_->get_non_speech_tokens();
  }

  std::vector<int> TokenizerWrapper::encode(const std::string &text) const {
    return tokenizer_->encode(text, false);
  }

  std::string TokenizerWrapper::decode(const std::vector<int> &tokens) const {
    return tokenizer_->decode(tokens, true);
  }

  std::pair<std::vector<std::string>, std::vector<std::vector<int>>>
  TokenizerWrapper::split_to_word_tokens(const std::vector<int> &tokens) const {
    return tokenizer_->split_to_word_tokens(tokens);
  }

  int TokenizerWrapper::get_language_token(const std::string &language_code) const {
    return tokenizer_->get_language_token(language_code);
  }

  bool TokenizerWrapper::is_multilingual() const {
    return tokenizer_->is_multilingual();
  }

} // namespace whisper