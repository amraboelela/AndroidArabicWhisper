#ifndef WHISPER_TOKENIZER_H
#define WHISPER_TOKENIZER_H

#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <optional>
#include <memory>

namespace whisper {

/**
 * Whisper tokenizer implementation compatible with whisper.cpp
 * Based on GPT-2 BPE tokenizer with whisper-specific special tokens
 */
class WhisperTokenizer {
public:
    // Special token IDs (matching whisper.cpp)
    static constexpr int EOT_TOKEN = 50256;
    static constexpr int SOT_TOKEN = 50257;
    static constexpr int TRANSCRIBE_TOKEN = 50359;
    static constexpr int TRANSLATE_TOKEN = 50358;
    static constexpr int NO_TIMESTAMPS_TOKEN = 50363;
    static constexpr int TIMESTAMP_BEGIN = 50364;
    static constexpr int SOT_PREV_TOKEN = 50362;
    static constexpr int SOT_LM_TOKEN = 50361;

    // Language token IDs (starting from 50259)
    static constexpr int LANGUAGE_TOKEN_START = 50259;

    /**
     * Constructor
     * @param vocab_file Path to vocabulary file (optional for built-in vocab)
     * @param multilingual Whether to support multiple languages
     */
    explicit WhisperTokenizer(const std::string& vocab_file = "", bool multilingual = true);

    /**
     * Initialize with built-in vocabulary (GPT-2 based)
     */
    void initialize_builtin_vocab();

    /**
     * Load vocabulary from file
     * @param vocab_file Path to vocabulary file
     */
    bool load_vocab_from_file(const std::string& vocab_file);

    /**
     * Encode text to token IDs
     * @param text Input text
     * @param add_special_tokens Whether to add special tokens
     * @return Vector of token IDs
     */
    std::vector<int> encode(const std::string& text, bool add_special_tokens = false) const;

    /**
     * Decode token IDs to text
     * @param tokens Vector of token IDs
     * @param skip_special_tokens Whether to skip special tokens
     * @return Decoded text
     */
    std::string decode(const std::vector<int>& tokens, bool skip_special_tokens = true) const;

    /**
     * Get token ID for a specific token string
     * @param token Token string
     * @return Token ID, -1 if not found
     */
    int token_to_id(const std::string& token) const;

    /**
     * Get token string for a specific token ID
     * @param id Token ID
     * @return Token string, empty if not found
     */
    std::string id_to_token(int id) const;

    /**
     * Get special token IDs
     */
    int get_eot_token() const { return EOT_TOKEN; }
    int get_sot_token() const { return SOT_TOKEN; }
    int get_transcribe_token() const { return TRANSCRIBE_TOKEN; }
    int get_translate_token() const { return TRANSLATE_TOKEN; }
    int get_no_timestamps_token() const { return NO_TIMESTAMPS_TOKEN; }
    int get_timestamp_begin() const { return TIMESTAMP_BEGIN; }
    int get_sot_prev_token() const { return SOT_PREV_TOKEN; }
    int get_sot_lm_token() const { return SOT_LM_TOKEN; }

    /**
     * Get language token ID for a language code
     * @param language_code Language code (e.g., "en", "ar", "fr")
     * @return Language token ID, -1 if not supported
     */
    int get_language_token(const std::string& language_code) const;

    /**
     * Get start-of-transcript sequence for given language and task
     * @param language_code Language code (optional)
     * @param task Task ("transcribe" or "translate")
     * @return SOT sequence tokens
     */
    std::vector<int> get_sot_sequence(const std::string& language_code = "",
                                      const std::string& task = "transcribe") const;

    /**
     * Get non-speech tokens (punctuation, symbols, etc.)
     * @return Vector of non-speech token IDs
     */
    std::vector<int> get_non_speech_tokens() const;

    /**
     * Check if token is a timestamp token
     * @param token_id Token ID to check
     * @return True if it's a timestamp token
     */
    bool is_timestamp_token(int token_id) const;

    /**
     * Convert timestamp token to seconds
     * @param token_id Timestamp token ID
     * @return Time in seconds
     */
    float timestamp_to_seconds(int token_id) const;

    /**
     * Convert seconds to timestamp token
     * @param seconds Time in seconds
     * @return Timestamp token ID
     */
    int seconds_to_timestamp(float seconds) const;

    /**
     * Get vocabulary size
     * @return Total number of tokens in vocabulary
     */
    size_t vocab_size() const { return vocab_to_id_.size(); }

    /**
     * Check if tokenizer supports multiple languages
     * @return True if multilingual
     */
    bool is_multilingual() const { return multilingual_; }

    /**
     * Split tokens into words (for word-level timestamps)
     * @param tokens Input tokens
     * @return Pair of (words, word_tokens)
     */
    std::pair<std::vector<std::string>, std::vector<std::vector<int>>>
    split_to_word_tokens(const std::vector<int>& tokens) const;

private:
    // BPE helper struct
    struct PairHash {
        size_t operator()(const std::pair<std::string, std::string>& p) const {
            return std::hash<std::string>{}(p.first) ^ (std::hash<std::string>{}(p.second) << 1);
        }
    };

    // Vocabulary mappings
    std::unordered_map<std::string, int> vocab_to_id_;
    std::unordered_map<int, std::string> id_to_vocab_;

    // BPE merges
    std::vector<std::pair<std::string, std::string>> bpe_merges_;
    std::unordered_map<std::pair<std::string, std::string>, int, PairHash> merge_ranks_;

    // Language support
    bool multilingual_;
    std::unordered_map<std::string, int> language_tokens_;

    // Non-speech tokens cache
    mutable std::optional<std::vector<int>> non_speech_tokens_cache_;

    // Helper methods
    void initialize_special_tokens();
    void initialize_language_tokens();
    std::vector<std::string> bpe_encode(const std::string& text) const;
    std::string normalize_text(const std::string& text) const;
    std::vector<std::string> tokenize_text(const std::string& text) const;
    std::pair<std::string, std::string> get_pairs(const std::vector<std::string>& word) const;
    std::vector<std::string> apply_bpe(const std::vector<std::string>& tokens) const;
};

/**
 * Convenience wrapper that matches the existing Tokenizer interface
 */
class TokenizerWrapper {
public:
    TokenizerWrapper(bool multilingual = true,
                    const std::string& language = "en",
                    const std::string& task = "transcribe");

    // Interface matching the existing tokenizer
    int get_transcribe() const;
    int get_translate() const;
    int get_sot() const;
    int get_sot_lm() const;
    int get_sot_prev() const;
    int get_eot() const;
    int get_no_timestamps() const;
    int get_timestamp_begin() const;
    std::vector<int> get_sot_sequence() const;
    std::vector<int> get_non_speech_tokens() const;

    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& tokens) const;

    std::pair<std::vector<std::string>, std::vector<std::vector<int>>>
    split_to_word_tokens(const std::vector<int>& tokens) const;

    bool is_multilingual() const;

private:
    std::unique_ptr<WhisperTokenizer> tokenizer_;
    std::string language_;
    std::string task_;
};

} // namespace whisper

#endif // WHISPER_TOKENIZER_H