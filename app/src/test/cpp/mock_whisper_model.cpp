#ifdef MOCK_CTRANSLATE2

#include "whisper_model.h"
#include "feature_extractor.h"
#include "tokenizer.h"
#include <iostream>
#include <memory>
#include <cmath>

// Mock WhisperModel implementation for testing
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
    std::cout << "Mock WhisperModel initialized with path: " << model_size_or_path << std::endl;

    // Initialize mock model
    model = nullptr; // Mock doesn't use real CTranslate2 model
    hf_tokenizer = nullptr;

    // Initialize feature extractor
    feature_extractor = FeatureExtractor();

    // Set mock parameters
    input_stride = 2;
    num_samples_per_token = feature_extractor.hop_length * input_stride;
    frames_per_second = feature_extractor.sampling_rate() / feature_extractor.hop_length;
    tokens_per_second = feature_extractor.sampling_rate() / num_samples_per_token;
    time_precision = 0.02;
    max_length = 448;
}

std::vector<std::string> WhisperModel::supported_languages() const {
    // Mock multilingual support
    return _LANGUAGE_CODES;
}

std::map<std::string, std::string> WhisperModel::get_feature_kwargs(
    const std::string &model_path,
    const std::optional<std::string> &preprocessor_bytes
) {
    std::cout << "Mock get_feature_kwargs called for path: " << model_path << std::endl;
    return std::map<std::string, std::string>{};
}

std::tuple<std::vector<Segment>, TranscriptionInfo> WhisperModel::transcribe(
    const std::vector<float> &audio,
    const std::optional<std::string> &language,
    bool multilingual
) {
    std::cout << "Mock transcribe called with " << audio.size() << " audio samples" << std::endl;

    // Create mock transcription results
    std::vector<Segment> segments;

    // Create a mock segment
    Segment segment;
    segment.id = 0;
    segment.seek = 0;
    segment.start = 0.0f;
    segment.end = static_cast<float>(audio.size()) / 16000.0f; // Duration in seconds
    segment.text = "مرحبا بك في اختبار الهمس العربي"; // "Welcome to Arabic Whisper test"
    segment.tokens = {50258, 50272, 50359, 15496, 1002, 50257}; // Mock tokens
    segment.temperature = 0.0f;
    segment.avg_logprob = -0.25f;
    segment.compression_ratio = 2.1f;
    segment.no_speech_prob = 0.02f;
    segment.words = std::nullopt;

    segments.push_back(segment);

    // Create mock transcription info
    TranscriptionInfo info;
    info.language = language.value_or("ar");
    info.language_probability = 0.95f;
    info.duration = static_cast<float>(audio.size()) / 16000.0f;
    info.all_language_probs = std::vector<std::pair<std::string, float>>{
        {"ar", 0.95f}, {"en", 0.03f}, {"fr", 0.02f}
    };

    std::cout << "Mock transcription completed: " << segments.size() << " segments" << std::endl;

    return {segments, info};
}

std::vector<Word> WhisperModel::generate_word_timestamps(
    const Segment& segment,
    Tokenizer& tokenizer
) {
    std::cout << "Mock generate_word_timestamps called" << std::endl;

    std::vector<Word> words;

    // Mock word-level timestamps
    Word word1{0.0f, 1.0f, "مرحبا", 0.95f};
    Word word2{1.0f, 2.0f, " بك", 0.92f};
    Word word3{2.0f, 3.0f, " في", 0.94f};

    words = {word1, word2, word3};

    return words;
}

// Additional mock implementations for completeness...
std::tuple<std::vector<Segment>, int, bool> WhisperModel::split_segments_by_timestamps(
    Tokenizer &tokenizer,
    const std::vector<int> &tokens,
    float time_offset,
    int segment_size,
    float segment_duration,
    int seek
) {
    std::cout << "Mock split_segments_by_timestamps called" << std::endl;

    std::vector<Segment> segments;
    Segment segment;
    segment.seek = seek;
    segment.start = time_offset;
    segment.end = time_offset + segment_duration;
    segment.tokens = tokens;
    segments.push_back(segment);

    return {segments, seek + segment_size, false};
}

std::vector<Segment> WhisperModel::generate_segments(
    const std::vector<std::vector<float>> &features,
    Tokenizer &tokenizer,
    const TranscriptionOptions &options
) {
    std::cout << "Mock generate_segments called" << std::endl;

    std::vector<Segment> segments;

    // Create mock segment
    Segment segment;
    segment.id = 0;
    segment.seek = 0;
    segment.start = 0.0f;
    segment.end = 30.0f;
    segment.text = "Mock Arabic transcription segment";
    segment.tokens = {50258, 50272, 50359, 15496, 1002, 50257};
    segment.temperature = 0.0f;
    segment.avg_logprob = -0.25f;
    segment.compression_ratio = 2.1f;
    segment.no_speech_prob = 0.02f;
    segment.words = std::nullopt;

    segments.push_back(segment);

    return segments;
}

// Mock encode function - this won't actually link to CTranslate2
ctranslate2::StorageView WhisperModel::encode(const std::vector<std::vector<float>> &features) {
    std::cout << "Mock encode called" << std::endl;

    // Create mock storage view
    ctranslate2::Shape shape = {1, 1500, 1280}; // Mock encoder output shape
    std::vector<float> data(1500 * 1280, 0.1f); // Mock data

    return ctranslate2::StorageView(shape, data);
}

std::tuple<std::vector<int>, float, float, float>
WhisperModel::generate_with_fallback(
    const ctranslate2::StorageView &encoder_output,
    const std::vector<int> &prompt,
    Tokenizer &tokenizer,
    const TranscriptionOptions &options
) {
    std::cout << "Mock generate_with_fallback called" << std::endl;

    // Mock generation result
    std::vector<int> tokens = {50258, 50272, 50359, 15496, 1002, 50257};
    float avg_logprob = -0.25f;
    float temperature = 0.0f;
    float compression_ratio = 2.1f;

    return {tokens, avg_logprob, temperature, compression_ratio};
}

std::vector<int> WhisperModel::get_prompt(
    Tokenizer &tokenizer,
    const std::vector<int> &previous_tokens,
    bool without_timestamps,
    std::optional<std::string> prefix,
    std::optional<std::string> hotwords
) {
    std::cout << "Mock get_prompt called" << std::endl;

    // Mock prompt generation
    std::vector<int> prompt = {50258, 50272, 50359}; // SOT, Arabic, transcribe
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
    std::cout << "Mock add_word_timestamps called" << std::endl;
    return last_speech_timestamp + 1.0f; // Mock return
}

std::vector<std::vector<std::map<std::string, std::any>>>
WhisperModel::find_alignment(
    Tokenizer &tokenizer,
    const std::vector<std::vector<int>> &text_tokens,
    const ctranslate2::StorageView &encoder_output,
    int num_frames,
    int median_filter_width
) {
    std::cout << "Mock find_alignment called" << std::endl;
    return {}; // Mock empty result
}

std::tuple<std::string, float, std::vector<std::pair<std::string, float>>>
WhisperModel::detect_language(
    const std::vector<float> *audio,
    const std::vector<std::vector<float>> *features,
    int language_detection_segments,
    float language_detection_threshold
) {
    std::cout << "Mock detect_language called" << std::endl;

    // Mock Arabic language detection
    std::string language = "ar";
    float probability = 0.95f;
    std::vector<std::pair<std::string, float>> all_probs = {
        {"ar", 0.95f}, {"en", 0.03f}, {"fr", 0.02f}
    };

    return {language, probability, all_probs};
}

#endif // MOCK_CTRANSLATE2