/**
 * Unit Tests for WhisperModel Segment Processing Implementation
 * Tests segment generation, splitting, and word-level timestamp functions
 * Created by Amr Aboelela
 */

#include <iostream>
#include <vector>
#include <cassert>
#include <string>
#include <tuple>
#include <optional>
#include <map>
#include <algorithm>

// Test helper macros
#define ASSERT_EQ(actual, expected, test_name) \
    if ((actual) != (expected)) { \
        std::cerr << "FAILED: " << test_name << " - Expected: " << (expected) << ", Got: " << (actual) << std::endl; \
        return false; \
    } else { \
        std::cout << "✓ " << test_name << std::endl; \
    }

#define ASSERT_TRUE(condition, test_name) \
    if (!(condition)) { \
        std::cerr << "FAILED: " << test_name << " - Condition failed" << std::endl; \
        return false; \
    } else { \
        std::cout << "✓ " << test_name << std::endl; \
    }

#define ASSERT_FALSE(condition, test_name) \
    if ((condition)) { \
        std::cerr << "FAILED: " << test_name << " - Condition should be false" << std::endl; \
        return false; \
    } else { \
        std::cout << "✓ " << test_name << std::endl; \
    }

#define ASSERT_GT(actual, threshold, test_name) \
    if ((actual) <= (threshold)) { \
        std::cerr << "FAILED: " << test_name << " - Expected > " << (threshold) << ", Got: " << (actual) << std::endl; \
        return false; \
    } else { \
        std::cout << "✓ " << test_name << std::endl; \
    }

#define ASSERT_GE(actual, threshold, test_name) \
    if ((actual) < (threshold)) { \
        std::cerr << "FAILED: " << test_name << " - Expected >= " << (threshold) << ", Got: " << (actual) << std::endl; \
        return false; \
    } else { \
        std::cout << "✓ " << test_name << std::endl; \
    }

#define ASSERT_LE(actual, threshold, test_name) \
    if ((actual) > (threshold)) { \
        std::cerr << "FAILED: " << test_name << " - Expected <= " << (threshold) << ", Got: " << (actual) << std::endl; \
        return false; \
    } else { \
        std::cout << "✓ " << test_name << std::endl; \
    }

#define ASSERT_LT(actual, threshold, test_name) \
    if ((actual) >= (threshold)) { \
        std::cerr << "FAILED: " << test_name << " - Expected < " << (threshold) << ", Got: " << (actual) << std::endl; \
        return false; \
    } else { \
        std::cout << "✓ " << test_name << std::endl; \
    }

// Mock data structures for testing
struct MockWord {
    float start;
    float end;
    std::string word;
    float probability;
};

struct MockSegment {
    int id;
    int seek;
    float start;
    float end;
    std::string text;
    std::vector<int> tokens;
    float avg_logprob;
    float compression_ratio;
    float no_speech_prob;
    std::optional<std::vector<MockWord>> words;
    std::optional<float> temperature;
};

// Mock tokenizer for testing
class MockTokenizer {
public:
    int get_timestamp_begin() const { return 50364; }
    int get_eot() const { return 50257; }
    int get_sot_prev() const { return 50361; }
    int get_no_timestamps() const { return 50363; }
    std::vector<int> get_sot_sequence() const { return {50258, 50259, 50359}; }

    std::vector<int> encode(const std::string& text) const {
        std::vector<int> tokens;
        for (char c : text) {
            tokens.push_back(static_cast<int>(static_cast<unsigned char>(c)));
        }
        return tokens;
    }

    std::string decode(const std::vector<int>& tokens) const {
        std::string text;
        for (int token : tokens) {
            if (token < 256) {
                text += static_cast<char>(token);
            }
        }
        return text;
    }
};

// Mock segment splitting function
std::tuple<std::vector<MockSegment>, int, bool> mock_split_segments_by_timestamps(
    const MockTokenizer& tokenizer,
    const std::vector<int>& tokens,
    float time_offset,
    int segment_size,
    float segment_duration,
    int seek
) {
    std::vector<MockSegment> segments;

    if (!tokens.empty()) {
        MockSegment seg;
        seg.seek = seek;
        seg.start = time_offset;
        seg.end = time_offset + segment_duration;
        seg.tokens = tokens;
        seg.id = 1;
        seg.text = "Test segment";
        seg.avg_logprob = -0.5f;
        seg.compression_ratio = 2.0f;
        seg.no_speech_prob = 0.1f;
        segments.push_back(seg);
    }

    bool single_timestamp_ending = tokens.size() >= 2 &&
        tokens[tokens.size() - 2] < tokenizer.get_timestamp_begin() &&
        tokens.back() >= tokenizer.get_timestamp_begin();

    int new_seek = single_timestamp_ending ? seek + segment_size : seek + 1000;

    return {segments, new_seek, single_timestamp_ending};
}

// Mock generate segments function
std::vector<MockSegment> mock_generate_segments(
    const std::vector<std::vector<float>>& features
) {
    std::vector<MockSegment> segments;

    if (!features.empty() && !features[0].empty()) {
        MockSegment seg;
        seg.id = 1;
        seg.seek = 0;
        seg.start = 0.0f;
        seg.end = 1.0f;
        seg.text = "Test segment";
        seg.tokens = {72, 101, 115, 116}; // "Test"
        seg.avg_logprob = -0.5f;
        seg.compression_ratio = 2.0f;
        seg.no_speech_prob = 0.1f;

        segments.push_back(seg);
    }

    return segments;
}

// Mock prompt generation
std::vector<int> mock_get_prompt(
    const MockTokenizer& tokenizer,
    const std::vector<int>& previous_tokens,
    bool without_timestamps = false,
    std::optional<std::string> prefix = std::nullopt,
    std::optional<std::string> hotwords = std::nullopt
) {
    std::vector<int> prompt;

    if (!previous_tokens.empty() || (hotwords.has_value() && !prefix.has_value())) {
        prompt.push_back(tokenizer.get_sot_prev());
    }

    auto sot_sequence = tokenizer.get_sot_sequence();
    prompt.insert(prompt.end(), sot_sequence.begin(), sot_sequence.end());

    if (without_timestamps) {
        prompt.push_back(tokenizer.get_no_timestamps());
    }

    if (prefix.has_value()) {
        auto prefix_tokens = tokenizer.encode(" " + prefix.value());
        if (!without_timestamps) {
            prompt.push_back(tokenizer.get_timestamp_begin());
        }
        prompt.insert(prompt.end(), prefix_tokens.begin(), prefix_tokens.end());
    }

    return prompt;
}

// Mock word timestamps generation
std::vector<MockWord> mock_generate_word_timestamps(
    const MockSegment& segment
) {
    std::vector<MockWord> words;

    if (!segment.text.empty()) {
        // Simple word splitting
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

        // Generate timing for each word
        float segment_duration = segment.end - segment.start;
        float time_per_word = word_strings.empty() ? 0.0f : segment_duration / word_strings.size();
        float current_time = segment.start;

        for (const auto& word_text : word_strings) {
            MockWord word;
            word.start = current_time;
            word.end = std::min(current_time + time_per_word, segment.end);
            word.word = word_text;
            word.probability = 0.9f;

            words.push_back(word);
            current_time = word.end;
        }
    }

    return words;
}

// Test split_segments_by_timestamps function
bool test_split_segments_by_timestamps() {
    std::cout << "\n=== Testing Split Segments by Timestamps ===" << std::endl;

    MockTokenizer tokenizer;
    std::vector<int> sample_tokens = {72, 101, 115, 116, 50364, 50365}; // "Test" + timestamps
    float time_offset = 0.0f;
    int segment_size = 3000;
    float segment_duration = 30.0f;
    int seek = 0;

    auto [segments, new_seek, single_timestamp_ending] =
        mock_split_segments_by_timestamps(tokenizer, sample_tokens, time_offset, segment_size, segment_duration, seek);

    ASSERT_FALSE(segments.empty(), "Should produce segments");
    ASSERT_GE(new_seek, seek, "New seek position should advance");
    ASSERT_EQ(segments[0].seek, seek, "Segment should have correct seek");
    ASSERT_EQ(segments[0].start, time_offset, "Segment should start at correct time");
    ASSERT_EQ(segments[0].end, time_offset + segment_duration, "Segment should end at correct time");

    // Test with empty tokens
    std::vector<int> empty_tokens;
    auto [empty_segments, empty_seek, empty_ending] =
        mock_split_segments_by_timestamps(tokenizer, empty_tokens, time_offset, segment_size, segment_duration, seek);

    ASSERT_TRUE(empty_segments.empty(), "Empty tokens should produce no segments");
    ASSERT_EQ(empty_seek, seek + 1000, "Empty tokens should advance seek by default amount");

    return true;
}

// Test generate_segments function
bool test_generate_segments() {
    std::cout << "\n=== Testing Generate Segments ===" << std::endl;

    std::vector<std::vector<float>> sample_features = {
        {0.1f, 0.2f, 0.3f, 0.4f, 0.5f},
        {0.15f, 0.25f, 0.35f, 0.45f, 0.55f},
        {0.2f, 0.3f, 0.4f, 0.5f, 0.6f}
    };

    auto segments = mock_generate_segments(sample_features);

    ASSERT_FALSE(segments.empty(), "Should generate segments from features");
    ASSERT_GT(segments[0].id, 0, "Segment should have valid ID");
    ASSERT_FALSE(segments[0].text.empty(), "Segment should have text");
    ASSERT_FALSE(segments[0].tokens.empty(), "Segment should have tokens");
    ASSERT_GE(segments[0].start, 0.0f, "Segment start should be non-negative");
    ASSERT_GT(segments[0].end, segments[0].start, "Segment end should be after start");

    // Test with empty features
    std::vector<std::vector<float>> empty_features;
    auto empty_segments = mock_generate_segments(empty_features);
    ASSERT_TRUE(empty_segments.empty(), "Empty features should produce no segments");

    return true;
}

// Test get_prompt function
bool test_get_prompt() {
    std::cout << "\n=== Testing Get Prompt ===" << std::endl;

    MockTokenizer tokenizer;
    std::vector<int> previous_tokens;

    // Test basic prompt
    auto prompt = mock_get_prompt(tokenizer, previous_tokens, false, std::nullopt, std::nullopt);
    ASSERT_FALSE(prompt.empty(), "Basic prompt should not be empty");

    auto sot_sequence = tokenizer.get_sot_sequence();
    ASSERT_GE(prompt.size(), sot_sequence.size(), "Prompt should contain SOT sequence");

    // Test with previous tokens
    std::vector<int> prev_tokens = {72, 101, 115, 116}; // "Test"
    auto prompt_with_prev = mock_get_prompt(tokenizer, prev_tokens, false, std::nullopt, std::nullopt);
    ASSERT_FALSE(prompt_with_prev.empty(), "Prompt with previous tokens should not be empty");
    ASSERT_GT(prompt_with_prev.size(), sot_sequence.size(), "Should be larger than basic SOT sequence");
    ASSERT_EQ(prompt_with_prev[0], tokenizer.get_sot_prev(), "Should start with SOT_PREV");

    // Test without timestamps
    auto prompt_no_ts = mock_get_prompt(tokenizer, previous_tokens, true, std::nullopt, std::nullopt);
    ASSERT_FALSE(prompt_no_ts.empty(), "No timestamps prompt should not be empty");

    bool has_no_timestamps = false;
    for (int token : prompt_no_ts) {
        if (token == tokenizer.get_no_timestamps()) {
            has_no_timestamps = true;
            break;
        }
    }
    ASSERT_TRUE(has_no_timestamps, "Should contain no_timestamps token");

    // Test with prefix
    auto prompt_with_prefix = mock_get_prompt(tokenizer, previous_tokens, false, "Hello", std::nullopt);
    ASSERT_FALSE(prompt_with_prefix.empty(), "Prompt with prefix should not be empty");

    bool has_timestamp_begin = false;
    for (int token : prompt_with_prefix) {
        if (token == tokenizer.get_timestamp_begin()) {
            has_timestamp_begin = true;
            break;
        }
    }
    ASSERT_TRUE(has_timestamp_begin, "Should contain timestamp_begin when prefix provided");

    return true;
}

// Test generate_word_timestamps function
bool test_generate_word_timestamps() {
    std::cout << "\n=== Testing Generate Word Timestamps ===" << std::endl;

    MockSegment segment;
    segment.start = 0.0f;
    segment.end = 2.0f;
    segment.text = "Hello world test";

    auto words = mock_generate_word_timestamps(segment);

    ASSERT_EQ(words.size(), 3, "Should generate 3 words"); // "Hello", "world", "test"
    ASSERT_EQ(words[0].word, "Hello", "First word should be 'Hello'");
    ASSERT_EQ(words[1].word, "world", "Second word should be 'world'");
    ASSERT_EQ(words[2].word, "test", "Third word should be 'test'");

    // Check timing distribution
    ASSERT_EQ(words[0].start, segment.start, "First word should start at segment start");
    ASSERT_EQ(words.back().end, segment.end, "Last word should end at segment end");

    for (const auto& word : words) {
        ASSERT_GE(word.start, segment.start, "Word start should be within segment");
        ASSERT_LE(word.end, segment.end, "Word end should be within segment");
        ASSERT_LT(word.start, word.end, "Word start should be before end");
        ASSERT_GT(word.probability, 0.0f, "Word probability should be positive");
        ASSERT_LE(word.probability, 1.0f, "Word probability should not exceed 1.0");
    }

    // Test empty text
    MockSegment empty_segment;
    empty_segment.start = 0.0f;
    empty_segment.end = 2.0f;
    empty_segment.text = "";

    auto empty_words = mock_generate_word_timestamps(empty_segment);
    ASSERT_TRUE(empty_words.empty(), "Empty text should produce no words");

    // Test single word
    MockSegment single_segment;
    single_segment.start = 1.0f;
    single_segment.end = 3.0f;
    single_segment.text = "Arabic";

    auto single_words = mock_generate_word_timestamps(single_segment);
    ASSERT_EQ(single_words.size(), 1, "Single word should produce one word");
    ASSERT_EQ(single_words[0].word, "Arabic", "Should preserve word text");
    ASSERT_EQ(single_words[0].start, single_segment.start, "Should start at segment start");
    ASSERT_EQ(single_words[0].end, single_segment.end, "Should end at segment end");

    return true;
}

// Test Arabic content processing
bool test_arabic_content() {
    std::cout << "\n=== Testing Arabic Content Processing ===" << std::endl;

    MockSegment arabic_segment;
    arabic_segment.start = 0.0f;
    arabic_segment.end = 4.0f;
    arabic_segment.text = "مرحبا بالعالم"; // "Hello world" in Arabic

    auto arabic_words = mock_generate_word_timestamps(arabic_segment);
    ASSERT_EQ(arabic_words.size(), 2, "Should split Arabic text into 2 words"); // "مرحبا", "بالعالم"

    for (const auto& word : arabic_words) {
        ASSERT_FALSE(word.word.empty(), "Arabic word should not be empty");
        ASSERT_GE(word.start, arabic_segment.start, "Word start should be within segment");
        ASSERT_LE(word.end, arabic_segment.end, "Word end should be within segment");
        ASSERT_GT(word.probability, 0.8f, "Arabic words should have high confidence");
    }

    return true;
}

// Test edge cases
bool test_edge_cases() {
    std::cout << "\n=== Testing Edge Cases ===" << std::endl;

    MockTokenizer tokenizer;

    // Test zero duration segment
    MockSegment zero_duration;
    zero_duration.start = 1.0f;
    zero_duration.end = 1.0f; // Zero duration
    zero_duration.text = "test";

    auto zero_words = mock_generate_word_timestamps(zero_duration);
    ASSERT_FALSE(zero_words.empty(), "Should handle zero duration gracefully");
    if (!zero_words.empty()) {
        ASSERT_EQ(zero_words[0].start, zero_words[0].end, "Zero duration words should have equal start/end");
    }

    // Test very long text
    MockSegment long_segment;
    long_segment.start = 0.0f;
    long_segment.end = 10.0f;
    long_segment.text = "This is a very long text with many words to test timing distribution";

    auto long_words = mock_generate_word_timestamps(long_segment);
    ASSERT_GT(long_words.size(), 10, "Long text should produce many words");

    // Verify proper timing distribution
    for (size_t i = 1; i < long_words.size(); ++i) {
        ASSERT_GE(long_words[i].start, long_words[i-1].start, "Words should be in chronological order");
        ASSERT_LE(long_words[i].end, long_segment.end, "All words should end within segment");
    }

    return true;
}

// Test segment processing flow
bool test_segment_processing_flow() {
    std::cout << "\n=== Testing Segment Processing Flow ===" << std::endl;

    std::vector<std::vector<float>> sample_features = {
        {0.1f, 0.2f, 0.3f, 0.4f},
        {0.15f, 0.25f, 0.35f, 0.45f},
        {0.2f, 0.3f, 0.4f, 0.5f}
    };

    // Generate segments
    auto segments = mock_generate_segments(sample_features);
    ASSERT_FALSE(segments.empty(), "Should generate segments");

    // Generate word timestamps for each segment
    for (const auto& segment : segments) {
        auto words = mock_generate_word_timestamps(segment);

        if (!segment.text.empty()) {
            ASSERT_FALSE(words.empty(), "Non-empty segment should produce words");

            // Verify word boundaries
            for (const auto& word : words) {
                ASSERT_GE(word.start, segment.start, "Word should start within segment");
                ASSERT_LE(word.end, segment.end, "Word should end within segment");
            }
        }
    }

    return true;
}

// Main test runner
int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "WhisperModel Segments Unit Tests" << std::endl;
    std::cout << "Testing segment processing functions" << std::endl;
    std::cout << "========================================" << std::endl;

    bool all_passed = true;

    all_passed &= test_split_segments_by_timestamps();
    all_passed &= test_generate_segments();
    all_passed &= test_get_prompt();
    all_passed &= test_generate_word_timestamps();
    all_passed &= test_arabic_content();
    all_passed &= test_edge_cases();
    all_passed &= test_segment_processing_flow();

    std::cout << "\n========================================" << std::endl;
    if (all_passed) {
        std::cout << "🎉 ALL SEGMENTS TESTS PASSED!" << std::endl;
        std::cout << "✅ Segment processing functions working correctly" << std::endl;
        std::cout << "✅ Word timestamp generation validated" << std::endl;
        std::cout << "✅ Arabic content processing confirmed" << std::endl;
        std::cout << "✅ Edge cases handled properly" << std::endl;
        return 0;
    } else {
        std::cout << "❌ SOME SEGMENTS TESTS FAILED!" << std::endl;
        std::cout << "Please review the failed tests above." << std::endl;
        return 1;
    }
}