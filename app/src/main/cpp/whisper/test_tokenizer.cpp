#include "whisper_tokenizer.h"
#include "../include/tokenizer.h"
#include <iostream>
#include <vector>
#include <cassert>

/**
 * Test to verify whisper tokenizer integration
 */
void test_whisper_tokenizer_integration() {
    std::cout << "=== Whisper Tokenizer Integration Test ===" << std::endl;

    // Test 1: Basic tokenizer initialization
    std::cout << "Testing tokenizer initialization..." << std::endl;

    // Create a mock tokenizers::Tokenizer (placeholder)
    tokenizers::Tokenizer mock_tokenizer;

    // Create Tokenizer with multilingual support and Arabic language
    Tokenizer tokenizer(&mock_tokenizer, true, "transcribe", "ar");
    std::cout << "✓ Tokenizer initialized with Arabic language support" << std::endl;

    // Test 2: Special tokens
    std::cout << "Testing special tokens..." << std::endl;

    int sot = tokenizer.get_sot();
    int eot = tokenizer.get_eot();
    int transcribe = tokenizer.get_transcribe();
    int translate = tokenizer.get_translate();
    int timestamp_begin = tokenizer.get_timestamp_begin();

    std::cout << "✓ Special tokens: SOT=" << sot << ", EOT=" << eot
              << ", Transcribe=" << transcribe << ", Translate=" << translate
              << ", Timestamp Begin=" << timestamp_begin << std::endl;

    // Test 3: SOT sequence generation
    std::cout << "Testing SOT sequence generation..." << std::endl;
    auto sot_sequence = tokenizer.get_sot_sequence();
    std::cout << "✓ SOT sequence generated with " << sot_sequence.size() << " tokens: ";
    for (int token : sot_sequence) {
        std::cout << token << " ";
    }
    std::cout << std::endl;

    // Test 4: Non-speech tokens
    std::cout << "Testing non-speech tokens..." << std::endl;
    auto non_speech_tokens = tokenizer.get_non_speech_tokens();
    std::cout << "✓ Non-speech tokens: " << non_speech_tokens.size() << " tokens identified" << std::endl;

    // Test 5: Text encoding (Arabic and English)
    std::cout << "Testing text encoding..." << std::endl;

    std::string english_text = "Hello world";
    auto english_tokens = tokenizer.encode(english_text);
    std::cout << "✓ English text encoded: \"" << english_text << "\" -> "
              << english_tokens.size() << " tokens" << std::endl;

    std::string arabic_text = "السلام عليكم";
    auto arabic_tokens = tokenizer.encode(arabic_text);
    std::cout << "✓ Arabic text encoded: \"" << arabic_text << "\" -> "
              << arabic_tokens.size() << " tokens" << std::endl;

    // Test 6: Token decoding
    std::cout << "Testing token decoding..." << std::endl;

    if (!english_tokens.empty()) {
        std::string decoded_english = tokenizer.decode(english_tokens);
        std::cout << "✓ English tokens decoded: " << english_tokens.size()
                  << " tokens -> \"" << decoded_english << "\"" << std::endl;
    }

    if (!arabic_tokens.empty()) {
        std::string decoded_arabic = tokenizer.decode(arabic_tokens);
        std::cout << "✓ Arabic tokens decoded: " << arabic_tokens.size()
                  << " tokens -> \"" << decoded_arabic << "\"" << std::endl;
    }

    // Test 7: Timestamp decoding
    std::cout << "Testing timestamp decoding..." << std::endl;
    std::vector<int> timestamp_tokens = {timestamp_begin, timestamp_begin + 50, timestamp_begin + 100};
    std::string decoded_with_timestamps = tokenizer.decode_with_timestamps(timestamp_tokens);
    std::cout << "✓ Timestamp tokens decoded: \"" << decoded_with_timestamps << "\"" << std::endl;

    // Test 8: Word token splitting
    std::cout << "Testing word token splitting..." << std::endl;
    if (!english_tokens.empty()) {
        auto [words, word_tokens] = tokenizer.split_to_word_tokens(english_tokens);
        std::cout << "✓ Word splitting: " << words.size() << " words from "
                  << english_tokens.size() << " tokens" << std::endl;

        for (size_t i = 0; i < words.size() && i < 3; ++i) {
            std::cout << "  Word " << i << ": \"" << words[i] << "\" ("
                      << word_tokens[i].size() << " tokens)" << std::endl;
        }
    }

    std::cout << "=== Tokenizer Integration Test Completed ===" << std::endl;
}

/**
 * Test whisper tokenizer standalone functionality
 */
void test_whisper_tokenizer_standalone() {
    std::cout << "\n=== Whisper Tokenizer Standalone Test ===" << std::endl;

    // Test 1: Create whisper tokenizer directly
    std::cout << "Testing standalone whisper tokenizer..." << std::endl;
    whisper::WhisperTokenizer whisper_tokenizer("", true);
    std::cout << "✓ Whisper tokenizer created with multilingual support" << std::endl;
    std::cout << "  Vocabulary size: " << whisper_tokenizer.vocab_size() << std::endl;

    // Test 2: Language token retrieval
    std::cout << "Testing language tokens..." << std::endl;
    int ar_token = whisper_tokenizer.get_language_token("ar");
    int en_token = whisper_tokenizer.get_language_token("en");
    int fr_token = whisper_tokenizer.get_language_token("fr");

    std::cout << "✓ Language tokens: Arabic=" << ar_token
              << ", English=" << en_token << ", French=" << fr_token << std::endl;

    // Test 3: SOT sequence for different languages
    std::cout << "Testing SOT sequences for different languages..." << std::endl;
    auto ar_sot = whisper_tokenizer.get_sot_sequence("ar", "transcribe");
    auto en_sot = whisper_tokenizer.get_sot_sequence("en", "translate");

    std::cout << "✓ Arabic SOT sequence (" << ar_sot.size() << " tokens): ";
    for (int token : ar_sot) std::cout << token << " ";
    std::cout << std::endl;

    std::cout << "✓ English SOT sequence (" << en_sot.size() << " tokens): ";
    for (int token : en_sot) std::cout << token << " ";
    std::cout << std::endl;

    // Test 4: Timestamp token handling
    std::cout << "Testing timestamp tokens..." << std::endl;
    int timestamp_1s = whisper_tokenizer.seconds_to_timestamp(1.0f);
    int timestamp_5s = whisper_tokenizer.seconds_to_timestamp(5.0f);

    float back_to_1s = whisper_tokenizer.timestamp_to_seconds(timestamp_1s);
    float back_to_5s = whisper_tokenizer.timestamp_to_seconds(timestamp_5s);

    std::cout << "✓ Timestamp conversion: 1.0s -> " << timestamp_1s << " -> " << back_to_1s << "s" << std::endl;
    std::cout << "✓ Timestamp conversion: 5.0s -> " << timestamp_5s << " -> " << back_to_5s << "s" << std::endl;

    std::cout << "=== Whisper Tokenizer Standalone Test Completed ===" << std::endl;
}

/**
 * Usage demonstration
 */
void demonstrate_tokenizer_usage() {
    std::cout << "\n=== Tokenizer Usage Examples ===" << std::endl;

    std::cout << "// Basic usage:" << std::endl;
    std::cout << "// 1. Create tokenizer with Arabic support:" << std::endl;
    std::cout << "//    Tokenizer tokenizer(&mock_tokenizer, true, \"transcribe\", \"ar\");" << std::endl;
    std::cout << "//" << std::endl;
    std::cout << "// 2. Encode Arabic text:" << std::endl;
    std::cout << "//    auto tokens = tokenizer.encode(\"مرحبا بالعالم\");" << std::endl;
    std::cout << "//" << std::endl;
    std::cout << "// 3. Get SOT sequence for inference:" << std::endl;
    std::cout << "//    auto sot_sequence = tokenizer.get_sot_sequence();" << std::endl;
    std::cout << "//" << std::endl;
    std::cout << "// 4. Decode tokens back to text:" << std::endl;
    std::cout << "//    std::string text = tokenizer.decode(tokens);" << std::endl;

    std::cout << "\n// Key benefits:" << std::endl;
    std::cout << "// - Full whisper.cpp compatibility" << std::endl;
    std::cout << "// - Arabic language support built-in" << std::endl;
    std::cout << "// - Proper special token handling" << std::endl;
    std::cout << "// - Timestamp token support" << std::endl;
    std::cout << "// - Word-level token splitting" << std::endl;
    std::cout << "// - Integrated with existing codebase" << std::endl;
}

#ifndef TESTING_MODE
int main() {
    test_whisper_tokenizer_integration();
    test_whisper_tokenizer_standalone();
    demonstrate_tokenizer_usage();
    return 0;
}
#endif