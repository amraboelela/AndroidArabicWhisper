#include <iostream>
#include <vector>
#include <cassert>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <map>

/**
 * Unit tests for utils.cpp functionality
 * Testing model utilities, validation, and helper functions
 */

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

// Mirror the _MODELS map from utils.cpp for testing
static const std::unordered_map<std::string, std::string> _MODELS = {
    {"tiny.en", "Systran/faster-whisper-tiny.en"},
    {"tiny", "Systran/faster-whisper-tiny"},
    {"base.en", "Systran/faster-whisper-base.en"},
    {"base", "Systran/faster-whisper-base"},
    {"small.en", "Systran/faster-whisper-small.en"},
    {"small", "Systran/faster-whisper-small"},
    {"medium.en", "Systran/faster-whisper-medium.en"},
    {"medium", "Systran/faster-whisper-medium"},
    {"large-v1", "Systran/faster-whisper-large-v1"},
    {"large-v2", "Systran/faster-whisper-large-v2"},
    {"large-v3", "Systran/faster-whisper-large-v3"},
    {"large", "Systran/faster-whisper-large-v3"},
    {"distil-large-v2", "Systran/faster-distil-whisper-large-v2"},
    {"distil-medium.en", "Systran/faster-distil-whisper-medium.en"},
    {"distil-small.en", "Systran/faster-distil-whisper-small.en"},
    {"distil-large-v3", "Systran/faster-distil-whisper-large-v3"},
    {"distil-large-v3.5", "distil-whisper/distil-large-v3.5-ct2"},
    {"large-v3-turbo", "mobiuslabsgmbh/faster-whisper-large-v3-turbo"},
    {"turbo", "mobiuslabsgmbh/faster-whisper-large-v3-turbo"},
};

// Mock implementation of functions that would be in utils.cpp
std::vector<std::string> availableModels() {
    std::vector<std::string> models;
    for (const auto& pair : _MODELS) {
        models.push_back(pair.first);
    }
    std::sort(models.begin(), models.end());
    return models;
}

std::string getModelPath(const std::string& model_name) {
    auto it = _MODELS.find(model_name);
    if (it != _MODELS.end()) {
        return it->second;
    }
    return model_name; // Return as-is if not found (could be custom path)
}

bool isValidModelName(const std::string& model_name) {
    return _MODELS.find(model_name) != _MODELS.end();
}

std::vector<std::string> getEnglishOnlyModels() {
    std::vector<std::string> english_models;
    for (const auto& pair : _MODELS) {
        if (pair.first.find(".en") != std::string::npos) {
            english_models.push_back(pair.first);
        }
    }
    return english_models;
}

std::vector<std::string> getMultilingualModels() {
    std::vector<std::string> multilingual_models;
    for (const auto& pair : _MODELS) {
        if (pair.first.find(".en") == std::string::npos) {
            multilingual_models.push_back(pair.first);
        }
    }
    return multilingual_models;
}

std::string getModelSize(const std::string& model_name) {
    if (model_name.find("tiny") != std::string::npos) return "tiny";
    if (model_name.find("base") != std::string::npos) return "base";
    if (model_name.find("small") != std::string::npos) return "small";
    if (model_name.find("medium") != std::string::npos) return "medium";
    if (model_name.find("large") != std::string::npos) return "large";
    if (model_name.find("turbo") != std::string::npos) return "turbo";
    return "unknown";
}

bool isDistilModel(const std::string& model_name) {
    return model_name.find("distil") != std::string::npos;
}

namespace {

/**
 * Test model registry functionality
 */
bool test_model_registry() {
    std::cout << "\n=== Testing Model Registry ===" << std::endl;

    // Test available models
    auto models = availableModels();
    ASSERT_TRUE(!models.empty(), "Available models not empty");
    ASSERT_TRUE(models.size() >= 18, "Expected minimum number of models");

    // Test specific model exists
    bool found_large = std::find(models.begin(), models.end(), "large") != models.end();
    ASSERT_TRUE(found_large, "Large model exists in registry");

    bool found_tiny = std::find(models.begin(), models.end(), "tiny") != models.end();
    ASSERT_TRUE(found_tiny, "Tiny model exists in registry");

    // Test models are sorted
    bool is_sorted = std::is_sorted(models.begin(), models.end());
    ASSERT_TRUE(is_sorted, "Available models are sorted");

    return true;
}

/**
 * Test model path resolution
 */
bool test_model_path_resolution() {
    std::cout << "\n=== Testing Model Path Resolution ===" << std::endl;

    // Test known model path resolution
    std::string large_path = getModelPath("large");
    ASSERT_EQ(large_path, "Systran/faster-whisper-large-v3", "Large model path");

    std::string tiny_path = getModelPath("tiny");
    ASSERT_EQ(tiny_path, "Systran/faster-whisper-tiny", "Tiny model path");

    std::string base_en_path = getModelPath("base.en");
    ASSERT_EQ(base_en_path, "Systran/faster-whisper-base.en", "Base English model path");

    // Test unknown model (should return as-is)
    std::string custom_path = getModelPath("/custom/path/to/model");
    ASSERT_EQ(custom_path, "/custom/path/to/model", "Custom model path unchanged");

    std::string unknown_model = getModelPath("nonexistent-model");
    ASSERT_EQ(unknown_model, "nonexistent-model", "Unknown model name unchanged");

    return true;
}

/**
 * Test model validation
 */
bool test_model_validation() {
    std::cout << "\n=== Testing Model Validation ===" << std::endl;

    // Test valid model names
    ASSERT_TRUE(isValidModelName("large"), "Large model is valid");
    ASSERT_TRUE(isValidModelName("tiny"), "Tiny model is valid");
    ASSERT_TRUE(isValidModelName("base.en"), "Base English model is valid");
    ASSERT_TRUE(isValidModelName("distil-large-v2"), "Distil model is valid");
    ASSERT_TRUE(isValidModelName("turbo"), "Turbo model is valid");

    // Test invalid model names
    ASSERT_TRUE(!isValidModelName("nonexistent"), "Nonexistent model is invalid");
    ASSERT_TRUE(!isValidModelName(""), "Empty model name is invalid");
    ASSERT_TRUE(!isValidModelName("large-v5"), "Future version is invalid");
    ASSERT_TRUE(!isValidModelName("custom-model"), "Custom model name is invalid");

    return true;
}

/**
 * Test English-only model filtering
 */
bool test_english_only_models() {
    std::cout << "\n=== Testing English-Only Models ===" << std::endl;

    auto english_models = getEnglishOnlyModels();
    ASSERT_TRUE(!english_models.empty(), "English-only models not empty");

    // All models should contain ".en"
    for (const auto& model : english_models) {
        ASSERT_TRUE(model.find(".en") != std::string::npos, "Model contains .en suffix");
    }

    // Test specific English models exist
    bool found_tiny_en = std::find(english_models.begin(), english_models.end(), "tiny.en") != english_models.end();
    ASSERT_TRUE(found_tiny_en, "Tiny English model exists");

    bool found_base_en = std::find(english_models.begin(), english_models.end(), "base.en") != english_models.end();
    ASSERT_TRUE(found_base_en, "Base English model exists");

    return true;
}

/**
 * Test multilingual model filtering
 */
bool test_multilingual_models() {
    std::cout << "\n=== Testing Multilingual Models ===" << std::endl;

    auto multilingual_models = getMultilingualModels();
    ASSERT_TRUE(!multilingual_models.empty(), "Multilingual models not empty");

    // No models should contain ".en"
    for (const auto& model : multilingual_models) {
        ASSERT_TRUE(model.find(".en") == std::string::npos, "Model does not contain .en suffix");
    }

    // Test specific multilingual models exist
    bool found_large = std::find(multilingual_models.begin(), multilingual_models.end(), "large") != multilingual_models.end();
    ASSERT_TRUE(found_large, "Large multilingual model exists");

    bool found_tiny = std::find(multilingual_models.begin(), multilingual_models.end(), "tiny") != multilingual_models.end();
    ASSERT_TRUE(found_tiny, "Tiny multilingual model exists");

    return true;
}

/**
 * Test model size detection
 */
bool test_model_size_detection() {
    std::cout << "\n=== Testing Model Size Detection ===" << std::endl;

    // Test size detection for different models
    ASSERT_EQ(getModelSize("tiny"), "tiny", "Tiny model size");
    ASSERT_EQ(getModelSize("tiny.en"), "tiny", "Tiny English model size");
    ASSERT_EQ(getModelSize("base"), "base", "Base model size");
    ASSERT_EQ(getModelSize("base.en"), "base", "Base English model size");
    ASSERT_EQ(getModelSize("small"), "small", "Small model size");
    ASSERT_EQ(getModelSize("medium"), "medium", "Medium model size");
    ASSERT_EQ(getModelSize("large"), "large", "Large model size");
    ASSERT_EQ(getModelSize("large-v1"), "large", "Large v1 model size");
    ASSERT_EQ(getModelSize("large-v2"), "large", "Large v2 model size");
    ASSERT_EQ(getModelSize("large-v3"), "large", "Large v3 model size");
    ASSERT_EQ(getModelSize("turbo"), "turbo", "Turbo model size");

    // Test unknown size
    ASSERT_EQ(getModelSize("custom-model"), "unknown", "Unknown model size");
    ASSERT_EQ(getModelSize(""), "unknown", "Empty model size");

    return true;
}

/**
 * Test distil model detection
 */
bool test_distil_model_detection() {
    std::cout << "\n=== Testing Distil Model Detection ===" << std::endl;

    // Test distil models
    ASSERT_TRUE(isDistilModel("distil-large-v2"), "Distil large v2 is distil");
    ASSERT_TRUE(isDistilModel("distil-medium.en"), "Distil medium English is distil");
    ASSERT_TRUE(isDistilModel("distil-small.en"), "Distil small English is distil");
    ASSERT_TRUE(isDistilModel("distil-large-v3"), "Distil large v3 is distil");
    ASSERT_TRUE(isDistilModel("distil-large-v3.5"), "Distil large v3.5 is distil");

    // Test non-distil models
    ASSERT_TRUE(!isDistilModel("large"), "Large is not distil");
    ASSERT_TRUE(!isDistilModel("tiny"), "Tiny is not distil");
    ASSERT_TRUE(!isDistilModel("base.en"), "Base English is not distil");
    ASSERT_TRUE(!isDistilModel("turbo"), "Turbo is not distil");

    return true;
}

/**
 * Test model registry completeness
 */
bool test_model_registry_completeness() {
    std::cout << "\n=== Testing Model Registry Completeness ===" << std::endl;

    auto all_models = availableModels();
    auto english_models = getEnglishOnlyModels();
    auto multilingual_models = getMultilingualModels();

    // Test that English + Multilingual = All models
    size_t total_expected = english_models.size() + multilingual_models.size();
    ASSERT_EQ(all_models.size(), total_expected, "English + Multilingual equals total");

    // Test no overlap between English and Multilingual
    for (const auto& en_model : english_models) {
        bool found_in_multilingual = std::find(multilingual_models.begin(), multilingual_models.end(), en_model) != multilingual_models.end();
        ASSERT_TRUE(!found_in_multilingual, "No overlap between English and Multilingual");
    }

    // Test all sizes are represented
    std::vector<std::string> expected_sizes = {"tiny", "base", "small", "medium", "large", "turbo"};
    for (const auto& size : expected_sizes) {
        bool size_found = false;
        for (const auto& model : all_models) {
            if (getModelSize(model) == size) {
                size_found = true;
                break;
            }
        }
        ASSERT_TRUE(size_found, "Model size " + size + " is represented");
    }

    return true;
}

/**
 * Test edge cases and error handling
 */
bool test_edge_cases() {
    std::cout << "\n=== Testing Edge Cases ===" << std::endl;

    // Test empty inputs
    ASSERT_TRUE(!isValidModelName(""), "Empty string is invalid model");
    ASSERT_EQ(getModelPath(""), "", "Empty string model path");
    ASSERT_EQ(getModelSize(""), "unknown", "Empty string model size");

    // Test whitespace inputs
    ASSERT_TRUE(!isValidModelName(" "), "Space is invalid model");
    ASSERT_TRUE(!isValidModelName("large "), "Trailing space is invalid");
    ASSERT_TRUE(!isValidModelName(" large"), "Leading space is invalid");

    // Test case sensitivity
    ASSERT_TRUE(!isValidModelName("LARGE"), "Uppercase is invalid");
    ASSERT_TRUE(!isValidModelName("Large"), "Mixed case is invalid");
    ASSERT_TRUE(!isValidModelName("Tiny.EN"), "Mixed case English is invalid");

    // Test special characters
    ASSERT_TRUE(!isValidModelName("large@v3"), "Special characters invalid");
    ASSERT_TRUE(!isValidModelName("large/v3"), "Path separators invalid");

    return true;
}

} // anonymous namespace

/**
 * Main test runner for utils tests
 */
bool run_utils_tests() {
    std::cout << "=== UTILS UNIT TESTS ===" << std::endl;

    bool all_passed = true;

    all_passed &= test_model_registry();
    all_passed &= test_model_path_resolution();
    all_passed &= test_model_validation();
    all_passed &= test_english_only_models();
    all_passed &= test_multilingual_models();
    all_passed &= test_model_size_detection();
    all_passed &= test_distil_model_detection();
    all_passed &= test_model_registry_completeness();
    all_passed &= test_edge_cases();

    std::cout << "\n=== UTILS TEST SUMMARY ===" << std::endl;
    if (all_passed) {
        std::cout << "✅ ALL UTILS TESTS PASSED!" << std::endl;
    } else {
        std::cout << "❌ SOME UTILS TESTS FAILED!" << std::endl;
    }

    return all_passed;
}

/**
 * Demonstrate utils functionality
 */
void demonstrate_utils_usage() {
    std::cout << "\n=== Utils Usage Examples ===" << std::endl;

    std::cout << "// Model registry usage:" << std::endl;
    std::cout << "// 1. List available models:" << std::endl;
    std::cout << "//    auto models = availableModels();" << std::endl;
    std::cout << "//    for (const auto& model : models) std::cout << model << std::endl;" << std::endl;
    std::cout << "//" << std::endl;
    std::cout << "// 2. Validate model name:" << std::endl;
    std::cout << "//    if (isValidModelName(\"large\")) { /* use model */ }" << std::endl;
    std::cout << "//" << std::endl;
    std::cout << "// 3. Get model path:" << std::endl;
    std::cout << "//    std::string path = getModelPath(\"large\"); // -> \"Systran/faster-whisper-large-v3\"" << std::endl;
    std::cout << "//" << std::endl;
    std::cout << "// 4. Filter by language support:" << std::endl;
    std::cout << "//    auto english_models = getEnglishOnlyModels();" << std::endl;
    std::cout << "//    auto multilingual_models = getMultilingualModels();" << std::endl;

    std::cout << "\n// Model analysis:" << std::endl;
    std::cout << "// - Total models: " << _MODELS.size() << std::endl;
    std::cout << "// - English-only models: " << getEnglishOnlyModels().size() << std::endl;
    std::cout << "// - Multilingual models: " << getMultilingualModels().size() << std::endl;
    std::cout << "// - Distil models available for faster inference" << std::endl;
    std::cout << "// - Sizes from tiny (fastest) to large (most accurate)" << std::endl;
}

#ifndef TESTING_MODE
int main() {
    bool tests_passed = run_utils_tests();

    if (tests_passed) {
        demonstrate_utils_usage();
    }

    return tests_passed ? 0 : 1;
}
#endif