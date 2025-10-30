# GEMINI.md

## Project Overview

This project is an Android application that integrates an Arabic Quran Whisper model for offline speech recognition. The application is written in Kotlin and uses Jetpack Compose for its UI. The core of the speech recognition is a C++ backend that leverages the CTranslate2 library for efficient inference of the Whisper model.

The project is structured as follows:

*   **`app`**: The Android application module.
    *   **`src/main/java`**: Contains the Kotlin source code for the Android application.
    *   **`src/main/cpp`**: Contains the C++ source code for the JNI bridge and the transcription logic.
    *   **`src/main/assets`**: Contains the Whisper model files.
*   **`faster-whisper`**: A submodule that contains the necessary scripts and tools to convert the Whisper model to the CTranslate2 format.

## Architecture

### Native C++ Components
- **whisper_model.cpp/.h** - Core Whisper model implementation using CTranslate2
- **whisper_jni.cpp** - JNI interface bridging Java and C++
- **tokenizer.cpp/.h** - Whisper tokenizer implementation with Arabic support
- **feature_extractor.cpp/.h** - Audio feature extraction for Whisper input
- **audio.cpp/.h** - Audio preprocessing and format conversion
- **utils.cpp/.h** - Utility functions for model downloading and JSON parsing
- **whisper/** - Enhanced Whisper-compatible audio and tokenizer components

### Key Dependencies
- **CTranslate2** - Fast inference library for Transformer models
- **Android NDK** - Native development kit for C++ integration
- **zlib** - Compression library for model optimization

## Building and Running

### Prerequisites

*   Android Studio with NDK support
*   CMake 3.10 or higher
*   CTranslate2 library built for Android (arm64-v8a)
*   Python 3.9 or greater (for model conversion)

### Model Conversion

1.  Navigate to the `faster-whisper` directory.
2.  Install the required Python packages: `pip install -r requirements.txt`
3.  Follow the instructions in `faster-whisper/README.md` to convert the Hugging Face model (`tarteel-ai/whisper-base-ar-quran`) to the CTranslate2 format. This will generate a `whisper_ct2` directory.
4.  Copy the `whisper_ct2` directory to `app/src/main/assets/`.

### Building and Running the App

1.  Open the project in Android Studio.
2.  Build and run the application on an Android device or emulator.

### Gradle Commands
```bash
# Build debug version
./gradlew assembleDebug

# Build release version
./gradlew assembleRelease

# Clean build
./gradlew clean
```

## Development Conventions

### Code Style
- Follow standard C++ conventions with clear naming
- **Use 2-space indentation** (consistent with Android Studio default formatting)
- Use RAII and smart pointers for memory management
- Implement proper error handling with exceptions and return codes
- Document complex algorithms and Arabic-specific processing
- Follow Android Studio's formatting guidelines for consistency

### Key Technical Decisions Made
1. **Removed ONNX Runtime** - Eliminated VAD (Voice Activity Detection) dependency to avoid ONNX Runtime complexity
2. **CTranslate2 Integration** - Uses CTranslate2's Whisper implementation for optimal performance
3. **Type Safety** - Extensive work done on `vector<int>` vs `vector<size_t>` compatibility between tokenizer and CTranslate2 APIs
4. **Memory Management** - Uses `unique_ptr` and proper RAII patterns throughout

## API Usage

### JNI Interface
```cpp
// Initialize model
Java_org_amr_arabicwhisper_WhisperHelper_initModel(JNIEnv* env, jobject thiz, jstring model_path)

// Transcribe audio
Java_org_amr_arabicwhisper_WhisperHelper_transcribe(JNIEnv* env, jobject thiz, jfloatArray audio_data)
```

### WhisperModel Class
```cpp
// Constructor - Initialize model with path and configuration
WhisperModel(const std::string &model_size_or_path, ...);

// Main transcription method
std::tuple<std::vector<Segment>, TranscriptionInfo> transcribe(
    const std::vector<float> &audio,
    const std::optional<std::string> &language = std::nullopt,
    bool multilingual = false
);
```

## Troubleshooting

### Common Issues and Solutions

#### Compilation Issues
- **Incomplete Type Errors**: Ensure complete type definitions are included, not just forward declarations
- **Duplicate Symbol Errors**: Check for multiple definitions across translation units; use extern declarations properly
- **Type Conversion**: CTranslate2 uses `size_t` while internal tokenizer uses `int` - explicit conversions required

#### Linking Issues
- **Missing zlib**: Add `z` to `target_link_libraries` in CMakeLists.txt
- **CTranslate2 Path**: Verify `CTRANSLATE2_ROOT` path is correct for your environment
- **NDK Version**: Ensure NDK version matches project configuration

### Debugging Tips
- Use `adb logcat` to monitor native crashes and JNI issues
- Enable NDK debugging in Android Studio for C++ breakpoints
- Check memory usage with Android Studio's memory profiler
- Validate audio input format matches Whisper expectations (16kHz, mono)

## Arabic Language Specific Notes
- Ensure model supports Arabic language code "ar"
- Test with various Arabic dialects and accents
- Consider preprocessing for Arabic-specific audio characteristics
- Validate output text encoding handles Arabic script correctly