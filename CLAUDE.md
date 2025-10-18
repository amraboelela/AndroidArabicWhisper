# CLAUDE.md
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is an Android NDK project that integrates Arabic speech recognition using OpenAI's Whisper model via the CTranslate2 framework. The project consists of:
- **Android App** - Java/Kotlin Android application with JNI interface
- **Native C++ Layer** - NDK implementation using CTranslate2 for Whisper inference
- **Arabic Language Support** - Optimized for Arabic speech recognition and transcription

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

## Build System
### Android Gradle Build
```bash
# Build debug version
./gradlew assembleDebug

# Build release version
./gradlew assembleRelease

# Clean build
./gradlew clean
```

### CMake Configuration
The native code is built using CMake with the following key configuration:
- **Target**: `libwhisper_jni.so` shared library
- **CTranslate2 Path**: `/Users/amraboelela/develop/android/CTranslate2`
- **NDK Version**: 27.0.12077973
- **Target Architecture**: arm64-v8a (primary), with multi-arch support

## Development Guidelines

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

### Common Issues and Solutions

#### Compilation Issues
- **Incomplete Type Errors**: Ensure complete type definitions are included, not just forward declarations
- **Duplicate Symbol Errors**: Check for multiple definitions across translation units; use extern declarations properly
- **Type Conversion**: CTranslate2 uses `size_t` while internal tokenizer uses `int` - explicit conversions required

#### Linking Issues
- **Missing zlib**: Add `z` to `target_link_libraries` in CMakeLists.txt
- **CTranslate2 Path**: Verify `CTRANSLATE2_ROOT` path is correct for your environment
- **NDK Version**: Ensure NDK version matches project configuration

### Testing Strategy
- **Unit Testing**: Test individual components like tokenizer and feature extraction
- **Integration Testing**: Test JNI interface with actual audio input
- **Performance Testing**: Monitor inference speed and memory usage
- **Arabic Language Testing**: Validate transcription quality with Arabic audio samples

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

## Environment Setup
### Prerequisites
- Android Studio with NDK support
- CMake 3.10 or higher
- CTranslate2 library built for Android (arm64-v8a)
- Android SDK and NDK 27.0.12077973

### Build Configuration
Update paths in CMakeLists.txt if your environment differs:
```cmake
set(CTRANSLATE2_ROOT "/Users/amraboelela/develop/android/CTranslate2")
```

## Debugging Tips
- Use `adb logcat` to monitor native crashes and JNI issues
- Enable NDK debugging in Android Studio for C++ breakpoints
- Check memory usage with Android Studio's memory profiler
- Validate audio input format matches Whisper expectations (16kHz, mono)

## Performance Considerations
- **Model Size**: Larger models (large-v3) provide better accuracy but slower inference
- **Audio Length**: Longer audio segments require more memory and processing time
- **Batch Processing**: Process audio in chunks for better memory management
- **Threading**: CTranslate2 handles internal threading; avoid additional threading complexity

## Arabic Language Specific Notes
- Ensure model supports Arabic language code "ar"
- Test with various Arabic dialects and accents
- Consider preprocessing for Arabic-specific audio characteristics
- Validate output text encoding handles Arabic script correctly

## Troubleshooting Build Issues
1. **Gradle Daemon Issues**: Use `./gradlew --no-daemon` or `./gradlew --stop`
2. **NDK Path**: Verify Android NDK installation and path configuration
3. **CTranslate2 Missing**: Ensure CTranslate2 is built for Android arm64-v8a
4. **Memory Issues**: Increase JVM heap size in gradle.properties if needed

## Important File Locations
- **JNI Source**: `app/src/main/cpp/`
- **Headers**: `app/src/main/cpp/include/`
- **CMake Config**: `app/src/main/cpp/CMakeLists.txt`
- **Gradle Config**: `app/build.gradle`
- **Models**: Should be placed in app assets or downloaded at runtime

## General rules
- Make test_whisper.sh like app/src/test/cpp/whisper_tokenizer_tests.sh
- Never write python script in cpp files like: scriptFile << "import os\n";
        scriptFile << "import sys\n";
        scriptFile << "os.environ['HF_HUB_OFFLINE'] = '1'\n";
        scriptFile << "from faster_whisper import WhisperModel\n";
        scriptFile << "from faster_whisper.audio import decode_audio\n";
        scriptFile << "\n";
        scriptFile << "def main():\n"; The only python script you can write inside cpp file is to call another python file
- We are trying to convert faster-whisper library from python to c++ one file at a time, from top to bottom.
  - So when I ask you to convert e.g. test_whisper_ct2_offline.py to test_whisper_ct2_offline.cpp then do not call test_whisper_ct2_offline.py inside test_whisper_ct2_offline.cpp but rather call from faster_whisper import WhisperModel or any other python imported packages.
- Inside faster_whisper do not do make test, just make is enough, all what i need to know that it is transcribing the audio 001.wav file to al-fatiha correctly, and i can know that by myself, no need to write code to verify that.
- 
