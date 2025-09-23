cmake_minimum_required(VERSION 3.10)
project(arabicwhisper)

# Add your shared library for JNI
add_library(
        whisper_jni
        SHARED
        whisper_jni.cpp       # your JNI wrapper file
        whisper_model.cpp
        utils.cpp
        tokenizer.cpp
        feature_extractor.cpp
        audio.cpp
        whisper/whisper_audio.cpp     # New whisper-compatible audio processing
        whisper/whisper_tokenizer.cpp # New whisper-compatible tokenizer
)

# Include directories for CTranslate2 headers
set(CTRANSLATE2_ROOT "/Users/amraboelela/develop/android/CTranslate2")
include_directories(${CTRANSLATE2_ROOT}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/whisper)  # Include whisper directory

# Prebuilt shared library for CTranslate2
add_library(ctranslate2 SHARED IMPORTED)
set_target_properties(ctranslate2 PROPERTIES
        IMPORTED_LOCATION ${CTRANSLATE2_ROOT}/build-android-arm64/libctranslate2.so
)

# Link libraries
target_link_libraries(
        whisper_jni
        ctranslate2
        z          # zlib for compression functions
        log
)
