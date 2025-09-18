cmake_minimum_required(VERSION 3.10)
project(arabicwhisper)

# Add your shared library for JNI
add_library(
        whisper_jni
        SHARED
        whisper_jni.cpp       # your JNI wrapper file
        whisper_model.cpp     # new WhisperModel implementation
)

# Include directories for CTranslate2 headers
include_directories(/Users/amraboelela/develop/android/CTranslate2/include)

# Prebuilt shared library for CTranslate2
add_library(ctranslate2 SHARED IMPORTED)
set_target_properties(ctranslate2 PROPERTIES
        IMPORTED_LOCATION /Users/amraboelela/develop/android/CTranslate2/build-android-arm64/libctranslate2.so
)

# Link libraries
target_link_libraries(
        whisper_jni
        ctranslate2
        log
)
