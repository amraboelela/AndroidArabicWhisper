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
        audio_decoder.cpp
        whisper/whisper_audio.cpp     # New whisper-compatible audio processing
        whisper/whisper_tokenizer.cpp # New whisper-compatible tokenizer
)

# Include directories for CTranslate2 headers
set(CTRANSLATE2_ROOT "/Users/amraboelela/develop/android/CTranslate2")
include_directories(${CTRANSLATE2_ROOT}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/whisper)  # Include whisper directory

# CTranslate2 and OpenMP libraries are now provided via jniLibs
# The libraries libctranslate2.so and libomp.so are in app/src/main/jniLibs/arm64-v8a/
# Define ctranslate2 as a prebuilt library that will be found in jniLibs
add_library(ctranslate2 SHARED IMPORTED)
set_target_properties(ctranslate2 PROPERTIES
        IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libctranslate2.so
)

# Try to link BLAS libraries for Android
# Check for different BLAS implementations
find_library(OPENBLAS_LIB openblas)
find_library(CBLAS_LIB cblas)
find_library(BLAS_LIB blas)

set(BLAS_LIBS "")
if(OPENBLAS_LIB)
    message(STATUS "Found OpenBLAS: ${OPENBLAS_LIB}")
    list(APPEND BLAS_LIBS ${OPENBLAS_LIB})
elseif(CBLAS_LIB)
    message(STATUS "Found CBLAS: ${CBLAS_LIB}")
    list(APPEND BLAS_LIBS ${CBLAS_LIB})
elseif(BLAS_LIB)
    message(STATUS "Found BLAS: ${BLAS_LIB}")
    list(APPEND BLAS_LIBS ${BLAS_LIB})
else()
    message(STATUS "No BLAS library found, using CTranslate2 with Ruy backend")
endif()

# Link libraries
target_link_libraries(
        whisper_jni
        ctranslate2     # Will be resolved from jniLibs at runtime
        z               # zlib for compression functions
        log             # Android logging
        ${BLAS_LIBS}    # Add BLAS libraries if available
)
