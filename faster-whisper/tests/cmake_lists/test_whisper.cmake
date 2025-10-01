cmake_minimum_required(VERSION 3.10)
project(WhisperTests)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Copy source files to build directory
configure_file(${CMAKE_SOURCE_DIR}/../test_whisper.cpp ${CMAKE_SOURCE_DIR}/test_whisper.cpp COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/../whisper_transcriber.hpp ${CMAKE_SOURCE_DIR}/whisper_transcriber.hpp COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/../whisper_model_caller.cpp ${CMAKE_SOURCE_DIR}/whisper_model_caller.cpp COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/../whisper_model_caller.py ${CMAKE_SOURCE_DIR}/whisper_model_caller.py COPYONLY)

# Copy data directory if it exists
if(EXISTS ${CMAKE_SOURCE_DIR}/../data)
    file(COPY ${CMAKE_SOURCE_DIR}/../data DESTINATION ${CMAKE_SOURCE_DIR})
endif()

# Add main test executable
add_executable(test_whisper test_whisper.cpp)

# Set paths for faster_whisper_cpp
set(FASTER_WHISPER_CPP_DIR "../../../app/src/main/cpp")
set(CTRANSLATE2_ROOT "/Users/amraboelela/develop/android/CTranslate2")

message(STATUS "Building native C++ whisper_model_caller with CTranslate2")

# Copy cpp directory structure to build directory
file(COPY ${FASTER_WHISPER_CPP_DIR}/include DESTINATION ${CMAKE_SOURCE_DIR}/cpp)
file(COPY ${FASTER_WHISPER_CPP_DIR}/whisper DESTINATION ${CMAKE_SOURCE_DIR}/cpp)

# Add whisper model caller with C++ implementation
add_executable(whisper_model_caller whisper_model_caller.cpp
    ${FASTER_WHISPER_CPP_DIR}/whisper_model.cpp
    ${FASTER_WHISPER_CPP_DIR}/audio_decoder.cpp
    ${FASTER_WHISPER_CPP_DIR}/feature_extractor.cpp
    ${FASTER_WHISPER_CPP_DIR}/tokenizer.cpp
    ${FASTER_WHISPER_CPP_DIR}/utils.cpp
    ${FASTER_WHISPER_CPP_DIR}/whisper/whisper_tokenizer.cpp
    ${FASTER_WHISPER_CPP_DIR}/whisper/whisper_audio.cpp
)

# Add include directories
target_include_directories(whisper_model_caller PRIVATE
    ${CMAKE_SOURCE_DIR}/cpp/include
    ${CMAKE_SOURCE_DIR}/cpp/whisper
    ${CTRANSLATE2_ROOT}/include
)

# Find and link CTranslate2 static library
set(CTRANSLATE2_LIB "${CTRANSLATE2_ROOT}/libctranslate2.a")
find_library(ZLIB_LIB z)

# On macOS, find Accelerate framework for BLAS support
if(APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
endif()

if(EXISTS ${CTRANSLATE2_LIB} AND ZLIB_LIB)
    target_link_libraries(whisper_model_caller ${CTRANSLATE2_LIB} ${ZLIB_LIB})

    # Add Accelerate framework on macOS
    if(APPLE AND ACCELERATE_FRAMEWORK)
        target_link_libraries(whisper_model_caller ${ACCELERATE_FRAMEWORK})
        message(STATUS "Linked Accelerate framework: ${ACCELERATE_FRAMEWORK}")
    endif()

    message(STATUS "Linked CTranslate2 static library: ${CTRANSLATE2_LIB}")
    message(STATUS "Linked zlib: ${ZLIB_LIB}")
else()
    if(NOT EXISTS ${CTRANSLATE2_LIB})
        message(WARNING "CTranslate2 static library not found: ${CTRANSLATE2_LIB}")
    endif()
    if(NOT ZLIB_LIB)
        message(WARNING "zlib not found")
    endif()
    message(WARNING "Native build may fail without required libraries")
endif()

# Link filesystem library based on platform and compiler
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
    # GCC < 9 needs explicit linking to stdc++fs
    target_link_libraries(test_whisper stdc++fs)
    target_link_libraries(whisper_model_caller stdc++fs)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
    # Older Clang might need explicit linking
    target_link_libraries(test_whisper c++fs)
    target_link_libraries(whisper_model_caller c++fs)
endif()

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(test_whisper PRIVATE -Wall -Wextra)
    target_compile_options(whisper_model_caller PRIVATE -Wall -Wextra)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    target_compile_options(test_whisper PRIVATE -Wall -Wextra)
    target_compile_options(whisper_model_caller PRIVATE -Wall -Wextra)
endif()

# Set output directory
set_target_properties(test_whisper PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

set_target_properties(whisper_model_caller PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Print build information
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")