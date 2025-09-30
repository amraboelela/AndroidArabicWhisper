cmake_minimum_required(VERSION 3.10)
project(WhisperTests)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Copy source files to build directory
configure_file(${CMAKE_SOURCE_DIR}/../test_whisper.cpp ${CMAKE_SOURCE_DIR}/test_whisper.cpp COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/../whisper_transcriber.hpp ${CMAKE_SOURCE_DIR}/whisper_transcriber.hpp COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/../whisper_model_caller.py ${CMAKE_SOURCE_DIR}/whisper_model_caller.py COPYONLY)

# Copy py_helpers directory
file(COPY ${CMAKE_SOURCE_DIR}/../py_helpers DESTINATION ${CMAKE_SOURCE_DIR})

# Copy data directory if it exists
if(EXISTS ${CMAKE_SOURCE_DIR}/../data)
    file(COPY ${CMAKE_SOURCE_DIR}/../data DESTINATION ${CMAKE_SOURCE_DIR})
endif()

# Add executable
add_executable(test_whisper test_whisper.cpp)

# Link filesystem library based on platform and compiler
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
    # GCC < 9 needs explicit linking to stdc++fs
    target_link_libraries(test_whisper stdc++fs)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
    # Older Clang might need explicit linking
    target_link_libraries(test_whisper c++fs)
endif()

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(test_whisper PRIVATE -Wall -Wextra)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    target_compile_options(test_whisper PRIVATE -Wall -Wextra)
endif()

# Set output directory
set_target_properties(test_whisper PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Print build information
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")