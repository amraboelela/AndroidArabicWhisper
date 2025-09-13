cmake_minimum_required(VERSION 3.10)
project(arabicwhisper)

add_library(
        whisper_jni
        SHARED
        whisper_jni.cpp   # your JNI wrapper file
)

# Point to CTranslate2 build
include_directories(/Users/amraboelela/develop/android/CTranslate2/include)

# Link against the prebuilt shared lib
add_library(ctranslate2 SHARED IMPORTED)
set_target_properties(ctranslate2 PROPERTIES
        IMPORTED_LOCATION /Users/amraboelela/develop/android/CTranslate2/build-android-arm64/libctranslate2.so
)

# Link
target_link_libraries(
        whisper_jni
        ctranslate2
        log
)
