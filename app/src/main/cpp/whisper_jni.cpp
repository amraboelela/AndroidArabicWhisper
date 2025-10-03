#include <jni.h>
#include "transcribe.h"
#include "audio.h"
#include <android/log.h>
#include <string>
#include <vector>
#include <fstream>
#include <codecvt>
#include <locale>

static WhisperModel* whisper_model = nullptr;

// Helper function to convert UTF-8 string to jstring properly
jstring createJavaStringFromUTF8(JNIEnv* env, const std::string& utf8_str) {
  __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "🔤 Converting UTF-8 string to Java string, length: %zu", utf8_str.length());

  // Debug: Print raw bytes for analysis
  std::string bytes_debug = "Raw UTF-8 bytes: ";
  for (size_t i = 0; i < std::min(utf8_str.length(), size_t(50)); ++i) {
    char byte_str[10];
    sprintf(byte_str, "\\x%02x", (unsigned char)utf8_str[i]);
    bytes_debug += byte_str;
  }
  __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "%s", bytes_debug.c_str());

  try {
    // Convert UTF-8 to UTF-16 using standard library
    std::wstring_convert<std::codecvt_utf8_utf16<char16_t>, char16_t> converter;
    std::u16string utf16_str = converter.from_bytes(utf8_str);

    __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "✅ UTF-8 to UTF-16 conversion successful, UTF-16 length: %zu", utf16_str.length());

    // Create Java string from UTF-16
    jstring result = env->NewString(reinterpret_cast<const jchar*>(utf16_str.c_str()), utf16_str.length());

    if (result == nullptr) {
      __android_log_print(ANDROID_LOG_ERROR, "#transcribe", "❌ Failed to create Java string from UTF-16");
      // Fallback to NewStringUTF
      return env->NewStringUTF(utf8_str.c_str());
    }

    return result;

  } catch (const std::exception& e) {
    __android_log_print(ANDROID_LOG_ERROR, "#transcribe", "❌ UTF-8 to UTF-16 conversion failed: %s", e.what());
    __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "Falling back to NewStringUTF");
    // Fallback to the original method
    return env->NewStringUTF(utf8_str.c_str());
  }
}

extern "C" JNIEXPORT void JNICALL
Java_org_amr_arabicwhisper_WhisperHelper_initModel(JNIEnv* env, jobject thiz, jstring model_path) {
  const char* path = env->GetStringUTFChars(model_path, nullptr);
  whisper_model = new WhisperModel(path);
  env->ReleaseStringUTFChars(model_path, path);
}

extern "C" JNIEXPORT jstring JNICALL
Java_org_amr_arabicwhisper_WhisperHelper_transcribe(JNIEnv* env, jobject thiz, jstring input_path) {
  if (!whisper_model) {
    return createJavaStringFromUTF8(env, "Model not initialized");
  }

  // Get audio file path from Java string
  const char* path_ptr = env->GetStringUTFChars(input_path, nullptr);
  std::string audio_path(path_ptr);
  env->ReleaseStringUTFChars(input_path, path_ptr);

  try {
    // Check if file exists
    std::ifstream file(audio_path);
    if (!file.good()) {
      std::string error_msg = "Audio file not found: " + audio_path;
      return createJavaStringFromUTF8(env, error_msg);
    }
    file.close();

    // Load and decode audio file
    std::vector<float> audio_data = AudioDecoder::decode_audio(audio_path, 16000);

    if (audio_data.empty()) {
      std::string error_msg = "Failed to decode audio file: " + audio_path;
      return createJavaStringFromUTF8(env, error_msg);
    }

    // Transcribe the audio using WhisperModel with Arabic language
    __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "🚀 JNI about to call whisper_model->transcribe()");
    auto [segments, info] = whisper_model->transcribe(audio_data, "ar", true);

    __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "🎯 JNI received %zu segments from whisper_model->transcribe - TRANSCRIBE CALL COMPLETED!", segments.size());

    // Build result string from segments
    std::string result;
    __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "JNI building result string from %zu segments...", segments.size());

    for (size_t i = 0; i < segments.size(); ++i) {
      const auto& segment = segments[i];
      __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "🔍 JNI processing segment %zu: '%s'", i, segment.text.c_str());

      // Debug: Print raw bytes of each segment
      std::string segment_bytes = "Segment " + std::to_string(i) + " raw bytes: ";
      for (size_t j = 0; j < std::min(segment.text.length(), size_t(50)); ++j) {
        char byte_str[10];
        sprintf(byte_str, "\\x%02x", (unsigned char)segment.text[j]);
        segment_bytes += byte_str;
      }
      __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "%s", segment_bytes.c_str());

      result += segment.text;
      if (!segment.text.empty() && segment.text.back() != ' ') {
        result += " ";
      }
    }

    __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "✅ JNI result string built! Final result: '%s'", result.c_str());

    // Remove trailing space
    if (!result.empty() && result.back() == ' ') {
      result.pop_back();
    }

    if (result.empty()) {
      result = "No speech detected in audio file";
    }

    __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "🏁 JNI about to call createJavaStringFromUTF8() and return to Kotlin");
    jstring java_result = createJavaStringFromUTF8(env, result);
    __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "✅ JNI UTF-8 to UTF-16 conversion completed, returning to Kotlin...");

    return java_result;
  } catch (const std::exception& e) {
    std::string error_msg = "Transcription error: " + std::string(e.what());
    return createJavaStringFromUTF8(env, error_msg);
  }
}
