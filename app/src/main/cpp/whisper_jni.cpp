#include <jni.h>
#include "whisper_model.h"
#include "audio_decoder.h"
#include <android/log.h>
#include <string>
#include <vector>
#include <fstream>

static WhisperModel* whisper_model = nullptr;

extern "C" JNIEXPORT void JNICALL
Java_org_amr_arabicwhisper_WhisperHelper_initModel(JNIEnv* env, jobject thiz, jstring model_path) {
  const char* path = env->GetStringUTFChars(model_path, nullptr);
  whisper_model = new WhisperModel(path);
  env->ReleaseStringUTFChars(model_path, path);
}

extern "C" JNIEXPORT jstring JNICALL
Java_org_amr_arabicwhisper_WhisperHelper_transcribe(JNIEnv* env, jobject thiz, jstring input_path) {
  if (!whisper_model) {
    return env->NewStringUTF("Model not initialized");
  }

  // Get audio file path from Java string
  const char* path_ptr = env->GetStringUTFChars(input_path, nullptr);
  std::string audio_path(path_ptr);
  env->ReleaseStringUTFChars(input_path, path_ptr);

  try {
    // Check if file exists
    std::ifstream file(audio_path);
    if (!file.good()) {
      return env->NewStringUTF(("Audio file not found: " + audio_path).c_str());
    }
    file.close();

    // Load and decode audio file
    std::vector<float> audio_data = AudioDecoder::decode_audio(audio_path, 16000);

    if (audio_data.empty()) {
      return env->NewStringUTF(("Failed to decode audio file: " + audio_path).c_str());
    }

    // Transcribe the audio using WhisperModel with Arabic language
    __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "🚀 JNI about to call whisper_model->transcribe()");
    auto [segments, info] = whisper_model->transcribe(audio_data, "ar", true);

    __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "🎯 JNI received %zu segments from whisper_model->transcribe - TRANSCRIBE CALL COMPLETED!", segments.size());

    // Build result string from segments
    std::string result;
    __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "JNI building result string from %zu segments...", segments.size());

    for (const auto& segment : segments) {
      __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "JNI processing segment: '%s'", segment.text.c_str());
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

    __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "🏁 JNI about to call env->NewStringUTF() and return to Kotlin");
    jstring java_result = env->NewStringUTF(result.c_str());
    __android_log_print(ANDROID_LOG_DEBUG, "#transcribe", "✅ JNI NewStringUTF completed, returning to Kotlin...");

    return java_result;
  } catch (const std::exception& e) {
    return env->NewStringUTF(("Transcription error: " + std::string(e.what())).c_str());
  }
}
