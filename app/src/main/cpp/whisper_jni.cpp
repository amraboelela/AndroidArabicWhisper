#include <jni.h>
#include "whisper_model.h"
#include <string>
#include <vector>

static WhisperModel* whisper_model = nullptr;

extern "C" JNIEXPORT void JNICALL
Java_org_amr_arabicwhisper_WhisperHelper_initModel(JNIEnv* env, jobject thiz, jstring model_path) {
  const char* path = env->GetStringUTFChars(model_path, nullptr);
  whisper_model = new WhisperModel(path);
  env->ReleaseStringUTFChars(model_path, path);
}

extern "C" JNIEXPORT jstring JNICALL
Java_org_amr_arabicwhisper_WhisperHelper_transcribe(JNIEnv* env, jobject thiz, jfloatArray audio_data) {
  if (!whisper_model) {
    return env->NewStringUTF("Model not initialized");
  }

  // Get audio data from Java float array
  jsize length = env->GetArrayLength(audio_data);
  jfloat* audio_ptr = env->GetFloatArrayElements(audio_data, nullptr);

  // Convert to std::vector<float>
  std::vector<float> audio(audio_ptr, audio_ptr + length);

  // Release the array
  env->ReleaseFloatArrayElements(audio_data, audio_ptr, JNI_ABORT);

  try {
    // Transcribe audio
    auto [segments, info] = whisper_model->transcribe(audio);

    // Build result string from segments
    std::string result;
    for (const auto& segment : segments) {
      result += segment.text + " ";
    }

    return env->NewStringUTF(result.c_str());
  } catch (const std::exception& e) {
    return env->NewStringUTF(("Error: " + std::string(e.what())).c_str());
  }
}
