#include <jni.h>
#include "whisper_model.h"
#include "audio_decoder.h"
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

    // Transcribe the audio using WhisperModel
    auto [segments, info] = whisper_model->transcribe(audio_data, std::nullopt, true);

    // Build result string from segments
    std::string result;
    for (const auto& segment : segments) {
      result += segment.text;
      if (!segment.text.empty() && segment.text.back() != ' ') {
        result += " ";
      }
    }

    // Remove trailing space
    if (!result.empty() && result.back() == ' ') {
      result.pop_back();
    }

    if (result.empty()) {
      result = "No speech detected in audio file";
    }

    return env->NewStringUTF(result.c_str());
  } catch (const std::exception& e) {
    return env->NewStringUTF(("Transcription error: " + std::string(e.what())).c_str());
  }
}
