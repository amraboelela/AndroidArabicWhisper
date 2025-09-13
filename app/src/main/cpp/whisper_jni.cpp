#include <jni.h>
#include <ctranslate2/generator.h>
#include <string>
#include <vector>

using namespace ctranslate2;

static Generator* generator = nullptr;

extern "C" JNIEXPORT void JNICALL
Java_org_amr_arabicwhisper_WhisperHelper_initModel(JNIEnv* env, jobject thiz, jstring model_path) {
    const char* path = env->GetStringUTFChars(model_path, nullptr);
    generator = new Generator(path, Device::CPU);
    env->ReleaseStringUTFChars(model_path, path);
}

extern "C" JNIEXPORT jstring JNICALL
Java_org_amr_arabicwhisper_WhisperHelper_transcribe(JNIEnv* env, jobject thiz, jstring input_text) {
    const char* input_cstr = env->GetStringUTFChars(input_text, nullptr);

    // Wrap input tokens
    std::vector<std::vector<std::string>> batch = {{input_cstr}};

    // Run generation
    auto results = generator->generate_batch(batch);

    // Collect output tokens
    std::string output;
    for (const auto& token : results[0].sequences[0]) {
        output += token + " ";
    }

    env->ReleaseStringUTFChars(input_text, input_cstr);
    return env->NewStringUTF(output.c_str());
}
