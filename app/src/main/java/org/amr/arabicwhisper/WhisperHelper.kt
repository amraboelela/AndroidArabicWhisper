package org.amr.arabicwhisper

import android.content.Context
import java.io.File

class WhisperHelper(context: Context, modelDirName: String = "whisper_ct2") {

  init {
    // Note: ctranslate2 is loaded as a dependency of whisper_jni
    System.loadLibrary("whisper_jni") // your JNI wrapper .so

    val modelDir = File(context.filesDir, modelDirName).absolutePath
    initModel(modelDir)
  }

  external fun initModel(modelPath: String)
  external fun transcribe(inputText: String): String
}
