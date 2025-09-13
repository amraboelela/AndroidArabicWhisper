package org.amr.arabicwhisper

import android.content.Context
import java.io.File

class WhisperHelper(context: Context, modelDirName: String = "whisper_ct2") {

  init {
    System.loadLibrary("ctranslate2")
    System.loadLibrary("whisper_jni") // your JNI wrapper .so

    val modelDir = File(context.filesDir, modelDirName).absolutePath
    initModel(modelDir)
  }

  external fun initModel(modelPath: String)
  external fun transcribe(inputText: String): String
}
