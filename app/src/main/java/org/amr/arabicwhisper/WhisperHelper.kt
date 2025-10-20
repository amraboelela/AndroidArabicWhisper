package org.amr.arabicwhisper

import android.content.Context
import android.util.Log
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform

class WhisperHelper(private val context: Context, private val modelSize: String = "base") {

  private val python: Python
  private val whisperModule: com.chaquo.python.PyObject

  init {
    // Load native CTranslate2 library before starting Python
    try {
      System.loadLibrary("ctranslate2")
      System.loadLibrary("omp")
      Log.d("WhisperHelper", "Native libraries loaded successfully")
    } catch (e: UnsatisfiedLinkError) {
      Log.e("WhisperHelper", "Failed to load native libraries", e)
    }

    // Start Python
    if (!Python.isStarted()) {
      Python.start(AndroidPlatform(context))
    }

    python = Python.getInstance()
    whisperModule = python.getModule("whisper_transcriber")

    // Initialize model
    try {
      val result = whisperModule.callAttr("init_model", modelSize, "cpu", "int8")
      Log.d("WhisperHelper", "Model initialization: $result")
    } catch (e: Exception) {
      Log.e("WhisperHelper", "Error initializing model", e)
    }
  }

  fun transcribe(audioFilePath: String, language: String = "ar", beamSize: Int = 5): String {
    return try {
      val result = whisperModule.callAttr("transcribe_audio", audioFilePath, language, beamSize)
      result.toString()
    } catch (e: Exception) {
      Log.e("WhisperHelper", "Error during transcription", e)
      "Error: ${e.message}"
    }
  }

  fun transcribeWithModel(audioFilePath: String, modelSize: String = "base", language: String = "ar"): String {
    return try {
      val result = whisperModule.callAttr("transcribe_audio_with_model", audioFilePath, modelSize, language)
      result.toString()
    } catch (e: Exception) {
      Log.e("WhisperHelper", "Error during transcription", e)
      "Error: ${e.message}"
    }
  }
}
