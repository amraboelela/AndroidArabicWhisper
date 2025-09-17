package org.amr.arabicwhisper

import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import android.content.Context

class FasterWhisper(private val context: Context) {

  private val pyModule: PyObject

  init {
    // Start Python if not started
    if (!Python.isStarted()) {
      Python.start(AndroidPlatform(context))
    }

    val py = Python.getInstance()
    pyModule = py.getModule("whisper_helper")
  }

  fun initModel(modelPath: String): String {
    val initResult = pyModule.callAttr("init_model", modelPath)
    return initResult.toString()
  }

  fun transcribe(audioPath: String): String {
    val transcription = pyModule.callAttr("transcribe", audioPath)
    return transcription.toString()
  }
}
