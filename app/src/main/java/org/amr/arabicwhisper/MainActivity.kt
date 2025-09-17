package org.amr.arabicwhisper

import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {

  private lateinit var whisper: FasterWhisper

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)

    whisper = FasterWhisper(this)

    // Initialize the model from assets (or filesDir path)
    val modelPath = filesDir.absolutePath + "/whisper_ct2"
    val initResult = whisper.initModel(modelPath)
    Log.d("WhisperDemo", "Model init: $initResult")

    // Transcribe a file in internal storage
    val transcription = whisper.transcribe(filesDir.absolutePath + "/001.wav")
    Log.d("WhisperDemo", "Transcription: $transcription")
  }
}
