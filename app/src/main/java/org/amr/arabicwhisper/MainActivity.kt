package org.amr.arabicwhisper

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import java.io.File
import android.content.res.AssetManager

class MainActivity : AppCompatActivity() {
  private lateinit var whisperHelper: WhisperHelper

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)

    // Use 'this' as the context
    val modelDir = File(this.filesDir, "whisper_ct2")
    if (!modelDir.exists()) {
      modelDir.mkdirs() // <-- now it works
    }
    copyModelFromAssets(assets, modelDir)
    // TODO: Copy model.bin and other files into modelDir if not already there

    whisperHelper = WhisperHelper(this)

    // Example call with a string
    val text = whisperHelper.transcribe("hello world")
    println("Transcription: $text")

    // Another instance, using wav path
    val transcription = whisperHelper.transcribe("/data/data/org.amr.arabicwhisper/files/001.wav")
    println("Wav transcription: $transcription")
  }
}

fun copyModelFromAssets(assetManager: AssetManager, destDir: File) {
  val files = assetManager.list("whisper_ct2") ?: return
  for (fileName in files) {
    val outFile = File(destDir, fileName)
    if (!outFile.exists()) {
      assetManager.open("whisper_ct2/$fileName").use { input ->
        outFile.outputStream().use { output ->
          input.copyTo(output)
        }
      }
    }
  }
}
