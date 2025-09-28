package org.amr.arabicwhisper

import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import org.amr.arabicwhisper.ui.theme.ArabicWhisperTheme
import java.io.File
import android.content.res.AssetManager

class MainActivity : ComponentActivity() {
  private lateinit var whisperHelper: WhisperHelper

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)

    // Initialize Whisper model
    val modelDir = File(this.filesDir, "whisper_ct2")
    if (!modelDir.exists()) {
      modelDir.mkdirs()
    }
    copyModelFromAssets(assets, modelDir)

    // Copy audio test files to internal storage
    copyAudioFromAssets(assets, this.filesDir)

    whisperHelper = WhisperHelper(this)

    // Create the audio file path using Context.getFilesDir()
    val audioFilePath = File(this.filesDir, "001.wav").absolutePath

    setContent {
      ArabicWhisperTheme {
        Surface(
          modifier = Modifier.fillMaxSize(),
          color = MaterialTheme.colorScheme.background
        ) {
          MainScreen(whisperHelper, audioFilePath)
        }
      }
    }
  }
}

@Composable
fun MainScreen(whisperHelper: WhisperHelper, audioFilePath: String) {
  Text(
    text = "Arabic Whisper App",
    style = MaterialTheme.typography.headlineMedium
  )

  // Example transcription calls using dynamic file path
  // val text = whisperHelper.transcribe("hello world")
  val transcription = whisperHelper.transcribe(audioFilePath)
  Log.d("#transcribe", "transcription: $transcription")
}

@Preview(showBackground = true)
@Composable
fun MainScreenPreview() {
  ArabicWhisperTheme {
    Text("Preview")
  }
}

fun copyModelFromAssets(assetManager: AssetManager, destDir: File) {
  try {
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
  } catch (e: Exception) {
    e.printStackTrace()
    // Handle missing assets gracefully
  }
}

fun copyAudioFromAssets(assetManager: AssetManager, destDir: File) {
  try {
    val audioFiles = listOf("001.wav", "002-01.wav", "test.wav")
    for (fileName in audioFiles) {
      val outFile = File(destDir, fileName)
      if (!outFile.exists()) {
        assetManager.open(fileName).use { input ->
          outFile.outputStream().use { output ->
            input.copyTo(output)
          }
        }
        Log.d("#transcribe", "Copied audio file: ${outFile.absolutePath}")
      }
    }
  } catch (e: Exception) {
    e.printStackTrace()
    Log.e("#transcribe", "Failed to copy audio files: ${e.message}")
  }
}
