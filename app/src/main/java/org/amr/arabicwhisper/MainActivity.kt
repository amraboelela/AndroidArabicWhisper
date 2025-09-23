package org.amr.arabicwhisper

import android.os.Bundle
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

    whisperHelper = WhisperHelper(this)

    setContent {
      ArabicWhisperTheme {
        Surface(
          modifier = Modifier.fillMaxSize(),
          color = MaterialTheme.colorScheme.background
        ) {
          MainScreen(whisperHelper)
        }
      }
    }
  }
}

@Composable
fun MainScreen(whisperHelper: WhisperHelper) {
  Text(
    text = "Arabic Whisper App",
    style = MaterialTheme.typography.headlineMedium
  )

  // Example transcription calls
  // val text = whisperHelper.transcribe("hello world")
  // val transcription = whisperHelper.transcribe("/data/data/org.amr.arabicwhisper/files/001.wav")
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
