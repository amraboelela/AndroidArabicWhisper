package org.amr.arabicwhisper

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import org.amr.arabicwhisper.ui.theme.ArabicWhisperTheme
import java.io.File
import android.content.res.AssetManager

class MainActivity : ComponentActivity() {
  private var muhaffezHelper: MuhaffezWhisperHelper? = null

  private val requestPermissionLauncher = registerForActivityResult(
    ActivityResultContracts.RequestPermission()
  ) { isGranted: Boolean ->
    if (isGranted) {
      Log.d("#transcribe", "Audio recording permission granted")
    } else {
      Log.e("#transcribe", "Audio recording permission denied")
    }
  }

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)

    // Check and request audio recording permission
    checkAudioPermission()

    // Initialize Muhaffez Whisper model for word-level transcription
    Log.d("#transcribe", "Initializing Muhaffez Whisper Helper")
    muhaffezHelper = MuhaffezWhisperHelper(this)

    // Copy audio test files to internal storage
    copyAudioFromAssets(assets, this.filesDir)

    // Create the audio file path using Context.getFilesDir()
    val audioFilePath = File(this.filesDir, "001.wav").absolutePath

    setContent {
      ArabicWhisperTheme {
        Surface(
          modifier = Modifier.fillMaxSize(),
          color = MaterialTheme.colorScheme.background
        ) {
          MainScreen(
            muhaffezHelper = muhaffezHelper,
            audioFilePath = audioFilePath,
            context = this
          )
        }
      }
    }
  }

  private fun checkAudioPermission() {
    when {
      ContextCompat.checkSelfPermission(
        this,
        Manifest.permission.RECORD_AUDIO
      ) == PackageManager.PERMISSION_GRANTED -> {
        Log.d("#transcribe", "Audio recording permission already granted")
      }
      else -> {
        requestPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
      }
    }
  }

  override fun onDestroy() {
    super.onDestroy()
    muhaffezHelper?.cleanup()
  }
}

@Composable
fun MainScreen(
  muhaffezHelper: MuhaffezWhisperHelper?,
  audioFilePath: String,
  context: android.content.Context
) {
  var transcription by remember { mutableStateOf("") }
  var isProcessing by remember { mutableStateOf(false) }
  var isRecording by remember { mutableStateOf(false) }

  // Create AudioRecorder with callback for each 1.3-second chunk
  val audioRecorder = remember {
    AudioRecorder { audioChunk ->
      // This callback is triggered every 1.3 seconds with trimmed audio
      Thread {
        try {
          Log.d("#transcribe", "Transcribing chunk: ${audioChunk.size} samples")
          val result = muhaffezHelper?.transcribe(audioChunk) ?: ""

          // Only append non-empty results (confidence > 20%)
          if (result.isNotEmpty()) {
            // Append new transcription to existing text immediately
            transcription = if (transcription.isEmpty()) {
              result
            } else {
              "$transcription $result"
            }
            Log.d("#transcribe", "Chunk transcription: $result")
          } else {
            Log.d("#transcribe", "Chunk transcription skipped (low confidence)")
          }
        } catch (e: Exception) {
          transcription = "$transcription [Error: ${e.message}]"
          Log.e("#transcribe", "Chunk transcription error", e)
        }
      }.start()
    }
  }

  Column(
    modifier = Modifier
      .fillMaxSize()
      .padding(16.dp),
    horizontalAlignment = Alignment.CenterHorizontally
  ) {
    Spacer(modifier = Modifier.height(24.dp))
    Text(
      text = "Muhaffez Arabic Whisper",
      style = MaterialTheme.typography.headlineMedium,
      modifier = Modifier.padding(bottom = 24.dp)
    )

    // Status text
    Text(
      text = when {
        transcription.isNotEmpty() -> transcription
        isRecording -> "🎙️ Recording... (speak now)"
        isProcessing -> "🔄 Processing..."
        else -> "Tap button to test transcription or record audio"
      },
      style = MaterialTheme.typography.bodyMedium,
      color = when {
        isRecording -> Color(0xFFF44336) // Red
        isProcessing -> Color(0xFFFF9800) // Orange
        else -> MaterialTheme.colorScheme.onBackground
      },
      modifier = Modifier.padding(vertical = 16.dp)
    )

    Spacer(modifier = Modifier.height(8.dp))

    // Record/Stop button
    Button(
      onClick = {
        if (isRecording) {
          // Stop recording
          isRecording = false
          audioRecorder.stopRecording()
          Log.d("#transcribe", "Recording stopped")
        } else {
          // Start recording
          isRecording = true
          transcription = ""
          audioRecorder.startRecording()
          Log.d("#transcribe", "Recording started")
        }
      },
      modifier = Modifier.fillMaxWidth(),
      colors = ButtonDefaults.buttonColors(
        containerColor = if (isRecording) Color(0xFFF44336) else MaterialTheme.colorScheme.primary
      )
    ) {
      Row(
        horizontalArrangement = Arrangement.Center,
        verticalAlignment = Alignment.CenterVertically
      ) {
        Icon(
          imageVector = if (isRecording) Icons.Default.Stop else Icons.Default.Mic,
          contentDescription = if (isRecording) "Stop recording" else "Start recording",
          modifier = Modifier.size(24.dp)
        )
        Spacer(modifier = Modifier.size(8.dp))
        Text(if (isRecording) "Stop Recording" else "Record Audio")
      }
    }

    Spacer(modifier = Modifier.height(16.dp))

    // Test with Muhaffez Whisper (word-level model)
    Button(
      onClick = {
        Thread {
          try {
            isProcessing = true
            transcription = "🔄 Processing..."
            val result = muhaffezHelper?.transcribeFile(audioFilePath) ?: "Error: No helper available"
            transcription = result
            isProcessing = false
            Log.d("#transcribe", "Muhaffez transcription: $result")
          } catch (e: Exception) {
            transcription = "Error: ${e.message}"
            isProcessing = false
            Log.e("#transcribe", "Muhaffez transcription error", e)
          }
        }.start()
      },
      enabled = !isRecording && !isProcessing,
      modifier = Modifier.fillMaxWidth()
    ) {
      Text("Test with 001.wav")
    }


  }
}

@Preview(showBackground = true)
@Composable
fun MainScreenPreview() {
  ArabicWhisperTheme {
    Text("Preview")
  }
}

fun copyAudioFromAssets(assetManager: AssetManager, destDir: File) {
  try {
    val audioFiles = listOf("001.wav")
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
