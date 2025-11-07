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
  private var whisperOnnxHelper: WhisperOnnxKotlinHelper? = null
  private var muhaffezHelper: MuhaffezWhisperHelper? = null
  private var audioRecorder: AudioRecorder? = null
  private var accumulatedTranscription = StringBuilder()

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

    // Initialize Whisper ONNX model
    Log.d("#transcribe", "Using ONNX implementation (pure Kotlin)")
    whisperOnnxHelper = WhisperOnnxKotlinHelper(this)

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
            whisperOnnxHelper = whisperOnnxHelper,
            muhaffezHelper = muhaffezHelper,
            audioFilePath = audioFilePath,
            onStartRecording = { startRecording() },
            onStopRecording = { stopRecording() },
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

  private fun startRecording() {
    if (ContextCompat.checkSelfPermission(
        this,
        Manifest.permission.RECORD_AUDIO
      ) != PackageManager.PERMISSION_GRANTED
    ) {
      requestPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
      return
    }

    // Clear state before starting new recording
    accumulatedTranscription.clear()
    Log.d("#transcribe", "Starting new recording, state cleared")

    audioRecorder = AudioRecorder { audioData ->
      // Call transcribeStream with ONNX helper
      whisperOnnxHelper?.transcribeStream(audioData) { result ->
        runOnUiThread {
          if (result.isNotEmpty()) {
            Log.d("#transcribe", "ONNX: New chunk transcribed: $result")
            if (accumulatedTranscription.isNotEmpty()) {
              accumulatedTranscription.append(" ")
            }
            accumulatedTranscription.append(result)
            whisperOnnxHelper?.onTranscriptionUpdate?.invoke(accumulatedTranscription.toString())
          }
        }
      }
    }.apply {
      startRecording()
    }
    Log.d("#transcribe", "Recording started")
  }

  private fun stopRecording() {
    audioRecorder?.stopRecording()
    audioRecorder = null
    Log.d("#transcribe", "Recording stopped")
  }

  override fun onDestroy() {
    super.onDestroy()
    audioRecorder?.stopRecording()
    muhaffezHelper?.cleanup()
  }
}

@Composable
fun MainScreen(
  whisperOnnxHelper: WhisperOnnxKotlinHelper?,
  muhaffezHelper: MuhaffezWhisperHelper?,
  audioFilePath: String,
  onStartRecording: () -> Unit,
  onStopRecording: () -> Unit,
  context: android.content.Context
) {
  var isRecording by remember { mutableStateOf(false) }
  var transcription by remember { mutableStateOf("") }
  var isProcessing by remember { mutableStateOf(false) }

  // Update the transcription state when helper provides new results
  whisperOnnxHelper?.onTranscriptionUpdate = {
    Log.d("#transcribe", "UI update with transcription: $it")
    transcription = it
  }

  // Update processing state
  whisperOnnxHelper?.onProcessingStateChange = {
    isProcessing = it
  }

  Column(
    modifier = Modifier
      .fillMaxSize()
      .padding(16.dp),
    horizontalAlignment = Alignment.CenterHorizontally
  ) {
    Spacer(modifier = Modifier.height(24.dp))
    Text(
      text = "Arabic Whisper",
      style = MaterialTheme.typography.headlineMedium,
      modifier = Modifier.padding(bottom = 24.dp)
    )

    // Microphone button
    Button(
      onClick = {
        if (isRecording) {
          isRecording = false
          onStopRecording()
        } else {
          isRecording = true
          transcription = ""
          onStartRecording()
        }
      },
      modifier = Modifier
        .size(120.dp)
        .padding(16.dp),
      colors = ButtonDefaults.buttonColors(
        containerColor = if (isRecording) Color.Red else MaterialTheme.colorScheme.primary
      )
    ) {
      Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
      ) {
        Icon(
          imageVector = if (isRecording) Icons.Default.Stop else Icons.Default.Mic,
          contentDescription = if (isRecording) "Stop Recording" else "Start Recording",
          modifier = Modifier.size(48.dp)
        )
        Spacer(modifier = Modifier.height(4.dp))
        Text(
          text = if (isRecording) "Stop" else "Record",
          style = MaterialTheme.typography.labelSmall
        )
      }
    }

    // Status text - show transcription with optional processing indicator below
    Column(
      modifier = Modifier.padding(vertical = 16.dp),
      horizontalAlignment = Alignment.CenterHorizontally
    ) {
      // Show transcription text if available
      if (transcription.isNotEmpty()) {
        Text(
          text = transcription,
          style = MaterialTheme.typography.bodyMedium,
          color = MaterialTheme.colorScheme.onBackground
        )
        Spacer(modifier = Modifier.height(8.dp))
      }

      // Show status indicator
      Text(
        text = when {
          isProcessing -> "🔄 Processing..."
          isRecording && transcription.isEmpty() -> "🎤 Recording..."
          transcription.isEmpty() -> "Tap microphone to start"
          else -> "" // Hide status when we have transcription and not processing
        },
        style = MaterialTheme.typography.bodySmall,
        color = when {
          isProcessing -> Color(0xFFFF9800) // Orange
          isRecording -> Color.Red
          else -> MaterialTheme.colorScheme.onBackground
        }
      )
    }

    // Test with existing audio file (BPE tokenizer model)
    Button(
      onClick = {
        Thread {
          try {
            transcription = "🔄 Processing..."
            val result = whisperOnnxHelper?.transcribe(audioFilePath) ?: "Error: No helper available"
            transcription = result
            Log.d("#transcribe", "Test transcription (BPE): $result")
          } catch (e: Exception) {
            transcription = "Error: ${e.message}"
            Log.e("#transcribe", "Test transcription error", e)
          }
        }.start()
      },
      modifier = Modifier.fillMaxWidth()
    ) {
      Text("Test with 001.wav (BPE Model)")
    }

    Spacer(modifier = Modifier.height(8.dp))

    // Test with Muhaffez Whisper (word-level model)
    Button(
      onClick = {
        Thread {
          try {
            transcription = "🔄 Processing Muhaffez model..."
            val result = muhaffezHelper?.transcribeFile(audioFilePath) ?: "Error: No helper available"
            transcription = result
            Log.d("#transcribe", "Muhaffez transcription: $result")
          } catch (e: Exception) {
            transcription = "Error: ${e.message}"
            Log.e("#transcribe", "Muhaffez transcription error", e)
          }
        }.start()
      },
      modifier = Modifier.fillMaxWidth()
    ) {
      Text("Test with 001.wav (Muhaffez Word Model)")
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
