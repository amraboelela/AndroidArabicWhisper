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
  private lateinit var whisperHelper: WhisperHelper
  private var audioRecorder: AudioRecorder? = null
  private var recordedAudioFile: File? = null

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
          MainScreen(
            whisperHelper = whisperHelper,
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

    recordedAudioFile = File(this.filesDir, "recorded_audio.wav")
    audioRecorder = AudioRecorder(recordedAudioFile!!).apply {
      startRecording()
    }
    Log.d("#transcribe", "Recording started")
  }

  private fun stopRecording(): String? {
    audioRecorder?.stopRecording()
    audioRecorder = null
    Log.d("#transcribe", "Recording stopped")
    return recordedAudioFile?.absolutePath
  }
}

@Composable
fun MainScreen(
  whisperHelper: WhisperHelper,
  audioFilePath: String,
  onStartRecording: () -> Unit,
  onStopRecording: () -> String?,
  context: android.content.Context
) {
  var isRecording by remember { mutableStateOf(false) }
  var transcription by remember { mutableStateOf("") }
  var isTranscribing by remember { mutableStateOf(false) }

  Column(
    modifier = Modifier
      .fillMaxSize()
      .padding(16.dp),
    horizontalAlignment = Alignment.CenterHorizontally
  ) {
    Text(
      text = "Arabic Whisper App",
      style = MaterialTheme.typography.headlineMedium,
      modifier = Modifier.padding(bottom = 24.dp)
    )

    // Microphone button
    Button(
      onClick = {
        if (isRecording) {
          isRecording = false
          val recordedPath = onStopRecording()
          if (recordedPath != null) {
            isTranscribing = true
            // Transcribe in background
            Thread {
              try {
                val result = whisperHelper.transcribe(recordedPath)
                transcription = result
                Log.d("#transcribe", "Transcription: $result")
              } catch (e: Exception) {
                transcription = "Error: ${e.message}"
                Log.e("#transcribe", "Transcription error", e)
              } finally {
                isTranscribing = false
              }
            }.start()
          }
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

    // Status text
    Text(
      text = when {
        isRecording -> "Recording..."
        isTranscribing -> "Transcribing..."
        transcription.isNotEmpty() -> "Transcription complete"
        else -> "Tap microphone to start"
      },
      style = MaterialTheme.typography.bodyMedium,
      color = if (isRecording) Color.Red else MaterialTheme.colorScheme.onBackground,
      modifier = Modifier.padding(vertical = 16.dp)
    )

    // Transcription result
    if (transcription.isNotEmpty()) {
      Text(
        text = "Result:",
        style = MaterialTheme.typography.titleMedium,
        modifier = Modifier
          .fillMaxWidth()
          .padding(bottom = 8.dp)
      )
      Text(
        text = transcription,
        style = MaterialTheme.typography.bodyLarge,
        modifier = Modifier.fillMaxWidth()
      )
    }

    Spacer(modifier = Modifier.height(24.dp))

    // Test with existing audio file
    Button(
      onClick = {
        isTranscribing = true
        Thread {
          try {
            val result = whisperHelper.transcribe(audioFilePath)
            transcription = result
            Log.d("#transcribe", "Test transcription: $result")
          } catch (e: Exception) {
            transcription = "Error: ${e.message}"
            Log.e("#transcribe", "Test transcription error", e)
          } finally {
            isTranscribing = false
          }
        }.start()
      },
      modifier = Modifier.fillMaxWidth()
    ) {
      Text("Test with 001.wav")
    }

    Spacer(modifier = Modifier.height(8.dp))

    // Button to copy recorded audio to Downloads for inspection
    Button(
      onClick = {
        Thread {
          try {
            val recordedFile = File(context.filesDir, "recorded_audio.wav")
            if (recordedFile.exists()) {
              val downloadsDir = android.os.Environment.getExternalStoragePublicDirectory(
                android.os.Environment.DIRECTORY_DOWNLOADS
              )
              val destFile = File(downloadsDir, "recorded_audio_${System.currentTimeMillis()}.wav")
              recordedFile.copyTo(destFile, overwrite = true)
              transcription = "Saved to: ${destFile.absolutePath}"
              Log.d("#transcribe", "Copied recording to: ${destFile.absolutePath}")
            } else {
              transcription = "No recording found"
            }
          } catch (e: Exception) {
            transcription = "Copy error: ${e.message}"
            Log.e("#transcribe", "Failed to copy recording", e)
          }
        }.start()
      },
      modifier = Modifier.fillMaxWidth()
    ) {
      Text("Save Last Recording to Downloads")
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
