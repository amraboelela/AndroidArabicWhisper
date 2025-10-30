package org.amr.arabicwhisper

import android.content.Context
import android.util.Log
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.nio.ByteBuffer
import java.nio.ByteOrder

class WhisperHelper(context: Context, modelDirName: String = "whisper_ct2") {

  var onTranscriptionUpdate: ((String) -> Unit)? = null
  var onProcessingStateChange: ((Boolean) -> Unit)? = null
  private val executorService: ExecutorService = Executors.newSingleThreadExecutor()

  // Buffer for accumulating audio chunks
  private val audioBuffer = mutableListOf<Byte>()
  private val bufferLock = Any()

  // 4 seconds at 16kHz, 16-bit = 128000 bytes (longer chunks = better accuracy)
  private val CHUNK_SIZE_BYTES = 128000
  private var isProcessing = false

  init {
    // Note: ctranslate2 is loaded as a dependency of whisper_jni
    System.loadLibrary("whisper_jni") // your JNI wrapper .so

    val modelDir = File(context.filesDir, modelDirName).absolutePath
    initModel(modelDir)

    Log.d("#transcribe", "📱 WhisperHelper initialized, isProcessing=$isProcessing")
  }

  fun transcribeStream(audioData: ByteArray, onResult: (String) -> Unit) {
    synchronized(bufferLock) {
      // Add audio data to buffer
      audioBuffer.addAll(audioData.toList())

      val bufferSize = audioBuffer.size
      val bufferSeconds = bufferSize / 32000.0f // 16kHz * 2 bytes

      // Only process if we have enough data and not already processing
      if (bufferSize >= CHUNK_SIZE_BYTES && !isProcessing) {

        // Check if audio chunk has significant energy (not silent)
        val audioFloats = audioBuffer.toByteArray().let { bytes ->
          FloatArray(bytes.size / 2) { i ->
            val sample = ((bytes[i * 2 + 1].toInt() shl 8) or (bytes[i * 2].toInt() and 0xFF)).toShort()
            sample / 32768.0f
          }
        }

        // Calculate RMS (Root Mean Square) energy
        val rms = kotlin.math.sqrt(audioFloats.map { it * it }.average()).toFloat()
        val silenceThreshold = 0.025f // Increased threshold to filter background noise better

        if (rms < silenceThreshold) {
          Log.d("#transcribe", "🔇 Silence detected (RMS: %.4f), skipping transcription".format(rms))
          audioBuffer.clear()
          return
        }

        Log.d("#transcribe", "🔊 Audio detected (RMS: %.4f), starting transcription".format(rms))

        isProcessing = true
        onProcessingStateChange?.invoke(true)
        Log.d("#transcribe", "⚡ Starting transcription, buffer size: $bufferSize bytes (%.2f sec)".format(bufferSeconds))

        // Extract chunk to process
        val chunkToProcess = audioBuffer.toByteArray()

        // Clear buffer completely - no overlap for real-time streaming
        // (overlap causes repeated transcriptions of the same audio)
        audioBuffer.clear()

        // Process in background thread
        executorService.execute {
          try {
            val result = transcribeStreamNative(chunkToProcess)

            if (result.isNotEmpty()) {
              Log.d("#transcribe", "✅ Transcription result: $result")
              onResult(result)
            } else {
              Log.d("#transcribe", "⚠️ Empty transcription result")
            }
          } catch (e: Exception) {
            Log.e("#transcribe", "❌ Transcription error", e)
          } finally {
            synchronized(bufferLock) {
              isProcessing = false
              onProcessingStateChange?.invoke(false)
              val currentBufferSize = audioBuffer.size
              Log.d("#transcribe", "🏁 Transcription done, isProcessing=false, buffer now: $currentBufferSize bytes")

              // Check if we accumulated enough audio during processing to trigger another transcription
              if (currentBufferSize >= CHUNK_SIZE_BYTES) {
                Log.d("#transcribe", "🔄 Buffer filled during processing, triggering another transcription")
                // Recursively call to process the accumulated buffer
                transcribeStream(ByteArray(0), onResult)
              }
            }
          }
        }
      } else {
        // Log occasionally to show accumulation
        if (bufferSize % 32000 < 1280 || bufferSize < 10000) {
          Log.d("#transcribe", "📊 Buffer: $bufferSize bytes (%.2f sec), isProcessing=$isProcessing".format(bufferSeconds))
        }
      }
    }
  }

  fun clearTranscription() {
    synchronized(bufferLock) {
      audioBuffer.clear()
      isProcessing = false
      Log.d("#transcribe", "🧹 Buffer and state cleared, isProcessing=false")
    }
    onProcessingStateChange?.invoke(false)
    clearTranscriptionNative()
  }

  fun shutdown() {
    executorService.shutdown()
  }

  external fun initModel(modelPath: String)
  external fun transcribe(inputText: String): String
  private external fun transcribeStreamNative(audioData: ByteArray): String
  private external fun clearTranscriptionNative()
}
