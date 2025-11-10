package org.amr.arabicwhisper

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.log10
import kotlin.math.max

/**
 * Audio recorder for capturing microphone input
 * Records in 16kHz, mono, 16-bit PCM format (compatible with Whisper)
 * Streams audio data and triggers transcription every 1.3 seconds
 */
class AudioRecorder(private val onChunkReady: (FloatArray) -> Unit) {

  private var audioRecord: AudioRecord? = null
  private var isRecording = false
  private var recordingThread: Thread? = null
  private val audioDataBuffer = mutableListOf<ByteArray>()

  companion object {
    private const val SAMPLE_RATE = 16000
    private const val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
    private const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT
    private const val BUFFER_SIZE_MULTIPLIER = 2
    private const val CHUNK_DURATION_SECONDS = 1.3f
    private const val CHUNK_SIZE_SAMPLES = (SAMPLE_RATE * CHUNK_DURATION_SECONDS).toInt()
    private const val CHUNK_SIZE_BYTES = CHUNK_SIZE_SAMPLES * 2  // 2 bytes per sample
    private const val TAG = "#recorder"
  }

  /**
   * Start recording audio from microphone
   */
  fun startRecording() {
    if (isRecording) {
      Log.w(TAG, "Already recording")
      return
    }

    // Clear previous recording data
    audioDataBuffer.clear()

    val bufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT) * BUFFER_SIZE_MULTIPLIER

    try {
      audioRecord = AudioRecord(
        MediaRecorder.AudioSource.MIC,
        SAMPLE_RATE,
        CHANNEL_CONFIG,
        AUDIO_FORMAT,
        bufferSize
      )

      if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
        Log.e(TAG, "AudioRecord initialization failed")
        return
      }

      isRecording = true
      audioRecord?.startRecording()
      Log.d(TAG, "Recording started")

      // Start recording thread
      recordingThread = Thread {
        recordAudioData(bufferSize)
      }
      recordingThread?.start()

    } catch (e: Exception) {
      Log.e(TAG, "Failed to start recording: ${e.message}", e)
      isRecording = false
    }
  }

  /**
   * Stop recording
   */
  fun stopRecording() {
    if (!isRecording) {
      Log.w(TAG, "Not recording")
      return
    }

    isRecording = false
    audioRecord?.stop()
    audioRecord?.release()
    audioRecord = null

    recordingThread?.join()
    recordingThread = null

    Log.d(TAG, "Recording stopped")
    audioDataBuffer.clear()
  }

  /**
   * Check if currently recording
   */
  fun isRecording(): Boolean = isRecording

  /**
   * Record audio data and trigger callback every 1.3 seconds
   */
  private fun recordAudioData(bufferSize: Int) {
    val buffer = ByteArray(bufferSize)
    var totalBytes = 0

    try {
      while (isRecording) {
        val bytesRead = audioRecord?.read(buffer, 0, buffer.size) ?: 0
        if (bytesRead > 0) {
          val chunk = buffer.copyOf(bytesRead)
          audioDataBuffer.add(chunk)
          totalBytes += bytesRead

          // Check if we've accumulated enough data for a chunk (1.3 seconds)
          if (totalBytes >= CHUNK_SIZE_BYTES) {
            Log.d(TAG, "Chunk ready: $totalBytes bytes")
            processChunk()
            totalBytes = 0
          }
        }
      }

    } catch (e: Exception) {
      Log.e(TAG, "Error recording audio: ${e.message}", e)
    }
  }

  /**
   * Process accumulated audio chunk: trim silence and trigger transcription
   */
  private fun processChunk() {
    // Convert to FloatArray
    val audioFloatArray = convertToFloatArray(audioDataBuffer)

    // Trim silence from beginning
    val trimmedAudio = trimLeadingSilence(audioFloatArray)

    // Clear buffer for next chunk
    audioDataBuffer.clear()

    // Trigger callback with trimmed audio
    if (trimmedAudio.isNotEmpty()) {
      Log.d(TAG, "Processing chunk: ${trimmedAudio.size} samples (${trimmedAudio.size / SAMPLE_RATE.toFloat()}s)")
      onChunkReady(trimmedAudio)
    } else {
      Log.d(TAG, "Chunk was all silence, skipping")
    }
  }

  /**
   * Trim leading silence from audio
   * Silence threshold: -30dB
   */
  private fun trimLeadingSilence(audio: FloatArray, thresholdDb: Float = -30f): FloatArray {
    val hopLength = SAMPLE_RATE / 20  // 50ms frames
    var startSample = 0

    // Find first non-silent frame
    var i = 0
    while (i < audio.size) {
      val end = minOf(i + hopLength, audio.size)
      val frame = audio.copyOfRange(i, end)
      val energy = frame.map { it * it }.average().toFloat()
      val energyDb = 10 * log10(max(energy, 1e-10f))

      if (energyDb >= thresholdDb) {
        startSample = i
        break
      }

      i += hopLength
    }

    return if (startSample < audio.size) {
      audio.copyOfRange(startSample, audio.size)
    } else {
      FloatArray(0)  // All silence
    }
  }

  /**
   * Convert buffered byte data to FloatArray
   * 16-bit PCM samples are converted to float values in range [-1.0, 1.0]
   */
  private fun convertToFloatArray(audioData: List<ByteArray>): FloatArray {
    // Calculate total samples (each sample is 2 bytes)
    val totalBytes = audioData.sumOf { it.size }
    val totalSamples = totalBytes / 2
    val floatArray = FloatArray(totalSamples)

    var sampleIndex = 0
    for (chunk in audioData) {
      val byteBuffer = ByteBuffer.wrap(chunk).order(ByteOrder.LITTLE_ENDIAN)
      while (byteBuffer.remaining() >= 2) {
        val sample = byteBuffer.short.toInt()
        // Convert 16-bit PCM to float in range [-1.0, 1.0]
        floatArray[sampleIndex++] = sample / 32768f
      }
    }

    return floatArray
  }
}
