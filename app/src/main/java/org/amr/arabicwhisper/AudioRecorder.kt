package org.amr.arabicwhisper

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import kotlin.concurrent.thread

class AudioRecorder(private val outputFile: File) {
  private var audioRecord: AudioRecord? = null
  private var isRecording = false
  private var recordingThread: Thread? = null

  companion object {
    private const val SAMPLE_RATE = 16000
    private const val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
    private const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT
    private const val TAG = "AudioRecorder"
  }

  private val bufferSize = AudioRecord.getMinBufferSize(
    SAMPLE_RATE,
    CHANNEL_CONFIG,
    AUDIO_FORMAT
  )

  fun startRecording() {
    if (isRecording) {
      Log.w(TAG, "Already recording")
      return
    }

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

      audioRecord?.startRecording()
      isRecording = true

      recordingThread = thread(start = true) {
        writeAudioDataToFile()
      }

      Log.d(TAG, "Recording started")
    } catch (e: SecurityException) {
      Log.e(TAG, "Permission denied for audio recording", e)
    } catch (e: Exception) {
      Log.e(TAG, "Error starting recording", e)
    }
  }

  fun stopRecording() {
    if (!isRecording) {
      Log.w(TAG, "Not currently recording")
      return
    }

    isRecording = false
    audioRecord?.stop()
    audioRecord?.release()
    audioRecord = null

    recordingThread?.join()
    recordingThread = null

    Log.d(TAG, "Recording stopped, saved to: ${outputFile.absolutePath}")
  }

  private fun writeAudioDataToFile() {
    val buffer = ByteArray(bufferSize)
    var outputStream: FileOutputStream? = null

    try {
      outputStream = FileOutputStream(outputFile)

      // Write WAV header
      writeWavHeader(outputStream, SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT)

      var totalBytesRead = 0

      while (isRecording) {
        val bytesRead = audioRecord?.read(buffer, 0, buffer.size) ?: 0
        if (bytesRead > 0) {
          outputStream.write(buffer, 0, bytesRead)
          totalBytesRead += bytesRead
        }
      }

      // Update WAV header with actual file size
      outputStream.close()
      updateWavHeader(outputFile, totalBytesRead)

    } catch (e: IOException) {
      Log.e(TAG, "Error writing audio data", e)
    } finally {
      outputStream?.close()
    }
  }

  private fun writeWavHeader(
    out: FileOutputStream,
    sampleRate: Int,
    channelConfig: Int,
    audioFormat: Int
  ) {
    val channels = if (channelConfig == AudioFormat.CHANNEL_IN_MONO) 1 else 2
    val bitsPerSample = if (audioFormat == AudioFormat.ENCODING_PCM_16BIT) 16 else 8

    val byteRate = sampleRate * channels * bitsPerSample / 8
    val blockAlign = (channels * bitsPerSample / 8).toShort()

    out.write("RIFF".toByteArray())
    out.write(intToByteArray(36), 0, 4) // Placeholder for file size - 8
    out.write("WAVE".toByteArray())
    out.write("fmt ".toByteArray())
    out.write(intToByteArray(16), 0, 4) // Sub-chunk size (16 for PCM)
    out.write(shortToByteArray(1), 0, 2) // Audio format (1 = PCM)
    out.write(shortToByteArray(channels.toShort()), 0, 2) // Number of channels
    out.write(intToByteArray(sampleRate), 0, 4) // Sample rate
    out.write(intToByteArray(byteRate), 0, 4) // Byte rate
    out.write(shortToByteArray(blockAlign), 0, 2) // Block align
    out.write(shortToByteArray(bitsPerSample.toShort()), 0, 2) // Bits per sample
    out.write("data".toByteArray())
    out.write(intToByteArray(0), 0, 4) // Placeholder for data chunk size
  }

  private fun updateWavHeader(file: File, dataSize: Int) {
    try {
      val randomAccessFile = java.io.RandomAccessFile(file, "rw")
      randomAccessFile.seek(4)
      randomAccessFile.write(intToByteArray(dataSize + 36), 0, 4)
      randomAccessFile.seek(40)
      randomAccessFile.write(intToByteArray(dataSize), 0, 4)
      randomAccessFile.close()
    } catch (e: IOException) {
      Log.e(TAG, "Error updating WAV header", e)
    }
  }

  private fun intToByteArray(value: Int): ByteArray {
    return byteArrayOf(
      (value and 0xFF).toByte(),
      ((value shr 8) and 0xFF).toByte(),
      ((value shr 16) and 0xFF).toByte(),
      ((value shr 24) and 0xFF).toByte()
    )
  }

  private fun shortToByteArray(value: Short): ByteArray {
    return byteArrayOf(
      (value.toInt() and 0xFF).toByte(),
      ((value.toInt() shr 8) and 0xFF).toByte()
    )
  }

  fun isRecording(): Boolean = isRecording
}
