package org.amr.arabicwhisper

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import kotlin.concurrent.thread

class AudioRecorder(private val onAudioData: (ByteArray) -> Unit) {
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
        readAudioData()
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
    recordingThread?.join()
    audioRecord?.stop()
    audioRecord?.release()
    audioRecord = null
    recordingThread = null

    Log.d(TAG, "Recording stopped")
  }

  private fun readAudioData() {
    val buffer = ByteArray(bufferSize)
    while (isRecording) {
      val bytesRead = audioRecord?.read(buffer, 0, buffer.size) ?: 0
      if (bytesRead > 0) {
        onAudioData(buffer.clone())
      }
    }
  }



  fun isRecording(): Boolean = isRecording
}
