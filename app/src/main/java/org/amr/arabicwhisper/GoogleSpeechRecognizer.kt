package org.amr.arabicwhisper

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.util.Log
import java.util.Locale

/**
 * Google Speech Recognition helper for Arabic language
 * Uses Android's built-in SpeechRecognizer API
 */
class GoogleSpeechRecognizer(
  private val context: Context,
  private val onResult: (String) -> Unit,
  private val onError: (String) -> Unit,
  private val onPartialResult: (String) -> Unit = {}
) {

  private var speechRecognizer: SpeechRecognizer? = null
  private var isListening = false
  private var shouldContinueListening = false

  companion object {
    private const val TAG = "#google-speech"
  }

  /**
   * Start listening for Arabic speech
   */
  fun startListening() {
    if (isListening) {
      Log.w(TAG, "Already listening")
      return
    }

    if (!SpeechRecognizer.isRecognitionAvailable(context)) {
      onError("Speech recognition not available on this device")
      Log.e(TAG, "Speech recognition not available")
      return
    }

    shouldContinueListening = true
    startRecognition()
  }

  /**
   * Internal method to start recognition session
   */
  private fun startRecognition() {
    if (!shouldContinueListening) {
      return
    }

    try {
      // Clean up existing recognizer before creating new one
      speechRecognizer?.destroy()
      speechRecognizer = null

      // Initialize speech recognizer
      speechRecognizer = SpeechRecognizer.createSpeechRecognizer(context)
      speechRecognizer?.setRecognitionListener(createRecognitionListener())

      // Create recognition intent
      val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
        putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
        putExtra(RecognizerIntent.EXTRA_LANGUAGE, "ar-SA") // Arabic (Saudi Arabia)
        putExtra(RecognizerIntent.EXTRA_LANGUAGE_PREFERENCE, "ar")
        putExtra(RecognizerIntent.EXTRA_ONLY_RETURN_LANGUAGE_PREFERENCE, "ar")
        putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
        putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
        // Shorter silence timeout for continuous recognition
        putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_COMPLETE_SILENCE_LENGTH_MILLIS, 1000L)
        putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_POSSIBLY_COMPLETE_SILENCE_LENGTH_MILLIS, 1000L)
      }

      isListening = true
      speechRecognizer?.startListening(intent)
      Log.d(TAG, "Started listening for Arabic speech")

    } catch (e: Exception) {
      onError("Failed to start speech recognition: ${e.message}")
      Log.e(TAG, "Failed to start listening", e)
      isListening = false
      shouldContinueListening = false
    }
  }

  /**
   * Stop listening
   */
  fun stopListening() {
    if (!isListening) {
      Log.w(TAG, "Not listening")
      return
    }

    shouldContinueListening = false
    isListening = false
    speechRecognizer?.stopListening()
    Log.d(TAG, "Stopped listening")
  }

  /**
   * Clean up resources
   */
  fun destroy() {
    shouldContinueListening = false
    isListening = false
    speechRecognizer?.destroy()
    speechRecognizer = null
    Log.d(TAG, "Speech recognizer destroyed")
  }

  /**
   * Create recognition listener
   */
  private fun createRecognitionListener() = object : RecognitionListener {
    override fun onReadyForSpeech(params: Bundle?) {
      Log.d(TAG, "Ready for speech")
    }

    override fun onBeginningOfSpeech() {
      Log.d(TAG, "Beginning of speech detected")
    }

    override fun onRmsChanged(rmsdB: Float) {
      // Audio level changed - can be used for visual feedback
    }

    override fun onBufferReceived(buffer: ByteArray?) {
      Log.d(TAG, "Buffer received: ${buffer?.size} bytes")
    }

    override fun onEndOfSpeech() {
      Log.d(TAG, "End of speech")
      isListening = false
      // Don't set shouldContinueListening to false - we want to continue
    }

    override fun onError(error: Int) {
      isListening = false
      val errorMessage = when (error) {
        SpeechRecognizer.ERROR_AUDIO -> "Audio recording error"
        SpeechRecognizer.ERROR_CLIENT -> "Client error"
        SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> "Insufficient permissions"
        SpeechRecognizer.ERROR_NETWORK -> "Network error"
        SpeechRecognizer.ERROR_NETWORK_TIMEOUT -> "Network timeout"
        SpeechRecognizer.ERROR_NO_MATCH -> "No speech match"
        SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> "Recognizer busy"
        SpeechRecognizer.ERROR_SERVER -> "Server error"
        SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> "No speech input"
        else -> "Unknown error: $error"
      }

      Log.e(TAG, "Recognition error: $errorMessage")

      // For certain errors, restart recognition automatically
      when (error) {
        SpeechRecognizer.ERROR_NO_MATCH,
        SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> {
          // These are normal - just restart
          Log.d(TAG, "Restarting recognition after normal timeout/no-match")
          android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
            startRecognition()
          }, 300)
        }
        SpeechRecognizer.ERROR_RECOGNIZER_BUSY,
        SpeechRecognizer.ERROR_SERVER,
        11 -> {
          // Recoverable errors - just restart (error 11 is often transient)
          Log.d(TAG, "Restarting recognition after recoverable error: $error")
          android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
            startRecognition()
          }, 500)
        }
        SpeechRecognizer.ERROR_NETWORK,
        SpeechRecognizer.ERROR_NETWORK_TIMEOUT -> {
          // Network errors - stop and notify user
          shouldContinueListening = false
          onError("$errorMessage - Please check your internet connection")
        }
        SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> {
          // Permission errors - stop and notify user
          shouldContinueListening = false
          onError(errorMessage)
        }
        else -> {
          // Unknown errors - try to recover once with longer delay
          Log.d(TAG, "Attempting to recover from error: $error")
          android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
            if (shouldContinueListening) {
              startRecognition()
            }
          }, 500)
        }
      }
    }

    override fun onResults(results: Bundle?) {
      isListening = false
      val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
      val confidences = results?.getFloatArray(SpeechRecognizer.CONFIDENCE_SCORES)

      if (!matches.isNullOrEmpty()) {
        val bestMatch = matches[0]
        val confidence = confidences?.getOrNull(0) ?: 0f

        Log.d(TAG, "Recognition result: '$bestMatch' (confidence: $confidence)")
        onResult(bestMatch)
      } else {
        Log.d(TAG, "No recognition results")
      }

      // Restart recognition for continuous listening
      if (shouldContinueListening) {
        Log.d(TAG, "Restarting recognition for continuous listening")
        android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
          startRecognition()
        }, 300)
      }
    }

    override fun onPartialResults(partialResults: Bundle?) {
      val matches = partialResults?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)

      if (!matches.isNullOrEmpty()) {
        val partialText = matches[0]
        Log.d(TAG, "Partial result: '$partialText'")
        onPartialResult(partialText)
      }
    }

    override fun onEvent(eventType: Int, params: Bundle?) {
      Log.d(TAG, "Event: $eventType")
    }
  }

  /**
   * Check if currently listening
   */
  fun isListening(): Boolean = isListening
}
