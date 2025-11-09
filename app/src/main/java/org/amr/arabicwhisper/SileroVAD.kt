package org.amr.arabicwhisper

import android.content.Context
import android.util.Log
import ai.onnxruntime.*
import java.io.File
import java.io.FileOutputStream
import kotlin.math.max

/**
 * Silero VAD v5 - Voice Activity Detection using ONNX Runtime
 * Port of faster-whisper's VAD implementation
 */
class SileroVAD(private val context: Context) {
  private var encoderSession: OrtSession? = null
  private var decoderSession: OrtSession? = null
  private var env: OrtEnvironment? = null

  data class VadOptions(
    val threshold: Float = 0.5f,
    val negThreshold: Float? = null,
    val minSpeechDurationMs: Int = 0,
    val maxSpeechDurationS: Float = Float.POSITIVE_INFINITY,
    val minSilenceDurationMs: Int = 2000,
    val speechPadMs: Int = 400
  )

  data class SpeechSegment(
    val start: Int,  // Sample index
    val end: Int     // Sample index
  )

  init {
    initializeVAD()
  }

  private fun initializeVAD() {
    try {
      env = OrtEnvironment.getEnvironment()

      // Copy ONNX models from assets to internal storage
      val vadDir = File(context.filesDir, "silero_vad")
      if (!vadDir.exists()) {
        vadDir.mkdirs()
        copyAssetFile("silero_vad/silero_encoder_v5.onnx", File(vadDir, "silero_encoder_v5.onnx"))
        copyAssetFile("silero_vad/silero_decoder_v5.onnx", File(vadDir, "silero_decoder_v5.onnx"))
      }

      val encoderPath = File(vadDir, "silero_encoder_v5.onnx").absolutePath
      val decoderPath = File(vadDir, "silero_decoder_v5.onnx").absolutePath

      val sessionOptions = OrtSession.SessionOptions()
      sessionOptions.setIntraOpNumThreads(1)
      sessionOptions.setInterOpNumThreads(1)

      encoderSession = env?.createSession(encoderPath, sessionOptions)
      decoderSession = env?.createSession(decoderPath, sessionOptions)

      Log.d("#silero_vad", "✅ Silero VAD initialized")
    } catch (e: Exception) {
      Log.e("#silero_vad", "❌ Failed to initialize Silero VAD: ${e.message}", e)
      throw e
    }
  }

  private fun copyAssetFile(assetPath: String, dest: File) {
    context.assets.open(assetPath).use { input ->
      FileOutputStream(dest).use { output ->
        input.copyTo(output)
      }
    }
  }

  /**
   * Get speech timestamps from audio using Silero VAD
   * @param audio Float array of audio samples (16kHz, mono)
   * @param samplingRate Sample rate (default 16000)
   * @param options VAD options
   * @return List of speech segments with start/end sample indices
   */
  fun getSpeechTimestamps(
    audio: FloatArray,
    samplingRate: Int = 16000,
    options: VadOptions = VadOptions()
  ): List<SpeechSegment> {
    val windowSizeSamples = 512
    val contextSizeSamples = 64

    // Pad audio to multiple of window size
    val paddingSize = windowSizeSamples - (audio.size % windowSizeSamples)
    val paddedAudio = FloatArray(audio.size + paddingSize)
    audio.copyInto(paddedAudio)

    // Reshape to (1, num_samples)
    val audioReshaped = Array(1) { paddedAudio }

    // Get speech probabilities from VAD model
    val speechProbs = runVADModel(audioReshaped, windowSizeSamples, contextSizeSamples)

    // Process speech probabilities to find segments
    return processVADOutput(speechProbs, audio.size, samplingRate, windowSizeSamples, options)
  }

  /**
   * Run Silero VAD encoder-decoder model
   */
  private fun runVADModel(
    audio: Array<FloatArray>,
    numSamples: Int,
    contextSize: Int
  ): FloatArray {
    val batchSize = audio.size
    val totalSamples = audio[0].size

    // Initialize state and context
    val state = Array(2) { Array(batchSize) { FloatArray(128) } }
    val context = Array(batchSize) { FloatArray(contextSize) }

    // Reshape audio into chunks
    val numChunks = totalSamples / numSamples
    val batchedAudio = mutableListOf<FloatArray>()

    for (chunk in 0 until numChunks) {
      val chunkWithContext = FloatArray(numSamples + contextSize)

      // Add context from previous chunk
      context[0].copyInto(chunkWithContext, 0)

      // Add current chunk
      audio[0].copyInto(
        chunkWithContext,
        contextSize,
        chunk * numSamples,
        (chunk + 1) * numSamples
      )

      batchedAudio.add(chunkWithContext)

      // Update context for next chunk
      audio[0].copyInto(
        context[0],
        0,
        (chunk + 1) * numSamples - contextSize,
        (chunk + 1) * numSamples
      )
    }

    // Run encoder in batches
    val encoderOutputs = mutableListOf<Array<Array<FloatArray>>>()
    val encoderBatchSize = 10000

    for (i in batchedAudio.indices step encoderBatchSize) {
      val batchEnd = kotlin.math.min(i + encoderBatchSize, batchedAudio.size)
      val batch = batchedAudio.subList(i, batchEnd).toTypedArray()

      val encoderInput = OnnxTensor.createTensor(env, batch)
      val encoderResult = encoderSession!!.run(mapOf("input" to encoderInput))
      val output = encoderResult[0].value as Array<Array<FloatArray>>

      encoderOutputs.add(output)
      encoderInput.close()
      encoderResult.close()
    }

    // Concatenate encoder outputs
    val allEncoderOutputs = encoderOutputs.flatMap { it.toList() }.toTypedArray()

    // Run decoder
    val decoderOutputs = mutableListOf<Float>()
    var currentState = state

    for (window in allEncoderOutputs) {
      val decoderInput = OnnxTensor.createTensor(env, arrayOf(window))
      val stateInput = OnnxTensor.createTensor(env, currentState)

      val decoderResult = decoderSession!!.run(
        mapOf("input" to decoderInput, "state" to stateInput)
      )

      val output = decoderResult[0].value as Array<FloatArray>
      val newState = decoderResult[1].value as Array<Array<FloatArray>>

      decoderOutputs.add(output[0][0])
      currentState = newState

      decoderInput.close()
      stateInput.close()
      decoderResult.close()
    }

    return decoderOutputs.toFloatArray()
  }

  /**
   * Process VAD output probabilities to find speech segments
   */
  private fun processVADOutput(
    speechProbs: FloatArray,
    audioLength: Int,
    samplingRate: Int,
    windowSize: Int,
    options: VadOptions
  ): List<SpeechSegment> {
    val threshold = options.threshold
    val negThreshold = options.negThreshold ?: max(threshold - 0.15f, 0.01f)
    val minSpeechSamples = samplingRate * options.minSpeechDurationMs / 1000
    val speechPadSamples = samplingRate * options.speechPadMs / 1000
    val maxSpeechSamples = (samplingRate * options.maxSpeechDurationS - windowSize - 2 * speechPadSamples).toInt()
    val minSilenceSamples = samplingRate * options.minSilenceDurationMs / 1000
    val minSilenceSamplesAtMaxSpeech = samplingRate * 98 / 1000

    var triggered = false
    val speeches = mutableListOf<SpeechSegment>()
    var currentSpeechStart = 0
    var tempEnd = 0
    var prevEnd = 0
    var nextStart = 0

    for (i in speechProbs.indices) {
      val speechProb = speechProbs[i]

      if (speechProb >= threshold && tempEnd > 0) {
        tempEnd = 0
        if (nextStart < prevEnd) {
          nextStart = windowSize * i
        }
      }

      if (speechProb >= threshold && !triggered) {
        triggered = true
        currentSpeechStart = windowSize * i
        continue
      }

      if (triggered && (windowSize * i - currentSpeechStart) > maxSpeechSamples) {
        val currentEnd = if (prevEnd > 0) {
          val end = prevEnd
          if (nextStart < prevEnd) {
            triggered = false
          } else {
            currentSpeechStart = nextStart
          }
          prevEnd = 0
          nextStart = 0
          tempEnd = 0
          end
        } else {
          triggered = false
          prevEnd = 0
          nextStart = 0
          tempEnd = 0
          windowSize * i
        }

        speeches.add(SpeechSegment(currentSpeechStart, currentEnd))
        continue
      }

      if (speechProb < negThreshold && triggered) {
        if (tempEnd == 0) {
          tempEnd = windowSize * i
        }

        if ((windowSize * i) - tempEnd > minSilenceSamplesAtMaxSpeech) {
          prevEnd = tempEnd
        }

        if ((windowSize * i) - tempEnd < minSilenceSamples) {
          continue
        } else {
          if ((tempEnd - currentSpeechStart) > minSpeechSamples) {
            speeches.add(SpeechSegment(currentSpeechStart, tempEnd))
          }
          currentSpeechStart = 0
          prevEnd = 0
          nextStart = 0
          tempEnd = 0
          triggered = false
        }
      }
    }

    // Add final segment if still triggered
    if (triggered && (audioLength - currentSpeechStart) > minSpeechSamples) {
      speeches.add(SpeechSegment(currentSpeechStart, audioLength))
    }

    // Apply padding
    return applyPadding(speeches, audioLength, speechPadSamples.toInt())
  }

  /**
   * Apply padding to speech segments
   */
  private fun applyPadding(
    speeches: List<SpeechSegment>,
    audioLength: Int,
    speechPadSamples: Int
  ): List<SpeechSegment> {
    if (speeches.isEmpty()) return speeches

    val paddedSpeeches = mutableListOf<SpeechSegment>()

    for (i in speeches.indices) {
      var start = speeches[i].start
      var end = speeches[i].end

      if (i == 0) {
        start = max(0, start - speechPadSamples)
      }

      if (i != speeches.size - 1) {
        val silenceDuration = speeches[i + 1].start - end
        if (silenceDuration < 2 * speechPadSamples) {
          end += silenceDuration / 2
        } else {
          end = kotlin.math.min(audioLength, end + speechPadSamples)
        }
      } else {
        end = kotlin.math.min(audioLength, end + speechPadSamples)
      }

      paddedSpeeches.add(SpeechSegment(start, end))
    }

    // Adjust overlapping segments
    for (i in 0 until paddedSpeeches.size - 1) {
      val silenceDuration = paddedSpeeches[i + 1].start - paddedSpeeches[i].end
      if (silenceDuration < 2 * speechPadSamples) {
        paddedSpeeches[i + 1] = SpeechSegment(
          paddedSpeeches[i].end + silenceDuration / 2,
          paddedSpeeches[i + 1].end
        )
      } else {
        paddedSpeeches[i + 1] = SpeechSegment(
          max(0, paddedSpeeches[i + 1].start - speechPadSamples),
          paddedSpeeches[i + 1].end
        )
      }
    }

    return paddedSpeeches
  }

  fun cleanup() {
    encoderSession?.close()
    decoderSession?.close()
    env?.close()
  }
}
