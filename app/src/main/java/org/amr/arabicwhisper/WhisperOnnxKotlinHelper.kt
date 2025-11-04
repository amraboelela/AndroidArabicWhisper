package org.amr.arabicwhisper

import android.content.Context
import android.util.Log
import ai.onnxruntime.*
import kotlinx.serialization.json.*
import org.jtransforms.fft.FloatFFT_1D
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.*

class WhisperOnnxKotlinHelper(private val context: Context) {

  var onTranscriptionUpdate: ((String) -> Unit)? = null
  var onProcessingStateChange: ((Boolean) -> Unit)? = null
  private val executorService: ExecutorService = Executors.newSingleThreadExecutor()

  // Buffer for accumulating audio chunks
  private val audioBuffer = mutableListOf<Byte>()
  private val bufferLock = Any()

  // 4 seconds at 16kHz, 16-bit = 128000 bytes
  private val CHUNK_SIZE_BYTES = 128000
  private var isProcessing = false

  // ONNX Runtime sessions
  private var encoderSession: OrtSession? = null
  private var decoderSession: OrtSession? = null
  private var env: OrtEnvironment? = null

  // Tokenizer data
  private lateinit var vocab: Map<String, Int>
  private lateinit var merges: List<Pair<String, String>>
  private lateinit var addedTokens: Map<String, Int>

  // Audio processing parameters (Whisper standard)
  private val SAMPLE_RATE = 16000
  private val N_FFT = 400
  private val HOP_LENGTH = 160
  private val N_MELS = 80
  private val N_SAMPLES = 480000 // 30 seconds (model expects 3000 frames)
  private val MEL_FILTERS by lazy { createMelFilterbank() }

  init {
    initializeOnnx()
    loadTokenizer()
    Log.d("#whisper-onnx", "📱 WhisperOnnxKotlinHelper initialized (pure Kotlin with FFT)")
  }

  private fun initializeOnnx() {
    try {
      env = OrtEnvironment.getEnvironment()

      // Copy ONNX models from assets to internal storage if needed
      val modelDir = File(context.filesDir, "whisper_onnx")
      if (!modelDir.exists()) {
        modelDir.mkdirs()
        copyAssetFile("whisper_onnx/encoder_model.onnx", File(modelDir, "encoder_model.onnx"))
        copyAssetFile("whisper_onnx/decoder_model.onnx", File(modelDir, "decoder_model.onnx"))
      }

      val encoderPath = File(modelDir, "encoder_model.onnx").absolutePath
      val decoderPath = File(modelDir, "decoder_model.onnx").absolutePath

      val sessionOptions = OrtSession.SessionOptions()
      sessionOptions.setIntraOpNumThreads(4)
      sessionOptions.setInterOpNumThreads(4)

      encoderSession = env?.createSession(encoderPath, sessionOptions)
      decoderSession = env?.createSession(decoderPath, sessionOptions)

      Log.d("#whisper-onnx", "✅ ONNX models loaded successfully")
    } catch (e: Exception) {
      Log.e("#whisper-onnx", "❌ Failed to initialize ONNX: ${e.message}", e)
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

  private fun loadTokenizer() {
    try {
      // Load vocab.json
      val vocabJson = context.assets.open("whisper_onnx/vocab.json").bufferedReader().readText()
      vocab = Json.parseToJsonElement(vocabJson).jsonObject.mapValues { it.value.jsonPrimitive.int }

      // Load merges.txt
      val mergesText = context.assets.open("whisper_onnx/merges.txt").bufferedReader().readLines()
      merges = mergesText.drop(1).map { line ->
        val parts = line.split(" ")
        Pair(parts[0], parts[1])
      }

      // Load added_tokens.json
      val addedTokensJson = context.assets.open("whisper_onnx/added_tokens.json").bufferedReader().readText()
      addedTokens = Json.parseToJsonElement(addedTokensJson).jsonObject.mapValues { it.value.jsonPrimitive.int }

      Log.d("#whisper-onnx", "✅ Tokenizer loaded: ${vocab.size} vocab, ${merges.size} merges")
    } catch (e: Exception) {
      Log.e("#whisper-onnx", "❌ Failed to load tokenizer: ${e.message}", e)
      throw e
    }
  }

  fun transcribeStream(audioData: ByteArray, onResult: (String) -> Unit) {
    synchronized(bufferLock) {
      audioBuffer.addAll(audioData.toList())

      val bufferSize = audioBuffer.size
      val bufferSeconds = bufferSize / 32000.0f

      if (bufferSize >= CHUNK_SIZE_BYTES && !isProcessing) {
        // Check if audio has significant energy
        val audioFloats = audioBuffer.toByteArray().let { bytes ->
          FloatArray(bytes.size / 2) { i ->
            val sample = ((bytes[i * 2 + 1].toInt() shl 8) or (bytes[i * 2].toInt() and 0xFF)).toShort()
            sample / 32768.0f
          }
        }

        val rms = sqrt(audioFloats.map { it * it }.average()).toFloat()
        val silenceThreshold = 0.025f

        if (rms < silenceThreshold) {
          Log.d("#whisper-onnx", "🔇 Silence detected (RMS: %.4f), skipping".format(rms))
          audioBuffer.clear()
          return
        }

        Log.d("#whisper-onnx", "🔊 Audio detected (RMS: %.4f)".format(rms))

        isProcessing = true
        onProcessingStateChange?.invoke(true)

        val chunkToProcess = audioBuffer.toByteArray()
        audioBuffer.clear()

        executorService.execute {
          try {
            val result = transcribeAudio(chunkToProcess)
            if (result.isNotEmpty()) {
              Log.d("#whisper-onnx", "✅ Transcription: $result")
              onResult(result)
            }
          } catch (e: Exception) {
            Log.e("#whisper-onnx", "❌ Transcription error", e)
          } finally {
            synchronized(bufferLock) {
              isProcessing = false
              onProcessingStateChange?.invoke(false)

              if (audioBuffer.size >= CHUNK_SIZE_BYTES) {
                transcribeStream(ByteArray(0), onResult)
              }
            }
          }
        }
      }
    }
  }

  private fun transcribeAudio(audioBytes: ByteArray): String {
    // Convert audio bytes to float array
    val audioFloats = FloatArray(audioBytes.size / 2) { i ->
      val sample = ((audioBytes[i * 2 + 1].toInt() shl 8) or (audioBytes[i * 2].toInt() and 0xFF)).toShort()
      sample / 32768.0f
    }

    // Extract mel spectrogram features (pure Kotlin)
    val melFeatures = extractMelFeatures(audioFloats)

    // Run encoder
    val encoderInputName = encoderSession!!.inputNames.iterator().next()
    val encoderInput = OnnxTensor.createTensor(env, arrayOf(melFeatures))
    val encoderOutputs = encoderSession!!.run(mapOf(encoderInputName to encoderInput))
    val encoderHiddenStates = encoderOutputs[0].value as Array<Array<FloatArray>>

    // Run decoder with autoregressive generation
    val decoderStartTokenId = 50258L  // <|startoftranscript|>
    val langTokenId = 50272L           // <|ar|>
    val taskTokenId = 50359L           // <|transcribe|>
    val noTimestampsTokenId = 50363L   // <|notimestamps|>
    val eosTokenId = 50257L            // <|endoftext|>

    val generatedTokens = mutableListOf(decoderStartTokenId, langTokenId, taskTokenId, noTimestampsTokenId)
    val maxLength = 200

    for (step in 0 until maxLength) {
      // Prepare decoder inputs
      val inputIds = Array(1) { generatedTokens.toLongArray() }
      val inputIdsTensor = OnnxTensor.createTensor(env, inputIds)
      val encoderHiddenStatesTensor = OnnxTensor.createTensor(env, encoderHiddenStates)

      val decoderInputs = mapOf(
        "input_ids" to inputIdsTensor,
        "encoder_hidden_states" to encoderHiddenStatesTensor
      )

      val decoderOutputs = decoderSession!!.run(decoderInputs)
      val logits = decoderOutputs[0].value as Array<Array<FloatArray>>

      // Get next token (greedy decoding)
      val lastLogits = logits[0].last()
      val nextToken = lastLogits.indices.maxByOrNull { lastLogits[it] }?.toLong() ?: eosTokenId

      if (nextToken == eosTokenId) break

      generatedTokens.add(nextToken)

      inputIdsTensor.close()
      encoderHiddenStatesTensor.close()
      decoderOutputs.close()
    }

    encoderInput.close()
    encoderOutputs.close()

    // Decode tokens to text
    return decodeTokens(generatedTokens.map { it.toInt() })
  }

  /**
   * Pure Kotlin mel spectrogram extraction
   * Based on Whisper's preprocessing pipeline
   */
  private fun extractMelFeatures(audio: FloatArray): Array<FloatArray> {
    Log.d("#whisper-onnx", "Extracting mel features from ${audio.size} samples")

    // Pad or truncate to N_SAMPLES
    val paddedAudio = FloatArray(N_SAMPLES)
    val copyLength = min(audio.size, N_SAMPLES)
    audio.copyInto(paddedAudio, 0, 0, copyLength)

    Log.d("#whisper-onnx", "Padded audio to ${paddedAudio.size} samples")

    // Compute STFT with standard formula, then drop last frame like Whisper does
    val nFrames = N_SAMPLES / HOP_LENGTH  // 1500 frames expected
    val stft = computeSTFT(paddedAudio, N_FFT, HOP_LENGTH, nFrames)

    Log.d("#whisper-onnx", "STFT shape: ${stft.size} x ${stft[0].size}")

    // Apply mel filterbank
    val melSpectrogram = applyMelFilterbank(stft)

    Log.d("#whisper-onnx", "Mel spectrogram shape: ${melSpectrogram.size} x ${melSpectrogram[0].size}")

    // Convert to log scale - use actual frame count from STFT
    val actualFrames = stft[0].size
    val logMelSpec = Array(N_MELS) { FloatArray(actualFrames) }
    for (i in 0 until N_MELS) {
      for (j in 0 until actualFrames) {
        logMelSpec[i][j] = ln(max(melSpectrogram[i][j], 1e-10f))
      }
    }

    // Normalize (optional, Whisper does this)
    val mean = logMelSpec.flatMap { it.toList() }.average().toFloat()
    val std = sqrt(logMelSpec.flatMap { it.toList() }.map { (it - mean) * (it - mean) }.average()).toFloat()

    for (i in 0 until N_MELS) {
      for (j in 0 until actualFrames) {
        logMelSpec[i][j] = (logMelSpec[i][j] - mean) / (std + 1e-5f)
      }
    }

    Log.d("#whisper-onnx", "Mel features shape: ${logMelSpec.size} x ${logMelSpec[0].size}")
    return logMelSpec
  }

  /**
   * Compute Short-Time Fourier Transform using JTransforms FFT
   * Uses center padding like Whisper/librosa, and drops last frame like C++ implementation
   */
  private fun computeSTFT(audio: FloatArray, nFFT: Int, hopLength: Int, targetFrames: Int): Array<FloatArray> {
    val fftBins = nFFT / 2 + 1

    // Hann window
    val window = FloatArray(nFFT) { i ->
      (0.5 * (1.0 - cos(2.0 * PI * i / (nFFT - 1)))).toFloat()
    }

    // Apply center padding (pad audio by N_FFT // 2 on both sides)
    val padSize = nFFT / 2
    val paddedAudio = FloatArray(audio.size + 2 * padSize)
    audio.copyInto(paddedAudio, padSize, 0, audio.size)

    Log.d("#whisper-onnx", "STFT: input=${audio.size}, padded=${paddedAudio.size}, padSize=$padSize")

    // Calculate number of frames using standard STFT formula
    val numFramesBeforeDrop = (paddedAudio.size - nFFT) / hopLength + 1

    Log.d("#whisper-onnx", "STFT: numFramesBeforeDrop=$numFramesBeforeDrop, targetFrames=$targetFrames")

    // Compute one extra frame that we'll drop (to match C++ whisper implementation)
    val magnitudes = Array(fftBins) { FloatArray(numFramesBeforeDrop) }

    // Create FFT instance (reuse for efficiency)
    val fft = FloatFFT_1D(nFFT.toLong())

    for (frame in 0 until numFramesBeforeDrop) {
      val offset = frame * hopLength

      // Prepare complex array for FFT (real, imag interleaved)
      val fftInput = FloatArray(nFFT * 2)

      // Apply window and copy to FFT input
      for (i in 0 until nFFT) {
        fftInput[i * 2] = paddedAudio[offset + i] * window[i]  // Real part
        fftInput[i * 2 + 1] = 0.0f  // Imaginary part
      }

      // Compute FFT (in-place)
      fft.complexForward(fftInput)

      // Extract magnitude spectrum
      for (i in 0 until fftBins) {
        val real = fftInput[i * 2]
        val imag = fftInput[i * 2 + 1]
        magnitudes[i][frame] = sqrt(real * real + imag * imag)
      }
    }

    // Drop the last frame to match C++ whisper implementation
    // Return only the first targetFrames frames
    val finalMagnitudes = Array(fftBins) { i ->
      magnitudes[i].copyOfRange(0, targetFrames)
    }

    Log.d("#whisper-onnx", "STFT: returning ${finalMagnitudes.size} x ${finalMagnitudes[0].size}")

    return finalMagnitudes
  }

  /**
   * Create mel filterbank matrix
   */
  private fun createMelFilterbank(): Array<FloatArray> {
    val fftBins = N_FFT / 2 + 1
    val melFilters = Array(N_MELS) { FloatArray(fftBins) }

    // Frequency to mel conversion
    fun hzToMel(hz: Float): Float = 2595.0f * log10(1.0f + hz / 700.0f)
    fun melToHz(mel: Float): Float = 700.0f * (10.0f.pow(mel / 2595.0f) - 1.0f)

    val melMin = hzToMel(0.0f)
    val melMax = hzToMel(SAMPLE_RATE / 2.0f)

    // Create mel points
    val melPoints = FloatArray(N_MELS + 2) { i ->
      melMin + (melMax - melMin) * i / (N_MELS + 1)
    }

    val hzPoints = melPoints.map { melToHz(it) }

    // Create filterbank
    for (i in 0 until N_MELS) {
      val startHz = hzPoints[i]
      val centerHz = hzPoints[i + 1]
      val endHz = hzPoints[i + 2]

      for (j in 0 until fftBins) {
        val freq = j * SAMPLE_RATE.toFloat() / N_FFT

        if (freq >= startHz && freq <= endHz) {
          if (freq <= centerHz) {
            melFilters[i][j] = (freq - startHz) / (centerHz - startHz)
          } else {
            melFilters[i][j] = (endHz - freq) / (endHz - centerHz)
          }
        }
      }
    }

    return melFilters
  }

  /**
   * Apply mel filterbank to STFT magnitudes
   */
  private fun applyMelFilterbank(stft: Array<FloatArray>): Array<FloatArray> {
    val nFrames = stft[0].size
    val melSpec = Array(N_MELS) { FloatArray(nFrames) }

    for (i in 0 until N_MELS) {
      for (j in 0 until nFrames) {
        var sum = 0.0f
        for (k in MEL_FILTERS[i].indices) {
          sum += MEL_FILTERS[i][k] * stft[k][j]
        }
        melSpec[i][j] = sum
      }
    }

    return melSpec
  }

  private fun decodeTokens(tokens: List<Int>): String {
    // Simple decoder - maps token IDs back to text
    // Skip special tokens
    val textTokens = tokens.filter { it < 50257 }

    val reverseVocab = vocab.entries.associate { it.value to it.key }

    val words = textTokens.mapNotNull { reverseVocab[it] }
    val text = words.joinToString("").replace("Ġ", " ").trim()

    return text
  }

  /**
   * Transcribe audio from a WAV file
   */
  fun transcribe(audioFilePath: String): String {
    try {
      // Read WAV file
      val file = File(audioFilePath)
      if (!file.exists()) {
        return "Error: File not found"
      }

      // Read all bytes and skip WAV header (44 bytes)
      val allBytes = file.readBytes()
      val audioBytes = allBytes.copyOfRange(44, allBytes.size)

      Log.d("#whisper-onnx", "Transcribing file: $audioFilePath (${audioBytes.size} bytes)")

      // Transcribe the audio
      return transcribeAudio(audioBytes)
    } catch (e: Exception) {
      Log.e("#whisper-onnx", "Transcription error", e)
      return "Error: ${e.message}"
    }
  }

  fun clearTranscription() {
    synchronized(bufferLock) {
      audioBuffer.clear()
      isProcessing = false
    }
    onProcessingStateChange?.invoke(false)
  }

  fun shutdown() {
    executorService.shutdown()
    encoderSession?.close()
    decoderSession?.close()
    env?.close()
  }
}
