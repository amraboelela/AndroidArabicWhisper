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
    Log.d("#whisper-onnx", "📱 WhisperOnnxKotlinHelper initialized (Kotlin preprocessing + Kotlin ONNX)")
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

      // Try to enable NNAPI (Android Neural Networks API) for GPU/NPU acceleration
      Log.d("#whisper-onnx", "🔧 Attempting to enable NNAPI acceleration...")
      try {
        sessionOptions.addNnapi()
        Log.d("#whisper-onnx", "✅ NNAPI (GPU/NPU acceleration) ENABLED")
      } catch (e: Exception) {
        Log.w("#whisper-onnx", "⚠️ NNAPI not available, using CPU only: ${e.message}")
        e.printStackTrace()
      }

      Log.d("#whisper-onnx", "📦 Loading encoder model from: $encoderPath")
      encoderSession = env?.createSession(encoderPath, sessionOptions)
      Log.d("#whisper-onnx", "📦 Loading decoder model from: $decoderPath")
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
    Log.d("#whisper-onnx", "🎯 transcribeAudio() called with ${audioBytes.size} bytes")

    try {
      // Convert audio bytes to float array
      val audioFloats = FloatArray(audioBytes.size / 2) { i ->
        val sample = ((audioBytes[i * 2 + 1].toInt() shl 8) or (audioBytes[i * 2].toInt() and 0xFF)).toShort()
        sample / 32768.0f
      }
      Log.d("#whisper-onnx", "✅ Converted to ${audioFloats.size} float samples")

      // Extract mel spectrogram features (C++ native)
      Log.d("#whisper-onnx", "🔧 Calling extractMelFeatures()...")
      val melFeatures = extractMelFeatures(audioFloats)
      Log.d("#whisper-onnx", "✅ Got mel features: ${melFeatures.size} x ${melFeatures[0].size}")

      // Run encoder
      Log.d("#whisper-onnx", "🔧 Running ONNX encoder...")
      val encoderStart = System.currentTimeMillis()
      val encoderInputName = encoderSession!!.inputNames.iterator().next()
      val encoderInput = OnnxTensor.createTensor(env, arrayOf(melFeatures))
      val encoderOutputs = encoderSession!!.run(mapOf(encoderInputName to encoderInput))
      val encoderHiddenStates = encoderOutputs[0].value as Array<Array<FloatArray>>
      val encoderTime = System.currentTimeMillis() - encoderStart
      Log.d("#whisper-onnx", "✅ Encoder output shape: ${encoderHiddenStates.size} x ${encoderHiddenStates[0].size} x ${encoderHiddenStates[0][0].size} (${encoderTime}ms)")

      // Run decoder with autoregressive generation
      Log.d("#whisper-onnx", "🔧 Running ONNX decoder...")
      val decoderStart = System.currentTimeMillis()
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

      if (nextToken == eosTokenId) {
        Log.d("#whisper-onnx", "🛑 EOS token reached at step $step")
        break
      }

      generatedTokens.add(nextToken)

      inputIdsTensor.close()
      encoderHiddenStatesTensor.close()
      decoderOutputs.close()
    }

    val decoderTime = System.currentTimeMillis() - decoderStart
    Log.d("#whisper-onnx", "✅ Decoder generated ${generatedTokens.size} tokens (${decoderTime}ms)")

    encoderInput.close()
    encoderOutputs.close()

    // Decode tokens to text
    Log.d("#whisper-onnx", "🔧 Decoding ${generatedTokens.size} tokens to text...")
    val result = decodeTokens(generatedTokens.map { it.toInt() })
    Log.d("#whisper-onnx", "✅ Final transcription: '$result'")

    val totalInferenceTime = encoderTime + decoderTime
    Log.d("#whisper-onnx", "⏱️ TOTAL inference time: ${totalInferenceTime}ms (Encoder: ${encoderTime}ms, Decoder: ${decoderTime}ms)")
    return result
    } catch (e: Exception) {
      Log.e("#whisper-onnx", "❌ transcribeAudio() failed: ${e.message}", e)
      e.printStackTrace()
      return ""
    }
  }

  /**
   * Extract mel spectrogram features using pure Kotlin implementation
   * This matches the Python faster-whisper preprocessing exactly
   */
  private fun extractMelFeatures(audio: FloatArray): Array<FloatArray> {
    val startTime = System.currentTimeMillis()
    Log.d("#whisper-onnx", "========================================")
    Log.d("#whisper-onnx", "🎯 Using Kotlin preprocessing for ${audio.size} samples")
    Log.d("#whisper-onnx", "========================================")

    try {
      // Pad or truncate to N_SAMPLES (30 seconds at 16kHz)
      val paddedAudio = FloatArray(N_SAMPLES)
      val copyLength = min(audio.size, N_SAMPLES)
      audio.copyInto(paddedAudio, 0, 0, copyLength)

      Log.d("#whisper-onnx", "✅ Padded audio to ${paddedAudio.size} samples")

      // Step 1: Compute STFT magnitudes
      Log.d("#whisper-onnx", "🔧 Computing STFT...")
      val stftStart = System.currentTimeMillis()
      val targetFrames = 3000  // Whisper expects exactly 3000 frames
      val stft = computeSTFT(paddedAudio, N_FFT, HOP_LENGTH, targetFrames)
      val stftTime = System.currentTimeMillis() - stftStart
      Log.d("#whisper-onnx", "✅ STFT shape: ${stft.size} x ${stft[0].size} (${stftTime}ms)")

      // Step 2: Apply mel filterbank
      Log.d("#whisper-onnx", "🔧 Applying mel filterbank...")
      val melStart = System.currentTimeMillis()
      val melSpec = applyMelFilterbank(stft)
      val melTime = System.currentTimeMillis() - melStart
      Log.d("#whisper-onnx", "✅ Mel spectrogram shape: ${melSpec.size} x ${melSpec[0].size} (${melTime}ms)")

      // Step 3: Apply log10 transform
      Log.d("#whisper-onnx", "🔧 Applying log10 transform...")
      val logStart = System.currentTimeMillis()
      val logMelSpec = Array(melSpec.size) { i ->
        FloatArray(melSpec[i].size) { j ->
          log10(max(melSpec[i][j], 1e-10f))
        }
      }
      val logTime = System.currentTimeMillis() - logStart
      Log.d("#whisper-onnx", "✅ Log-mel shape: ${logMelSpec.size} x ${logMelSpec[0].size} (${logTime}ms)")

      // Step 4: Apply Whisper normalization (subtract mean, divide by std)
      // These are the global statistics from Whisper training data
      Log.d("#whisper-onnx", "🔧 Applying Whisper normalization...")
      val normStart = System.currentTimeMillis()
      val WHISPER_MEL_MEAN = -4.2677393f
      val WHISPER_MEL_STD = 4.5689974f

      val normalizedMel = Array(logMelSpec.size) { i ->
        FloatArray(logMelSpec[i].size) { j ->
          (logMelSpec[i][j] - WHISPER_MEL_MEAN) / WHISPER_MEL_STD
        }
      }
      val normTime = System.currentTimeMillis() - normStart
      Log.d("#whisper-onnx", "✅ Normalization complete (${normTime}ms)")

      // Log statistics for verification
      val min = normalizedMel.flatMap { it.toList() }.minOrNull() ?: 0f
      val max = normalizedMel.flatMap { it.toList() }.maxOrNull() ?: 0f
      val mean = normalizedMel.flatMap { it.toList() }.average().toFloat()
      val std = sqrt(normalizedMel.flatMap { it.toList() }.map { (it - mean) * (it - mean) }.average()).toFloat()
      Log.d("#whisper-onnx", "📊 Mel stats: min=${"%.6f".format(min)}, max=${"%.6f".format(max)}, mean=${"%.6f".format(mean)}, std=${"%.6f".format(std)}")
      Log.d("#whisper-onnx", "📊 Mel[0] first 10: ${normalizedMel[0].take(10).joinToString(", ") { "%.6f".format(it) }}")
      Log.d("#whisper-onnx", "📊 Mel[40] first 10: ${normalizedMel[40].take(10).joinToString(", ") { "%.6f".format(it) }}")

      val totalTime = System.currentTimeMillis() - startTime
      Log.d("#whisper-onnx", "⏱️ TOTAL Kotlin preprocessing time: ${totalTime}ms (STFT: ${stftTime}ms, Mel: ${melTime}ms, Log: ${logTime}ms, Norm: ${normTime}ms)")
      Log.d("#whisper-onnx", "========================================")

      return normalizedMel
    } catch (e: Exception) {
      Log.e("#whisper-onnx", "❌ Kotlin preprocessing FAILED: ${e.message}", e)
      e.printStackTrace()
      throw RuntimeException("Kotlin mel extraction failed", e)
    }
  }

  /**
   * Compute Short-Time Fourier Transform using JTransforms FFT
   * Uses center padding like Whisper/librosa, and drops last frame like C++ implementation
   */
  private fun computeSTFT(audio: FloatArray, nFFT: Int, hopLength: Int, targetFrames: Int): Array<FloatArray> {
    val fftBins = nFFT / 2 + 1

    // Hann window - match Python's np.hanning(n_fft + 1)[:-1]
    // Create window_size + 1 elements, then drop the last one
    val windowTemp = FloatArray(nFFT + 1) { i ->
      (0.5 * (1.0 - cos(2.0 * PI * i / nFFT))).toFloat()
    }
    val window = windowTemp.copyOfRange(0, nFFT)  // Drop the last element

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

      // Extract magnitude SQUARED (to match C++ and Python librosa with power=2.0)
      for (i in 0 until fftBins) {
        val real = fftInput[i * 2]
        val imag = fftInput[i * 2 + 1]
        val mag = sqrt(real * real + imag * imag)
        magnitudes[i][frame] = mag * mag  // Square the magnitude
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

      // Apply Slaney-style normalization (same as C++ implementation)
      // This normalizes the mel filters to have consistent energy
      val enorm = 2.0f / (endHz - startHz)
      for (j in 0 until fftBins) {
        melFilters[i][j] *= enorm
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

  // GPT-2 BPE byte decoder (from transformers library)
  private val byteDecoder by lazy {
    val bs = mutableListOf<Int>()
    bs.addAll(('!'.code..'~'.code))
    bs.addAll(('¡'.code..'¬'.code))
    bs.addAll(('®'.code..'ÿ'.code))

    val cs = bs.toMutableList()
    var n = 0
    for (b in 0..255) {
      if (b !in bs) {
        bs.add(b)
        cs.add(256 + n)
        n++
      }
    }

    cs.map { it.toChar() }.zip(bs).toMap()
  }

  private fun decodeTokens(tokens: List<Int>): String {
    Log.d("#whisper-onnx", "🔧 Decoding ${tokens.size} total tokens...")

    // Skip special tokens (>= 50257)
    val textTokens = tokens.filter { it < 50257 }

    Log.d("#whisper-onnx", "Text tokens after filtering: ${textTokens.size}")

    if (textTokens.isEmpty()) {
      Log.w("#whisper-onnx", "⚠️ No text tokens found! Only special tokens were generated.")
      return ""
    }

    val reverseVocab = vocab.entries.associate { it.value to it.key }

    // Get BPE token strings
    val tokenStrings = textTokens.mapNotNull {
      val str = reverseVocab[it]
      if (str == null) {
        Log.w("#whisper-onnx", "⚠️ Token $it not found in vocab")
      }
      str
    }

    // Join all tokens
    val joined = tokenStrings.joinToString("")

    try {
      // Decode using GPT-2 BPE byte decoder
      val byteList = mutableListOf<Byte>()
      for (char in joined) {
        val byteVal = byteDecoder[char]
        if (byteVal != null) {
          byteList.add(byteVal.toByte())
        } else {
          Log.w("#whisper-onnx", "Unknown char in BPE: '${char}' (U+${char.code.toString(16)})")
        }
      }

      val decoded = String(byteList.toByteArray(), Charsets.UTF_8)
        .replace("Ġ", " ")  // GPT-2 uses Ġ for space
        .trim()

      Log.d("#whisper-onnx", "✅ Decoded text (${decoded.length} chars): '$decoded'")

      return decoded
    } catch (e: Exception) {
      Log.e("#whisper-onnx", "❌ Token decoding error: ${e.message}", e)
      return joined.replace("Ġ", " ").trim()
    }
  }

  /**
   * Transcribe audio from a WAV file
   */
  fun transcribe(audioFilePath: String): String {
    Log.d("#whisper-onnx", "========================================")
    Log.d("#whisper-onnx", "🎤 transcribe() called")
    Log.d("#whisper-onnx", "📁 File: $audioFilePath")

    try {
      // Read WAV file
      val file = File(audioFilePath)
      if (!file.exists()) {
        Log.e("#whisper-onnx", "❌ File not found: $audioFilePath")
        return "Error: File not found"
      }

      Log.d("#whisper-onnx", "✅ File exists, size: ${file.length()} bytes")

      // Read all bytes and skip WAV header (44 bytes)
      val allBytes = file.readBytes()
      val audioBytes = allBytes.copyOfRange(44, allBytes.size)

      Log.d("#whisper-onnx", "✅ Read WAV file: ${audioBytes.size} bytes (after skipping 44-byte header)")

      // Transcribe the audio
      Log.d("#whisper-onnx", "🔧 Calling transcribeAudio()...")
      val result = transcribeAudio(audioBytes)
      Log.d("#whisper-onnx", "🏁 transcribe() returning: '$result'")
      Log.d("#whisper-onnx", "========================================")
      return result
    } catch (e: Exception) {
      Log.e("#whisper-onnx", "❌ transcribe() failed: ${e.message}", e)
      e.printStackTrace()
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
