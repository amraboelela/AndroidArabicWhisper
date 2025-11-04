package org.amr.arabicwhisper

import android.content.Context
import android.util.Log
import ai.onnxruntime.*
import kotlinx.serialization.json.*
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.*

class WhisperOnnxHelper(private val context: Context) {

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
  private lateinit var normalizer: JsonObject

  // Audio processing parameters from preprocessor_config.json
  private val SAMPLE_RATE = 16000
  private val N_FFT = 400
  private val HOP_LENGTH = 160
  private val N_MELS = 80
  private val N_SAMPLES = 480000 // 30 seconds

  init {
    // Load native library for mel feature extraction
    System.loadLibrary("whisper_jni")

    initializeOnnx()
    loadTokenizer()
    Log.d("#whisper-onnx", "📱 WhisperOnnxHelper initialized")
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

      // Load normalizer.json
      val normalizerJson = context.assets.open("whisper_onnx/normalizer.json").bufferedReader().readText()
      normalizer = Json.parseToJsonElement(normalizerJson).jsonObject

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

    // Extract mel spectrogram features
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

  private fun extractMelFeatures(audio: FloatArray): Array<FloatArray> {
    // Use native JNI method for proper mel spectrogram extraction
    return extractMelFeaturesNative(audio)
  }

  // Native JNI method for mel feature extraction
  private external fun extractMelFeaturesNative(audioData: FloatArray): Array<FloatArray>

  private fun decodeTokens(tokens: List<Int>): String {
    // Simple decoder - maps token IDs back to text
    // Skip special tokens
    val textTokens = tokens.filter { it < 50257 }

    val reverseVocab = vocab.entries.associate { it.value to it.key }

    val words = textTokens.mapNotNull { reverseVocab[it] }
    val text = words.joinToString("").replace("Ġ", " ").trim()

    return text
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
