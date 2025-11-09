package org.amr.arabicwhisper

import android.content.Context
import android.util.Log
import ai.onnxruntime.*
import kotlinx.serialization.json.*
import org.jtransforms.fft.FloatFFT_1D
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import kotlin.math.*

/**
 * Muhaffez Whisper Helper - Word-level Arabic Quran transcription
 * Uses custom trained model with word-based vocabulary (not BPE)
 */
class MuhaffezWhisperHelper(private val context: Context) {

  // ONNX Runtime sessions
  private var encoderSession: OrtSession? = null
  private var decoderSession: OrtSession? = null
  private var env: OrtEnvironment? = null

  // Silero VAD for voice activity detection
  private var sileroVAD: SileroVAD? = null

  // Word-level vocabulary
  private lateinit var vocabulary: List<String>
  private lateinit var wordToIdx: Map<String, Int>

  // Audio processing parameters (Whisper standard)
  private val SAMPLE_RATE = 16000
  private val N_FFT = 400
  private val HOP_LENGTH = 160
  private val N_MELS = 80
  private val N_SAMPLES = 480000 // 30 seconds (model expects 3000 frames)
  private val MEL_FILTERS by lazy { createMelFilterbank() }

  // Special tokens
  private val SOS_TOKEN = 1L  // <s> start of sequence
  private val EOS_TOKEN = 2L  // </s> end of sequence

  init {
    Log.d("#muhaffez", "Initializing Muhaffez Whisper Helper...")
    initializeOnnx()
    loadVocabulary()
    try {
      sileroVAD = SileroVAD(context)
      Log.d("#muhaffez", "✅ Silero VAD initialized successfully")
    } catch (e: Exception) {
      Log.w("#muhaffez", "⚠️ Silero VAD initialization failed, will use fallback: ${e.message}")
      sileroVAD = null
    }
    Log.d("#muhaffez", "✅ Muhaffez Whisper Helper initialized")
  }

  private fun initializeOnnx() {
    try {
      env = OrtEnvironment.getEnvironment()

      // Copy ONNX models from assets to internal storage if needed
      val modelDir = File(context.filesDir, "muhaffez_whisper")
      if (!modelDir.exists()) {
        modelDir.mkdirs()
        copyAssetFile("muhaffez_whisper/encoder_model.onnx", File(modelDir, "encoder_model.onnx"))
        copyAssetFile("muhaffez_whisper/decoder_model.onnx", File(modelDir, "decoder_model.onnx"))
      }

      val encoderPath = File(modelDir, "encoder_model.onnx").absolutePath
      val decoderPath = File(modelDir, "decoder_model.onnx").absolutePath

      val sessionOptions = OrtSession.SessionOptions()
      sessionOptions.setIntraOpNumThreads(4)
      sessionOptions.setInterOpNumThreads(4)

      // Try to enable NNAPI for better performance
      try {
        sessionOptions.addNnapi()
        Log.d("#muhaffez", "✅ NNAPI enabled")
      } catch (e: Exception) {
        Log.w("#muhaffez", "NNAPI not available, using CPU")
      }

      encoderSession = env?.createSession(encoderPath, sessionOptions)
      decoderSession = env?.createSession(decoderPath, sessionOptions)

      Log.d("#muhaffez", "✅ ONNX models loaded")
    } catch (e: Exception) {
      Log.e("#muhaffez", "❌ Failed to initialize ONNX: ${e.message}", e)
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

  private fun loadVocabulary() {
    try {
      val vocabJson = context.assets.open("muhaffez_whisper/vocabulary.json").bufferedReader().readText()
      vocabulary = Json.parseToJsonElement(vocabJson).jsonArray.map { it.jsonPrimitive.content }
      wordToIdx = vocabulary.withIndex().associate { it.value to it.index }

      Log.d("#muhaffez", "✅ Vocabulary loaded: ${vocabulary.size} words")
    } catch (e: Exception) {
      Log.e("#muhaffez", "❌ Failed to load vocabulary: ${e.message}", e)
      throw e
    }
  }

  /**
   * Transcribe audio segment
   * @param audio Float array of audio samples (16kHz, mono)
   * @return Transcribed text
   */
  fun transcribe(audio: FloatArray): String {
    try {
      Log.d("#muhaffez", "🎙️ Transcribing ${audio.size} samples...")

      // Extract mel features
      val melFeatures = extractMelFeatures(audio)

      // Transpose to (1, n_mels=80, time)
      val transposedMel = Array(1) { Array(N_MELS) { FloatArray(melFeatures[0].size) } }
      for (i in 0 until N_MELS) {
        for (j in melFeatures[i].indices) {
          transposedMel[0][i][j] = melFeatures[i][j]
        }
      }

      // Run encoder
      val encoderStart = System.currentTimeMillis()
      val encoderInput = OnnxTensor.createTensor(
        env,
        transposedMel
      )

      val encoderOutputs = encoderSession!!.run(mapOf("input_features" to encoderInput))
      val encoderHiddenStates = encoderOutputs[0].value as Array<Array<FloatArray>>
      val encoderTime = System.currentTimeMillis() - encoderStart

      Log.d("#muhaffez", "✅ Encoder completed (${encoderTime}ms)")

      // Run decoder with greedy decoding
      val decoderStart = System.currentTimeMillis()
      val generatedTokens = mutableListOf<Long>()
      val maxTokens = 50  // Maximum number of tokens to generate

      for (step in 0 until maxTokens) {
        // Build input_ids: [SOS] + generated tokens so far
        val inputIds = Array(1) { LongArray(1 + generatedTokens.size) }
        inputIds[0][0] = SOS_TOKEN
        for (i in generatedTokens.indices) {
          inputIds[0][i + 1] = generatedTokens[i]
        }

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
        val nextToken = lastLogits.indices.maxByOrNull { lastLogits[it] }?.toLong() ?: EOS_TOKEN

        if (nextToken == EOS_TOKEN) {
          break
        }

        generatedTokens.add(nextToken)

        inputIdsTensor.close()
        encoderHiddenStatesTensor.close()
        decoderOutputs.close()
      }

      val decoderTime = System.currentTimeMillis() - decoderStart
      Log.d("#muhaffez", "✅ Decoder generated ${generatedTokens.size} tokens (${decoderTime}ms)")

      encoderInput.close()
      encoderOutputs.close()

      // Decode tokens to text (word-level vocabulary, just join with spaces)
      val result = generatedTokens.map { idx ->
        if (idx.toInt() < vocabulary.size) vocabulary[idx.toInt()] else "<unk>"
      }.joinToString(" ")

      Log.d("#muhaffez", "✅ Transcription: '$result'")
      return result
    } catch (e: Exception) {
      Log.e("#muhaffez", "❌ Transcription failed: ${e.message}", e)
      e.printStackTrace()
      return ""
    }
  }

  /**
   * Transcribe audio file by segmenting it and transcribing each segment
   * @param audioFilePath Path to audio file
   * @return Full transcription
   */
  fun transcribeFile(audioFilePath: String): String {
    Log.d("#muhaffez", "📂 Loading audio file: $audioFilePath")

    // Load audio file
    val audio = loadAudioFile(audioFilePath)
    Log.d("#muhaffez", "✅ Loaded ${audio.size} samples")

    // Segment audio into chunks
    val segments = segmentAudio(audio)
    Log.d("#muhaffez", "✅ Segmented into ${segments.size} segments")

    // Transcribe each segment and concatenate
    val transcriptions = segments.mapIndexed { index, segment ->
      Log.d("#muhaffez", "🎤 Transcribing segment ${index + 1}/${segments.size}")
      transcribe(segment)
    }

    val fullTranscription = transcriptions.joinToString(" ")
    Log.d("#muhaffez", "✅ Full transcription: $fullTranscription")

    return fullTranscription
  }

  /**
   * Segment audio using Silero VAD (like faster-whisper)
   */
  private fun segmentAudio(audio: FloatArray): List<FloatArray> {
    // Use Silero VAD to detect speech segments
    val vadOptions = SileroVAD.VadOptions(
      threshold = 0.5f,
      minSpeechDurationMs = 1000,  // 1 second minimum for speech segments
      maxSpeechDurationS = Float.POSITIVE_INFINITY,
      minSilenceDurationMs = 20,  // 20ms = 2 Whisper frames (10ms hop, 100 fps)
      speechPadMs = 400  // Add 400ms padding around speech
    )

    val speechSegments = sileroVAD?.getSpeechTimestamps(audio, SAMPLE_RATE, vadOptions)
      ?: run {
        Log.w("#muhaffez", "⚠️  Silero VAD not available, using fallback segmentation")
        return segmentAudioFallback(audio)
      }

    Log.d("#muhaffez", "✂️ Silero VAD found ${speechSegments.size} speech segments")

    // Extract and pad each segment
    val segments = mutableListOf<FloatArray>()
    for ((i, segment) in speechSegments.withIndex()) {
      val segmentAudio = audio.copyOfRange(segment.start, min(segment.end, audio.size))
      val duration = segmentAudio.size.toFloat() / SAMPLE_RATE

      Log.d("#muhaffez", "  Segment ${i + 1}: ${duration}s (${segment.start} - ${segment.end})")

      // Note: minSpeechDurationMs already filtered this, but keep the check for safety
      if (segmentAudio.size >= SAMPLE_RATE / 2) {  // Skip segments < 0.5s
        // Pad to 30 seconds if necessary
        val padded = FloatArray(N_SAMPLES)
        segmentAudio.copyInto(padded, 0, 0, min(segmentAudio.size, N_SAMPLES))
        segments.add(padded)
      }
    }

    return segments
  }

  /**
   * Fallback segmentation using energy-based silence detection
   * (Used if Silero VAD is not available)
   */
  private fun segmentAudioFallback(audio: FloatArray): List<FloatArray> {
    val thresholdDb = -30f
    val minSilenceFrames = 11
    val hopLength = SAMPLE_RATE / 20  // 20 frames per second

    // Detect energy levels
    val frameEnergy = mutableListOf<Float>()
    var i = 0
    while (i < audio.size) {
      val end = min(i + hopLength, audio.size)
      val frame = audio.copyOfRange(i, end)

      // Calculate RMS energy in dB
      val energy = frame.map { it * it }.average().toFloat()
      val energyDb = 10 * log10(max(energy, 1e-10f))
      frameEnergy.add(energyDb)

      i += hopLength
    }

    // Find silent frames
    val silentFrames = frameEnergy.indices.filter { frameEnergy[it] < thresholdDb }

    // Group consecutive silent frames into regions
    val silentRegions = mutableListOf<Pair<Int, Int>>()
    if (silentFrames.isNotEmpty()) {
      var start = silentFrames[0]
      var prev = silentFrames[0]

      for (frame in silentFrames.drop(1)) {
        if (frame != prev + 1) {
          // End of silent region
          if (prev - start + 1 >= minSilenceFrames) {
            silentRegions.add(Pair(start, prev))
          }
          start = frame
        }
        prev = frame
      }

      // Add last region
      if (prev - start + 1 >= minSilenceFrames) {
        silentRegions.add(Pair(start, prev))
      }
    }

    // Convert frame indices to sample indices
    val silentSamples = silentRegions.map { (start, end) ->
      Pair(start * hopLength, end * hopLength)
    }

    // Create segments between silent regions
    val segments = mutableListOf<FloatArray>()
    var currentStart = 0

    for ((silenceStart, silenceEnd) in silentSamples) {
      if (silenceStart > currentStart) {
        val segment = audio.copyOfRange(currentStart, min(silenceStart, audio.size))
        if (segment.size >= SAMPLE_RATE / 2) {  // Skip segments < 0.5s
          // Pad to 30 seconds if necessary
          val padded = FloatArray(N_SAMPLES)
          segment.copyInto(padded, 0, 0, min(segment.size, N_SAMPLES))
          segments.add(padded)
        }
      }
      currentStart = min(silenceEnd, audio.size)
    }

    // Add final segment if any audio remains
    if (currentStart < audio.size) {
      val segment = audio.copyOfRange(currentStart, audio.size)
      if (segment.size >= SAMPLE_RATE / 2) {  // Skip segments < 0.5s
        val padded = FloatArray(N_SAMPLES)
        segment.copyInto(padded, 0, 0, min(segment.size, N_SAMPLES))
        segments.add(padded)
      }
    }

    Log.d("#muhaffez", "✂️ Fallback: Created ${segments.size} segments")
    return segments
  }

  /**
   * Load audio file and resample to 16kHz if necessary
   */
  private fun loadAudioFile(filePath: String): FloatArray {
    val file = File(filePath)
    val bytes = file.readBytes()

    // Parse WAV header to get sample rate and channels
    val sampleRate = parseSampleRate(bytes)
    val numChannels = parseNumChannels(bytes)

    Log.d("#muhaffez", "📊 Audio info: ${sampleRate}Hz, $numChannels channel(s)")

    // Skip WAV header (44 bytes) and convert 16-bit PCM to float
    val audioSamples = (bytes.size - 44) / 2 / numChannels
    val audio = FloatArray(audioSamples)

    // Read audio and convert stereo to mono if necessary
    for (i in 0 until audioSamples) {
      val byteIndex = 44 + i * 2 * numChannels

      if (numChannels == 1) {
        // Mono
        val sample = ((bytes[byteIndex + 1].toInt() shl 8) or (bytes[byteIndex].toInt() and 0xFF)).toShort()
        audio[i] = sample / 32768.0f
      } else {
        // Stereo - average left and right channels
        var sum = 0f
        for (ch in 0 until numChannels) {
          val offset = byteIndex + ch * 2
          val sample = ((bytes[offset + 1].toInt() shl 8) or (bytes[offset].toInt() and 0xFF)).toShort()
          sum += sample / 32768.0f
        }
        audio[i] = sum / numChannels
      }
    }

    // Resample to 16kHz if necessary
    return if (sampleRate != SAMPLE_RATE) {
      Log.d("#muhaffez", "🔄 Resampling from ${sampleRate}Hz to ${SAMPLE_RATE}Hz...")
      resampleAudio(audio, sampleRate, SAMPLE_RATE)
    } else {
      audio
    }
  }

  /**
   * Parse sample rate from WAV header
   */
  private fun parseSampleRate(bytes: ByteArray): Int {
    // Sample rate is at bytes 24-27 (little endian)
    return (bytes[27].toInt() and 0xFF shl 24) or
           (bytes[26].toInt() and 0xFF shl 16) or
           (bytes[25].toInt() and 0xFF shl 8) or
           (bytes[24].toInt() and 0xFF)
  }

  /**
   * Parse number of channels from WAV header
   */
  private fun parseNumChannels(bytes: ByteArray): Int {
    // Number of channels is at bytes 22-23 (little endian)
    return (bytes[23].toInt() and 0xFF shl 8) or (bytes[22].toInt() and 0xFF)
  }

  /**
   * Resample audio using linear interpolation
   */
  private fun resampleAudio(audio: FloatArray, fromRate: Int, toRate: Int): FloatArray {
    val ratio = fromRate.toFloat() / toRate
    val newLength = (audio.size / ratio).toInt()
    val resampled = FloatArray(newLength)

    for (i in 0 until newLength) {
      val srcPos = i * ratio
      val srcIdx = srcPos.toInt()

      if (srcIdx >= audio.size - 1) {
        resampled[i] = audio[audio.size - 1]
      } else {
        // Linear interpolation
        val frac = srcPos - srcIdx
        resampled[i] = audio[srcIdx] * (1 - frac) + audio[srcIdx + 1] * frac
      }
    }

    Log.d("#muhaffez", "✅ Resampled: ${audio.size} -> ${resampled.size} samples")
    return resampled
  }

  /**
   * Extract mel spectrogram features using Whisper global normalization
   */
  private fun extractMelFeatures(audio: FloatArray): Array<FloatArray> {
    try {
      // Pad or truncate to N_SAMPLES (30 seconds at 16kHz)
      val paddedAudio = FloatArray(N_SAMPLES)
      val copyLength = min(audio.size, N_SAMPLES)
      audio.copyInto(paddedAudio, 0, 0, copyLength)

      // Step 1: Compute STFT magnitudes
      val targetFrames = 3000  // Whisper expects exactly 3000 frames
      val stft = computeSTFT(paddedAudio, N_FFT, HOP_LENGTH, targetFrames)

      // Step 2: Apply mel filterbank
      val melSpec = applyMelFilterbank(stft)

      // Step 3: Apply log10 transform
      val logMelSpec = Array(melSpec.size) { i ->
        FloatArray(melSpec[i].size) { j ->
          log10(max(melSpec[i][j], 1e-10f))
        }
      }

      // Step 4: Apply Whisper global normalization
      // IMPORTANT: These values must match Python training normalization
      val WHISPER_MEL_MEAN = -4.2677f
      val WHISPER_MEL_STD = 4.5689f

      val normalizedMel = Array(logMelSpec.size) { i ->
        FloatArray(logMelSpec[i].size) { j ->
          (logMelSpec[i][j] - WHISPER_MEL_MEAN) / WHISPER_MEL_STD
        }
      }

      return normalizedMel
    } catch (e: Exception) {
      Log.e("#muhaffez", "❌ Mel extraction failed: ${e.message}", e)
      throw e
    }
  }

  /**
   * Compute Short-Time Fourier Transform using JTransforms FFT
   */
  private fun computeSTFT(audio: FloatArray, nFFT: Int, hopLength: Int, targetFrames: Int): Array<FloatArray> {
    val fftBins = nFFT / 2 + 1

    // Hann window
    val windowTemp = FloatArray(nFFT + 1) { i ->
      (0.5 * (1.0 - cos(2.0 * PI * i / nFFT))).toFloat()
    }
    val window = windowTemp.copyOfRange(0, nFFT)

    // Apply center padding
    val padSize = nFFT / 2
    val paddedAudio = FloatArray(audio.size + 2 * padSize)
    audio.copyInto(paddedAudio, padSize, 0, audio.size)

    // Calculate number of frames
    val numFramesBeforeDrop = (paddedAudio.size - nFFT) / hopLength + 1
    val magnitudes = Array(fftBins) { FloatArray(numFramesBeforeDrop) }

    val fft = FloatFFT_1D(nFFT.toLong())

    for (frame in 0 until numFramesBeforeDrop) {
      val start = frame * hopLength
      val frameData = FloatArray(nFFT * 2) // Complex: [real0, imag0, real1, imag1, ...]

      for (i in 0 until nFFT) {
        frameData[i * 2] = paddedAudio[start + i] * window[i]
        frameData[i * 2 + 1] = 0f
      }

      fft.complexForward(frameData)

      // Compute magnitude
      for (i in 0 until fftBins) {
        val real = frameData[i * 2]
        val imag = frameData[i * 2 + 1]
        magnitudes[i][frame] = sqrt(real * real + imag * imag)
      }
    }

    // Drop last frame and take exactly targetFrames
    val result = Array(fftBins) { FloatArray(targetFrames) }
    for (i in 0 until fftBins) {
      for (j in 0 until targetFrames) {
        result[i][j] = magnitudes[i][j]
      }
    }

    return result
  }

  /**
   * Apply mel filterbank to STFT magnitudes
   */
  private fun applyMelFilterbank(stft: Array<FloatArray>): Array<FloatArray> {
    val numFrames = stft[0].size
    val melSpec = Array(N_MELS) { FloatArray(numFrames) }

    for (frame in 0 until numFrames) {
      for (mel in 0 until N_MELS) {
        var sum = 0f
        for (bin in MEL_FILTERS[mel].indices) {
          sum += stft[bin][frame] * MEL_FILTERS[mel][bin]
        }
        melSpec[mel][frame] = sum
      }
    }

    return melSpec
  }

  /**
   * Create mel filterbank (Whisper standard)
   */
  private fun createMelFilterbank(): Array<FloatArray> {
    val fftBins = N_FFT / 2 + 1
    val fmin = 0f
    val fmax = SAMPLE_RATE / 2f

    // Mel scale conversion
    fun hzToMel(hz: Float) = 2595f * log10(1f + hz / 700f)
    fun melToHz(mel: Float) = 700f * (10f.pow(mel / 2595f) - 1f)

    val melMin = hzToMel(fmin)
    val melMax = hzToMel(fmax)

    val melPoints = FloatArray(N_MELS + 2) { i ->
      melToHz(melMin + (melMax - melMin) * i / (N_MELS + 1))
    }

    val binPoints = melPoints.map { hz ->
      (fftBins * hz / (SAMPLE_RATE / 2f)).toInt()
    }

    val filterbank = Array(N_MELS) { FloatArray(fftBins) }

    for (i in 0 until N_MELS) {
      val left = binPoints[i]
      val center = binPoints[i + 1]
      val right = binPoints[i + 2]

      for (j in left until center) {
        filterbank[i][j] = (j - left).toFloat() / (center - left)
      }
      for (j in center until right) {
        filterbank[i][j] = (right - j).toFloat() / (right - center)
      }
    }

    // Normalize
    for (i in 0 until N_MELS) {
      val sum = filterbank[i].sum()
      if (sum > 0) {
        for (j in 0 until fftBins) {
          filterbank[i][j] /= sum
        }
      }
    }

    return filterbank
  }

  fun cleanup() {
    encoderSession?.close()
    decoderSession?.close()
    sileroVAD?.cleanup()
    env?.close()
  }
}
