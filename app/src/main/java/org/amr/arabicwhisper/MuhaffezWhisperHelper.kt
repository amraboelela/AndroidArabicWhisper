package org.amr.arabicwhisper

import android.content.Context
import android.util.Log
import ai.onnxruntime.*
import kotlinx.serialization.json.*
import java.io.File
import java.io.FileOutputStream
import kotlin.math.*

/**
 * Muhaffez Whisper Helper - Word-level Arabic Quran transcription
 * Uses custom trained model with word-based vocabulary (not BPE)
 */
class MuhaffezWhisperHelper(private val context: Context) {

    private var encoderSession: OrtSession? = null
    private var decoderSession: OrtSession? = null
    private var env: OrtEnvironment? = null

    private lateinit var vocabulary: List<String>
    private lateinit var wordToIdx: Map<String, Int>

    private val SAMPLE_RATE = 16000
    private val N_FFT = 400
    private val HOP_LENGTH = 160
    private val N_MELS = 80

    private val SOS_TOKEN = 1L
    private val EOS_TOKEN = 2L

    init {
        Log.d("#muhaffez", "Initializing Muhaffez Whisper Helper...")
        initializeOnnx()
        loadVocabulary()
        Log.d("#muhaffez", "✅ Muhaffez Whisper Helper initialized")
    }

    private fun initializeOnnx() {
        try {
            env = OrtEnvironment.getEnvironment()
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
            val indices = mapOf(
                "<unk>" to vocabulary.indexOf("<unk>"),
                "<s>" to vocabulary.indexOf("<s>"),
                "</s>" to vocabulary.indexOf("</s>")
            )
            Log.d("#muhaffez", "Vocab special tokens indices: $indices")
        } catch (e: Exception) {
            Log.e("#muhaffez", "❌ Failed to load vocabulary: ${e.message}", e)
            throw e
        }
    }

    fun transcribe(audio: FloatArray): String {
        try {
            Log.d("#muhaffez", "🎙️ Transcribing ${audio.size} samples (${audio.size / SAMPLE_RATE.toFloat()} sec)...")

            val melFeatures = AudioPreprocessor.extractMelFeatures(audio)

            val numFrames = melFeatures[0].size

            // Log first 10 values of first mel bin for debugging
            val mel0Preview = melFeatures[0].take(10).joinToString(", ") { "%.3f".format(it) }
            Log.d("#muhaffez", "Mel[0] first 10 frames: $mel0Preview")

            // Log stats
            val allValues = melFeatures.flatMap { it.toList() }
            val minVal = allValues.minOrNull() ?: 0f
            val maxVal = allValues.maxOrNull() ?: 0f
            val meanVal = allValues.average().toFloat()
            val stdVal = sqrt(allValues.map { (it - meanVal) * (it - meanVal) }.average().toFloat())
            Log.d("#muhaffez", "Mel stats: min=%.4f, max=%.4f, mean=%.4f, std=%.4f".format(minVal, maxVal, meanVal, stdVal))
            Log.d("#muhaffez", "Mel shape: ${melFeatures.size} x $numFrames")

            val transposedMel = Array(1) { Array(N_MELS) { FloatArray(numFrames) } }
            for (mel in 0 until N_MELS) {
                val row = melFeatures[mel]
                for (frame in row.indices) transposedMel[0][mel][frame] = row[frame]
            }

            val encoderInput = OnnxTensor.createTensor(env, transposedMel)
            val encoderOutputs = encoderSession!!.run(mapOf("input_features" to encoderInput))
            val encoderHiddenStates = encoderOutputs[0].value as Array<Array<FloatArray>>

            val generatedTokens = mutableListOf<Long>()
            val maxTokens = 50

            for (step in 0 until maxTokens) {
                val inputIds = Array(1) { LongArray(1 + generatedTokens.size) }
                inputIds[0][0] = SOS_TOKEN
                for (i in generatedTokens.indices) inputIds[0][i + 1] = generatedTokens[i]

                val inputIdsTensor = OnnxTensor.createTensor(env, inputIds)
                val encoderHiddenStatesTensor = OnnxTensor.createTensor(env, encoderHiddenStates)
                val decoderOutputs = decoderSession!!.run(mapOf(
                    "input_ids" to inputIdsTensor,
                    "encoder_hidden_states" to encoderHiddenStatesTensor
                ))

                val logits = decoderOutputs[0].value as Array<Array<FloatArray>>
                val lastLogits = logits[0].last()
                val nextToken = lastLogits.indices.maxByOrNull { lastLogits[it] }?.toLong() ?: EOS_TOKEN

                inputIdsTensor.close()
                encoderHiddenStatesTensor.close()
                decoderOutputs.close()

                if (nextToken == EOS_TOKEN) break
                generatedTokens.add(nextToken)
            }

            encoderInput.close()

            return generatedTokens.map { idx -> if (idx.toInt() < vocabulary.size) vocabulary[idx.toInt()] else "<unk>" }
                .joinToString(" ")
        } catch (e: Exception) {
            Log.e("#muhaffez", "❌ Transcription failed: ${e.message}", e)
            return ""
        }
    }

    fun transcribeFile(audioFilePath: String): String {
        val audio = AudioPreprocessor.loadAudioFile(audioFilePath)
        val audioDuration = audio.size / SAMPLE_RATE.toFloat()
        Log.d("#muhaffez", "✅ Loaded ${audio.size} samples (%.2fs)".format(audioDuration))

        val segments = segmentAudio(audio)
        Log.d("#muhaffez", "✅ Segmented into ${segments.size} segments")

        segments.forEachIndexed { index, segment ->
            val duration = segment.size / SAMPLE_RATE.toFloat()
            Log.d("#muhaffez", "  Segment ${index + 1}: %.2fs (${segment.size} samples)".format(duration))
        }

        // Transcribe all segments
        val allTranscriptions = mutableListOf<String>()
        segments.forEachIndexed { index, segment ->
            Log.d("#muhaffez", "🎤 Transcribing segment ${index + 1}/${segments.size}")
            val transcription = transcribe(segment)
            Log.d("#muhaffez", "📝 Segment ${index + 1} transcription: $transcription")
            allTranscriptions.add(transcription)
        }

        return allTranscriptions.joinToString(" ")
    }

    private fun segmentAudio(audio: FloatArray): List<FloatArray> = segmentAudioEnergyBased(audio)

    /**
     * Segment audio using energy-based silence detection
     * Matches Python implementation in segment_audio_001.py
     */
    private fun segmentAudioEnergyBased(audio: FloatArray): List<FloatArray> {
        val thresholdDb = -30f
        val minSilenceFrames = 1  // 1 frame = 50ms minimum silence
        val hopLength = SAMPLE_RATE / 20  // 20 frames per second (50ms per frame)

        // Calculate energy for each frame
        val frameEnergy = mutableListOf<Float>()
        var i = 0
        while (i < audio.size) {
            val end = min(i + hopLength, audio.size)
            val frame = audio.copyOfRange(i, end)
            val energy = frame.map { it * it }.average().toFloat()
            val energyDb = 10 * log10(max(energy, 1e-10f))
            frameEnergy.add(energyDb)
            i += hopLength
        }

        Log.d("#muhaffez", "Total frames: ${frameEnergy.size}")

        // Find silent frames
        val silentFrames = frameEnergy.indices.filter { frameEnergy[it] < thresholdDb }
        Log.d("#muhaffez", "Silent frames: ${silentFrames.size}")
        Log.d("#muhaffez", "First 20 silent frame indices: ${silentFrames.take(20)}")

        // Group consecutive silent frames into regions
        val silentRegions = mutableListOf<Pair<Int, Int>>()
        if (silentFrames.isNotEmpty()) {
            var start = silentFrames[0]
            var prev = silentFrames[0]
            for (frame in silentFrames.drop(1)) {
                if (frame != prev + 1) {
                    if (prev - start + 1 >= minSilenceFrames) silentRegions.add(Pair(start, prev))
                    start = frame
                }
                prev = frame
            }
            if (prev - start + 1 >= minSilenceFrames) silentRegions.add(Pair(start, prev))
        }

        Log.d("#muhaffez", "Silent regions (frames): ${silentRegions.size}")
        silentRegions.forEach { (s, e) ->
            val silenceStartSample = s * hopLength
            val silenceEndSample = e * hopLength
            Log.d("#muhaffez", "  Silent region: frames $s-$e (${e - s + 1} frames), samples $silenceStartSample-$silenceEndSample")
        }

        // Convert frame indices to sample indices
        val silentSamples = silentRegions.map { (s, e) -> Pair(s * hopLength, e * hopLength) }

        // Create segments between silent regions
        val allSegments = mutableListOf<Pair<Int, Int>>()
        var currentStart = 0
        for ((silenceStart, silenceEnd) in silentSamples) {
            if (silenceStart > currentStart) {
                allSegments.add(Pair(currentStart, min(silenceStart, audio.size)))
            }
            currentStart = min(silenceEnd, audio.size)
        }
        if (currentStart < audio.size) {
            allSegments.add(Pair(currentStart, audio.size))
        }

        Log.d("#muhaffez", "Total segments created: ${allSegments.size}")
        allSegments.forEachIndexed { idx, (start, end) ->
            val duration = (end - start) / SAMPLE_RATE.toFloat()
            Log.d("#muhaffez", "  Segment ${idx + 1}: %.2fs (${end - start} samples)".format(duration))
        }

        // Filter segments >= 0.5s (matches Python)
        val segments = allSegments.filter { (start, end) -> (end - start) >= SAMPLE_RATE / 2 }
            .map { (start, end) -> audio.copyOfRange(start, end) }

        Log.d("#muhaffez", "Segments after filtering (>= 0.5s): ${segments.size}")

        return segments
    }

    fun cleanup() {
        encoderSession?.close()
        decoderSession?.close()
        env?.close()
    }
}
