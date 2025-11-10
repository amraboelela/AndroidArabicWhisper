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
            val tokenProbabilities = mutableListOf<Float>()
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

                // Calculate softmax probabilities
                val maxLogit = lastLogits.maxOrNull() ?: 0f
                val expLogits = lastLogits.map { exp(it - maxLogit) }
                val sumExp = expLogits.sum()
                val probabilities = expLogits.map { it / sumExp }

                val nextToken = lastLogits.indices.maxByOrNull { lastLogits[it] }?.toLong() ?: EOS_TOKEN
                val nextTokenProb = if (nextToken.toInt() < probabilities.size) probabilities[nextToken.toInt()] else 0f

                inputIdsTensor.close()
                encoderHiddenStatesTensor.close()
                decoderOutputs.close()

                if (nextToken == EOS_TOKEN) break
                generatedTokens.add(nextToken)
                tokenProbabilities.add(nextTokenProb)
            }

            encoderInput.close()

            // Calculate average confidence
            val avgConfidence = if (tokenProbabilities.isNotEmpty()) {
                tokenProbabilities.average().toFloat() * 100
            } else {
                0f
            }

            Log.d("#muhaffez", "Confidence: %.1f%%".format(avgConfidence))

            // Only return transcription if confidence is above 20%
            if (avgConfidence < 20f) {
                Log.d("#muhaffez", "⚠️ Low confidence (%.1f%%), skipping transcription".format(avgConfidence))
                return ""
            }

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

        val segments = AudioPreprocessor.segmentAudioEnergyBased(audio, thresholdDb = -30f, minSilenceFrames = 11)
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

    fun cleanup() {
        encoderSession?.close()
        decoderSession?.close()
        env?.close()
    }
}
