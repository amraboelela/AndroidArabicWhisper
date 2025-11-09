package org.amr.arabicwhisper

import android.content.Context
import android.util.Log
import ai.onnxruntime.*
import kotlinx.serialization.json.*
import org.jtransforms.fft.FloatFFT_1D
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
    private val MEL_FILTERS by lazy { createMelFilterbank() }

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

            val melFeatures = extractMelFeatures(audio)

            val numFrames = melFeatures[0].size
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
        val audio = loadAudioFile(audioFilePath)
        val segments = segmentAudio(audio)
        return segments.joinToString(" ") { transcribe(it) }
    }

    private fun segmentAudio(audio: FloatArray): List<FloatArray> = segmentAudioEnergyBased(audio)

    private fun segmentAudioEnergyBased(audio: FloatArray): List<FloatArray> {
        val thresholdDb = -30f
        val minSilenceFrames = 2
        val hopLength = SAMPLE_RATE / 20

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

        val silentFrames = frameEnergy.indices.filter { frameEnergy[it] < thresholdDb }
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

        val silentSamples = silentRegions.map { (s, e) -> Pair(s * hopLength, e * hopLength) }
        val segments = mutableListOf<FloatArray>()
        var currentStart = 0
        for ((silenceStart, silenceEnd) in silentSamples) {
            if (silenceStart > currentStart) {
                val segment = audio.copyOfRange(currentStart, min(silenceStart, audio.size))
                if (segment.size >= SAMPLE_RATE / 4) segments.add(segment)
            }
            currentStart = min(silenceEnd, audio.size)
        }
        if (currentStart < audio.size) {
            val segment = audio.copyOfRange(currentStart, audio.size)
            if (segment.size >= SAMPLE_RATE / 4) segments.add(segment)
        }
        return segments
    }

    private fun loadAudioFile(filePath: String): FloatArray {
        val bytes = File(filePath).readBytes()
        val sampleRate = parseSampleRate(bytes)
        val numChannels = parseNumChannels(bytes)
        val audioSamples = ((bytes.size - 44) / 2) / max(1, numChannels)
        val audio = FloatArray(audioSamples)
        for (i in 0 until audioSamples) {
            val byteIndex = 44 + i * 2 * numChannels
            if (numChannels == 1) {
                val low = bytes[byteIndex].toInt() and 0xFF
                val high = bytes[byteIndex + 1].toInt()
                val sample = (high shl 8) or low
                audio[i] = if (sample >= 0x8000) (sample - 0x10000) / 32768f else sample / 32768f
            } else {
                var sum = 0f
                for (ch in 0 until numChannels) {
                    val offset = byteIndex + ch * 2
                    val low = bytes[offset].toInt() and 0xFF
                    val high = bytes[offset + 1].toInt()
                    val sample = (high shl 8) or low
                    sum += if (sample >= 0x8000) (sample - 0x10000) / 32768f else sample / 32768f
                }
                audio[i] = sum / numChannels
            }
        }
        return if (sampleRate != SAMPLE_RATE) resampleAudio(audio, sampleRate, SAMPLE_RATE) else audio
    }

    private fun parseSampleRate(bytes: ByteArray): Int =
        ((bytes[27].toInt() and 0xFF) shl 24) or
                ((bytes[26].toInt() and 0xFF) shl 16) or
                ((bytes[25].toInt() and 0xFF) shl 8) or
                (bytes[24].toInt() and 0xFF)

    private fun parseNumChannels(bytes: ByteArray): Int =
        ((bytes[23].toInt() and 0xFF) shl 8) or
                (bytes[22].toInt() and 0xFF)

    private fun resampleAudio(audio: FloatArray, fromRate: Int, toRate: Int): FloatArray {
        val ratio = fromRate.toFloat() / toRate
        val newLength = max(1, (audio.size / ratio).toInt())
        val resampled = FloatArray(newLength)
        for (i in 0 until newLength) {
            val srcPos = i * ratio
            val srcIdx = srcPos.toInt()
            resampled[i] = if (srcIdx >= audio.size - 1) audio[audio.size - 1] else
                audio[srcIdx] * (1 - (srcPos - srcIdx)) + audio[srcIdx + 1] * (srcPos - srcIdx)
        }
        return resampled
    }

    private fun extractMelFeatures(audio: FloatArray): Array<FloatArray> {
        val stft = computeSTFT(audio, N_FFT, HOP_LENGTH)
        val melSpec = applyMelFilterbank(stft)
        val logMelSpec = Array(melSpec.size) { i -> FloatArray(melSpec[i].size) { j -> ln(max(melSpec[i][j], 1e-9f)) } }
        val WHISPER_MEL_MEAN = -4.2677393f
        val WHISPER_MEL_STD = 4.5689974f
        return Array(logMelSpec.size) { i -> FloatArray(logMelSpec[i].size) { j -> (logMelSpec[i][j] - WHISPER_MEL_MEAN) / WHISPER_MEL_STD } }
    }

    private fun computeSTFT(audio: FloatArray, nFFT: Int, hopLength: Int): Array<FloatArray> {
        val fftBins = nFFT / 2 + 1
        // PyTorch periodic Hann window: 0.5 - 0.5 * cos(2*pi*i/n)
        val window = FloatArray(nFFT) { i -> (0.5 - 0.5 * cos(2.0 * PI * i / nFFT)).toFloat() }

        // Apply reflect padding (PyTorch default for center=True)
        val padSize = nFFT / 2
        val paddedAudio = FloatArray(audio.size + 2 * padSize)
        for (i in 0 until padSize) {
            paddedAudio[i] = audio[padSize - i]  // Reflect left
            paddedAudio[padSize + audio.size + i] = audio[audio.size - 2 - i]  // Reflect right
        }
        System.arraycopy(audio, 0, paddedAudio, padSize, audio.size)

        val numFrames = max(1, (paddedAudio.size - nFFT) / hopLength + 1)
        val magnitudes = Array(fftBins) { FloatArray(numFrames) }
        val fft = FloatFFT_1D(nFFT.toLong())

        for (frame in 0 until numFrames) {
            val start = frame * hopLength
            val frameData = FloatArray(nFFT)
            for (i in 0 until nFFT) frameData[i] = paddedAudio.getOrElse(start + i) { 0f } * window[i]

            fft.realForward(frameData)

            // Extract power spectrum (magnitude squared) from packed format
            magnitudes[0][frame] = frameData[0] * frameData[0]  // DC
            magnitudes[fftBins - 1][frame] = frameData[1] * frameData[1]  // Nyquist
            for (k in 1 until fftBins - 1) {
                val real = frameData[2 * k]
                val imag = frameData[2 * k + 1]
                magnitudes[k][frame] = real * real + imag * imag  // Power
            }
        }
        return magnitudes
    }

    private fun applyMelFilterbank(stft: Array<FloatArray>): Array<FloatArray> {
        val numFrames = stft[0].size
        val melSpec = Array(N_MELS) { FloatArray(numFrames) }
        for (frame in 0 until numFrames) {
            for (mel in 0 until N_MELS) {
                var sum = 0f
                for (bin in stft.indices) {
                    sum += stft[bin][frame] * MEL_FILTERS[mel][bin]
                }
                melSpec[mel][frame] = sum
            }
        }
        return melSpec
    }

    private fun createMelFilterbank(): Array<FloatArray> {
        val fftBins = N_FFT / 2 + 1
        val fmin = 0f
        val fmax = SAMPLE_RATE / 2f
        fun hzToMel(hz: Float) = 2595f * log10(1f + hz / 700f)
        fun melToHz(mel: Float) = 700f * (10f.pow(mel / 2595f) - 1f)
        val melMin = hzToMel(fmin)
        val melMax = hzToMel(fmax)
        val melPointsHz = FloatArray(N_MELS + 2) { i -> melToHz(melMin + (melMax - melMin) * i / (N_MELS + 1)) }
        val binPointsFloat = melPointsHz.map { hz -> (fftBins - 1) * hz / fmax }
        val filterbank = Array(N_MELS) { FloatArray(fftBins) }
        for (m in 0 until N_MELS) {
            val leftBin = binPointsFloat[m]
            val centerBin = binPointsFloat[m + 1]
            val rightBin = binPointsFloat[m + 2]
            for (fftBin in 0 until fftBins) {
                val freq = fftBin.toFloat()
                filterbank[m][fftBin] = when {
                    freq >= leftBin && freq <= centerBin && centerBin > leftBin -> (freq - leftBin) / (centerBin - leftBin)
                    freq >= centerBin && freq <= rightBin && rightBin > centerBin -> (rightBin - freq) / (rightBin - centerBin)
                    else -> 0f
                }
            }
        }
        return filterbank
    }

    fun cleanup() {
        encoderSession?.close()
        decoderSession?.close()
        env?.close()
    }
}
