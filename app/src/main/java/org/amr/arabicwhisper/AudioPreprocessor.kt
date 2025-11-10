package org.amr.arabicwhisper

import android.util.Log
import org.jtransforms.fft.FloatFFT_1D
import java.io.File
import kotlin.math.*

/**
 * Audio preprocessing utilities for Whisper model
 * Handles audio loading, resampling, and mel spectrogram extraction
 */
class AudioPreprocessor {
    
    companion object {
        private const val SAMPLE_RATE = 16000
        private const val N_FFT = 400
        private const val HOP_LENGTH = 160
        private const val N_MELS = 80
        
        private val MEL_FILTERS by lazy { createMelFilterbank() }
        
        /**
         * Load audio file from path
         * Properly parses WAV format including metadata chunks
         * Supports both 16-bit PCM and 32-bit IEEE Float formats
         */
        fun loadAudioFile(filePath: String): FloatArray {
            val bytes = File(filePath).readBytes()
            val sampleRate = parseSampleRate(bytes)
            val numChannels = parseNumChannels(bytes)
            val audioFormat = parseAudioFormat(bytes)

            // Find the "data" chunk to get actual audio data size
            var dataChunkOffset = 44  // Default assumption
            var dataChunkSize = bytes.size - 44

            var offset = 12  // After "RIFF" header
            while (offset < bytes.size - 8) {
                val chunkId = String(bytes.copyOfRange(offset, offset + 4), Charsets.US_ASCII)
                val chunkSize = ((bytes[offset + 7].toInt() and 0xFF) shl 24) or
                               ((bytes[offset + 6].toInt() and 0xFF) shl 16) or
                               ((bytes[offset + 5].toInt() and 0xFF) shl 8) or
                               (bytes[offset + 4].toInt() and 0xFF)

                if (chunkId == "data") {
                    dataChunkOffset = offset + 8
                    dataChunkSize = chunkSize
                    break
                }

                offset += 8 + chunkSize
            }

            val audio = when (audioFormat) {
                1 -> loadPCM16(bytes, dataChunkOffset, dataChunkSize, numChannels)
                3 -> loadIEEEFloat32(bytes, dataChunkOffset, dataChunkSize, numChannels)
                else -> {
                    Log.e("AudioPreprocessor", "Unsupported audio format: $audioFormat (only PCM=1 and IEEE Float=3 are supported)")
                    FloatArray(0)
                }
            }

            return if (sampleRate != SAMPLE_RATE) resampleAudio(audio, sampleRate, SAMPLE_RATE) else audio
        }

        /**
         * Parse audio format from WAV header
         * 1 = PCM, 3 = IEEE Float
         */
        private fun parseAudioFormat(bytes: ByteArray): Int =
            ((bytes[21].toInt() and 0xFF) shl 8) or
                    (bytes[20].toInt() and 0xFF)

        /**
         * Load 16-bit PCM audio data
         */
        private fun loadPCM16(bytes: ByteArray, dataChunkOffset: Int, dataChunkSize: Int, numChannels: Int): FloatArray {
            val audioSamples = (dataChunkSize / 2) / max(1, numChannels)
            val audio = FloatArray(audioSamples)

            for (i in 0 until audioSamples) {
                val byteIndex = dataChunkOffset + i * 2 * numChannels
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

            Log.d("AudioPreprocessor", "Loaded 16-bit PCM: $audioSamples samples, $numChannels channel(s)")
            return audio
        }

        /**
         * Load 32-bit IEEE Float audio data
         */
        private fun loadIEEEFloat32(bytes: ByteArray, dataChunkOffset: Int, dataChunkSize: Int, numChannels: Int): FloatArray {
            val audioSamples = (dataChunkSize / 4) / max(1, numChannels)
            val audio = FloatArray(audioSamples)

            for (i in 0 until audioSamples) {
                val byteIndex = dataChunkOffset + i * 4 * numChannels
                if (numChannels == 1) {
                    // Read 32-bit float (little-endian)
                    val bits = ((bytes[byteIndex + 3].toInt() and 0xFF) shl 24) or
                              ((bytes[byteIndex + 2].toInt() and 0xFF) shl 16) or
                              ((bytes[byteIndex + 1].toInt() and 0xFF) shl 8) or
                              (bytes[byteIndex].toInt() and 0xFF)
                    audio[i] = Float.fromBits(bits)
                } else {
                    var sum = 0f
                    for (ch in 0 until numChannels) {
                        val offset = byteIndex + ch * 4
                        val bits = ((bytes[offset + 3].toInt() and 0xFF) shl 24) or
                                  ((bytes[offset + 2].toInt() and 0xFF) shl 16) or
                                  ((bytes[offset + 1].toInt() and 0xFF) shl 8) or
                                  (bytes[offset].toInt() and 0xFF)
                        sum += Float.fromBits(bits)
                    }
                    audio[i] = sum / numChannels
                }
            }

            Log.d("AudioPreprocessor", "Loaded 32-bit IEEE Float: $audioSamples samples, $numChannels channel(s)")
            return audio
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

        /**
         * Extract mel spectrogram features from audio
         * Returns array of shape [N_MELS][numFrames]
         */
        fun extractMelFeatures(audio: FloatArray): Array<FloatArray> {
            val stft = computeSTFT(audio, N_FFT, HOP_LENGTH)
            val melSpec = applyMelFilterbank(stft)
            val logMelSpec = Array(melSpec.size) { i -> FloatArray(melSpec[i].size) { j -> ln(max(melSpec[i][j], 1e-9f)) } }
            val WHISPER_MEL_MEAN = -4.2677393f
            val WHISPER_MEL_STD = 4.5689974f
            return Array(logMelSpec.size) { i -> FloatArray(logMelSpec[i].size) { j -> (logMelSpec[i][j] - WHISPER_MEL_MEAN) / WHISPER_MEL_STD } }
        }

        /**
         * Segment audio using energy-based silence detection
         * Matches Python implementation in segment_audio_001.py
         */
        fun segmentAudioEnergyBased(audio: FloatArray, thresholdDb: Float = -30f, minSilenceFrames: Int = 11): List<FloatArray> {
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

            // Find silent frames
            val silentFrames = frameEnergy.indices.filter { frameEnergy[it] < thresholdDb }

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

            // Filter segments >= 0.5s (matches Python)
            val segments = allSegments.filter { (start, end) -> (end - start) >= SAMPLE_RATE / 2 }
                .map { (start, end) -> audio.copyOfRange(start, end) }

            return segments
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
    }
}
