"""Audio processing utilities for Whisper model.

This module provides audio loading, resampling, and feature extraction
functionality for the Whisper speech recognition model.
"""

import numpy as np
import struct
import math

# Constants
WHISPER_SAMPLE_RATE = 16000
WHISPER_N_FFT = 400
WHISPER_HOP_LENGTH = 160
WHISPER_N_MEL = 80


class WavReader:
    """WAV file reader for audio processing."""

    @staticmethod
    def read_wav_file(filename):
        """Read a WAV file and return audio data and header information.

        Args:
            filename: Path to the WAV file

        Returns:
            Tuple of (audio_data, header_dict) where audio_data is a numpy array
            of float32 samples in range [-1, 1] and header_dict contains:
                - num_channels: Number of audio channels
                - sample_rate: Sample rate in Hz
                - bits_per_sample: Bits per sample
                - data_size: Size of audio data in bytes
        """
        try:
            with open(filename, 'rb') as file:
                # Read RIFF header
                riff_header = file.read(12)
                if len(riff_header) != 12:
                    return None, None

                # Check RIFF header
                if riff_header[0:4] != b'RIFF' or riff_header[8:12] != b'WAVE':
                    return None, None

                # Initialize header
                header = {
                    'num_channels': 0,
                    'sample_rate': 0,
                    'bits_per_sample': 0,
                    'data_size': 0
                }

                found_fmt = False
                found_data = False

                # Read chunks
                while not (found_fmt and found_data):
                    chunk_header = file.read(8)
                    if len(chunk_header) != 8:
                        break

                    chunk_id = chunk_header[0:4]
                    chunk_size = struct.unpack('<I', chunk_header[4:8])[0]

                    if chunk_id == b'fmt ':
                        if chunk_size < 16:
                            return None, None

                        fmt_data = file.read(16)
                        if len(fmt_data) != 16:
                            return None, None

                        audio_format = struct.unpack('<H', fmt_data[0:2])[0]
                        header['num_channels'] = struct.unpack('<H', fmt_data[2:4])[0]
                        header['sample_rate'] = struct.unpack('<I', fmt_data[4:8])[0]
                        header['bits_per_sample'] = struct.unpack('<H', fmt_data[14:16])[0]

                        if audio_format != 1:  # Only support PCM
                            return None, None

                        found_fmt = True

                        # Skip remaining bytes in chunk
                        if chunk_size > 16:
                            file.seek(chunk_size - 16, 1)

                    elif chunk_id == b'data':
                        header['data_size'] = chunk_size
                        found_data = True
                        break

                    else:
                        # Skip unknown chunk
                        file.seek(chunk_size, 1)

                    # Align to even byte boundary
                    if chunk_size % 2 == 1:
                        file.seek(1, 1)

                if not (found_fmt and found_data):
                    return None, None

                # Read audio data
                num_samples = header['data_size'] // (header['bits_per_sample'] // 8)

                if header['bits_per_sample'] == 16:
                    int16_data = file.read(header['data_size'])
                    if len(int16_data) != header['data_size']:
                        return None, None

                    # Convert to float32 array
                    audio = np.frombuffer(int16_data, dtype=np.int16)
                    audio = audio.astype(np.float32) / 32768.0
                else:
                    # Only support 16-bit for now
                    return None, None

                return audio, header

        except Exception as e:
            print(f"Error reading WAV file: {e}")
            return None, None


class AudioProcessor:
    """Audio processing utilities for Whisper model."""

    @staticmethod
    def load_audio(filename):
        """Load audio file and preprocess for Whisper model.

        Args:
            filename: Path to audio file

        Returns:
            Numpy array of preprocessed audio samples
        """
        audio, header = WavReader.read_wav_file(filename)

        if audio is None:
            print(f"Failed to load audio file: {filename}")
            return np.array([])

        # Convert to mono if stereo
        if header['num_channels'] == 2:
            audio = AudioProcessor.stereo_to_mono(audio)

        # Resample to 16kHz if needed
        if header['sample_rate'] != WHISPER_SAMPLE_RATE:
            audio = AudioProcessor.resample_audio(audio, header['sample_rate'])

        # Normalize audio
        audio = AudioProcessor.normalize_audio(audio)

        return audio

    @staticmethod
    def resample_audio(audio, input_sample_rate):
        """Resample audio to WHISPER_SAMPLE_RATE using linear interpolation.

        Args:
            audio: Input audio array
            input_sample_rate: Input sample rate

        Returns:
            Resampled audio array
        """
        if input_sample_rate == WHISPER_SAMPLE_RATE:
            return audio

        ratio = input_sample_rate / WHISPER_SAMPLE_RATE
        output_size = int(len(audio) / ratio)
        resampled = np.zeros(output_size, dtype=np.float32)

        for i in range(output_size):
            src_index = i * ratio
            index = int(src_index)
            frac = src_index - index

            if index + 1 < len(audio):
                resampled[i] = audio[index] * (1.0 - frac) + audio[index + 1] * frac
            else:
                resampled[i] = audio[index]

        return resampled

    @staticmethod
    def stereo_to_mono(stereo_audio):
        """Convert stereo audio to mono by averaging channels.

        Args:
            stereo_audio: Stereo audio array (interleaved L/R samples)

        Returns:
            Mono audio array
        """
        mono_audio = (stereo_audio[0::2] + stereo_audio[1::2]) * 0.5
        return mono_audio

    @staticmethod
    def pad_or_trim(audio, length):
        """Pad or trim audio to specified length.

        Args:
            audio: Input audio array
            length: Target length

        Returns:
            Padded or trimmed audio array
        """
        if len(audio) == length:
            return audio
        elif len(audio) > length:
            return audio[:length]
        else:
            padded = np.pad(audio, (0, length - len(audio)), mode='constant')
            return padded

    @staticmethod
    def normalize_audio(audio):
        """Normalize audio to [-1, 1] range.

        Args:
            audio: Input audio array

        Returns:
            Normalized audio array
        """
        if len(audio) == 0:
            return audio

        max_val = np.max(np.abs(audio))

        if max_val == 0.0:
            return audio

        return audio / max_val

    @staticmethod
    def apply_preemphasis(audio, alpha=0.97):
        """Apply pre-emphasis filter to audio.

        Args:
            audio: Input audio array
            alpha: Pre-emphasis coefficient

        Returns:
            Filtered audio array
        """
        if len(audio) == 0:
            return audio

        filtered = np.zeros_like(audio)
        filtered[0] = audio[0]
        filtered[1:] = audio[1:] - alpha * audio[:-1]

        return filtered

    @staticmethod
    def apply_hann_window(window_size):
        """Create Hann window.

        Args:
            window_size: Size of window

        Returns:
            Hann window array
        """
        n = np.arange(window_size)
        window = 0.5 * (1.0 - np.cos(2.0 * np.pi * n / (window_size - 1)))
        return window

    @staticmethod
    def compute_stft(audio):
        """Compute Short-Time Fourier Transform magnitude.

        Args:
            audio: Input audio array

        Returns:
            2D array of STFT magnitudes [frames, frequencies]
        """
        window_size = WHISPER_N_FFT
        hop_size = WHISPER_HOP_LENGTH

        window = AudioProcessor.apply_hann_window(window_size)

        num_frames = (len(audio) - window_size) // hop_size + 1
        if num_frames <= 0:
            num_frames = 1

        stft_magnitude = []

        for frame in range(num_frames):
            start_idx = frame * hop_size
            end_idx = start_idx + window_size

            if end_idx > len(audio):
                audio_frame = np.pad(audio[start_idx:], (0, end_idx - len(audio)), mode='constant')
            else:
                audio_frame = audio[start_idx:end_idx]

            windowed = audio_frame * window

            # Compute DFT (simplified, for production use numpy.fft)
            magnitudes = []
            for freq in range(window_size // 2 + 1):
                angle = -2.0 * np.pi * freq * np.arange(window_size) / window_size
                real_part = np.sum(windowed * np.cos(angle))
                imag_part = np.sum(windowed * np.sin(angle))
                magnitude = np.sqrt(real_part**2 + imag_part**2)
                magnitudes.append(magnitude)

            stft_magnitude.append(magnitudes)

        return np.array(stft_magnitude)

    @staticmethod
    def hz_to_mel(hz):
        """Convert frequency in Hz to mel scale.

        Args:
            hz: Frequency in Hz

        Returns:
            Frequency in mel scale
        """
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    @staticmethod
    def mel_to_hz(mel):
        """Convert mel scale to frequency in Hz.

        Args:
            mel: Frequency in mel scale

        Returns:
            Frequency in Hz
        """
        return 700.0 * (10.0**(mel / 2595.0) - 1.0)

    @staticmethod
    def get_mel_filter_bank():
        """Create mel filter bank for mel spectrogram computation.

        Returns:
            2D array of mel filters [n_mel, n_fft//2 + 1]
        """
        mel_filters = np.zeros((WHISPER_N_MEL, WHISPER_N_FFT // 2 + 1), dtype=np.float32)

        mel_low = AudioProcessor.hz_to_mel(0.0)
        mel_high = AudioProcessor.hz_to_mel(WHISPER_SAMPLE_RATE / 2.0)

        # Create equally spaced mel points
        mel_points = np.linspace(mel_low, mel_high, WHISPER_N_MEL + 2)

        # Convert to Hz
        hz_points = AudioProcessor.mel_to_hz(mel_points)

        # Convert to FFT bins
        bin_points = np.floor((WHISPER_N_FFT + 1) * hz_points / WHISPER_SAMPLE_RATE).astype(int)

        # Create triangular filters
        for mel in range(WHISPER_N_MEL):
            left = bin_points[mel]
            center = bin_points[mel + 1]
            right = bin_points[mel + 2]

            # Left slope
            for bin_idx in range(left, center):
                if bin_idx < len(mel_filters[mel]):
                    mel_filters[mel][bin_idx] = (bin_idx - left) / (center - left)

            # Right slope
            for bin_idx in range(center, right):
                if bin_idx < len(mel_filters[mel]):
                    mel_filters[mel][bin_idx] = (right - bin_idx) / (right - center)

        return mel_filters

    @staticmethod
    def extract_mel_spectrogram(audio):
        """Extract mel spectrogram from audio.

        Args:
            audio: Input audio array

        Returns:
            2D array of mel spectrogram [n_mel, frames]
        """
        # Apply pre-emphasis
        filtered_audio = AudioProcessor.apply_preemphasis(audio)

        # Compute STFT
        stft = AudioProcessor.compute_stft(filtered_audio)

        # Get mel filter bank
        mel_filters = AudioProcessor.get_mel_filter_bank()

        # Apply mel filters to STFT magnitude
        mel_spec = np.dot(mel_filters, stft.T)

        return mel_spec

    @staticmethod
    def apply_log_transform(mel_spectrogram):
        """Apply logarithmic transform to mel spectrogram.

        Args:
            mel_spectrogram: Input mel spectrogram

        Returns:
            Log mel spectrogram
        """
        return np.log(np.maximum(mel_spectrogram, 1e-10))