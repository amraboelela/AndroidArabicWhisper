"""Audio decoding for Android - simplified version without PyAV dependency.

For Android, we use a simple WAV file reader since PyAV doesn't work well on Android.
This is sufficient for our use case with pre-recorded WAV files.
"""

import io
import struct
import wave
from typing import BinaryIO, Union

import numpy as np


def decode_audio(
    input_file: Union[str, BinaryIO],
    sampling_rate: int = 16000,
    split_stereo: bool = False,
):
    """Decodes the audio from WAV file.

    Args:
      input_file: Path to the input WAV file or a file-like object.
      sampling_rate: Target sample rate (must match input for now).
      split_stereo: Return separate left and right channels.

    Returns:
      A float32 Numpy array.

      If `split_stereo` is enabled, the function returns a 2-tuple with the
      separated left and right channels.
    """
    # Open WAV file
    if isinstance(input_file, str):
        wav_file = wave.open(input_file, 'rb')
    else:
        wav_file = wave.open(input_file, 'rb')

    # Get audio parameters
    n_channels = wav_file.getnchannels()
    sampwidth = wav_file.getsampwidth()
    framerate = wav_file.getframerate()
    n_frames = wav_file.getnframes()

    # Read audio data
    audio_data = wav_file.readframes(n_frames)
    wav_file.close()

    # Convert bytes to numpy array
    if sampwidth == 1:
        dtype = np.uint8
        audio = np.frombuffer(audio_data, dtype=dtype)
        audio = (audio.astype(np.float32) - 128) / 128.0
    elif sampwidth == 2:
        dtype = np.int16
        audio = np.frombuffer(audio_data, dtype=dtype)
        audio = audio.astype(np.float32) / 32768.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    # Handle stereo to mono conversion
    if n_channels == 2 and not split_stereo:
        audio = audio.reshape(-1, 2)
        audio = audio.mean(axis=1)
    elif n_channels == 2 and split_stereo:
        audio = audio.reshape(-1, 2)
        left_channel = audio[:, 0]
        right_channel = audio[:, 1]
        return left_channel, right_channel

    # Simple resampling if needed (basic decimation/interpolation)
    if framerate != sampling_rate:
        # Calculate resampling ratio
        ratio = sampling_rate / framerate
        new_length = int(len(audio) * ratio)

        # Simple linear interpolation for resampling
        indices = np.linspace(0, len(audio) - 1, new_length)
        audio = np.interp(indices, np.arange(len(audio)), audio)

    return audio


def pad_or_trim(array, length: int = 3000, *, axis: int = -1):
    """
    Pad or trim the Mel features array to 3000, as expected by the encoder.
    """
    if array.shape[axis] > length:
        array = array.take(indices=range(length), axis=axis)

    if array.shape[axis] < length:
        pad_widths = [(0, 0)] * array.ndim
        pad_widths[axis] = (0, length - array.shape[axis])
        array = np.pad(array, pad_widths)

    return array
