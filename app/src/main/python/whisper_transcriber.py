"""
Whisper transcription module using faster-whisper
"""
from faster_whisper import WhisperModel
import os


# Module-level model instance
_model = None


def init_model(model_size="base", device="cpu", compute_type="int8"):
    """Initialize the Whisper model"""
    global _model
    _model = WhisperModel(model_size, device=device, compute_type=compute_type)
    return "Model initialized"


def transcribe_audio(audio_path, language="ar", beam_size=5):
    """
    Transcribe audio file

    Args:
        audio_path: Path to audio file
        language: Language code (ar for Arabic, en for English)
        beam_size: Beam size for decoding

    Returns:
        Transcription text
    """
    global _model

    if not os.path.exists(audio_path):
        return f"Error: Audio file not found: {audio_path}"

    # Initialize model if not already done
    if _model is None:
        _model = WhisperModel("base", device="cpu", compute_type="int8")

    try:
        segments, info = _model.transcribe(
            audio_path,
            language=language,
            beam_size=beam_size,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )

        # Collect all segments
        transcription = " ".join(segment.text for segment in segments)
        return transcription.strip()

    except Exception as e:
        return f"Error during transcription: {str(e)}"


def transcribe_audio_with_model(audio_path, model_size="base", language="ar"):
    """
    Transcribe audio file with specific model size

    Args:
        audio_path: Path to audio file
        model_size: Model size (tiny, base, small, medium, large-v3)
        language: Language code (ar for Arabic, en for English)

    Returns:
        Transcription text
    """
    if not os.path.exists(audio_path):
        return f"Error: Audio file not found: {audio_path}"

    try:
        # Create new model instance with specified size
        model = WhisperModel(model_size, device="cpu", compute_type="int8")

        segments, info = model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )

        # Collect all segments
        transcription = " ".join(segment.text for segment in segments)
        return transcription.strip()

    except Exception as e:
        return f"Error during transcription: {str(e)}"

