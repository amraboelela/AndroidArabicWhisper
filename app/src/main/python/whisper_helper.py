from faster_whisper import WhisperModel

# You can keep a global model so it loads once
model = None

def init_model(model_path: str):
    global model
    model = WhisperModel(model_path, device="cpu")
    return "Model initialized"

def transcribe(audio_path: str):
    if model is None:
        return "Model not initialized"

    segments, info = model.transcribe(audio_path)
    text = ""
    for segment in segments:
        text += segment.text + " "
    return text.strip()

