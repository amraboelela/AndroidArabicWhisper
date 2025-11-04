import numpy as np
import time
import onnxruntime as ort
from faster_whisper.audio import decode_audio
from faster_whisper.feature_extractor import FeatureExtractor

print("="*60)
print("ONNX Runtime Inference Speed Test (Python)")
print("="*60)

# Load audio
print("\n📁 Loading audio...")
audio_path = "../app/src/main/assets/001.wav"
audio = decode_audio(audio_path, sampling_rate=16000)
print(f"✅ Audio shape: {audio.shape}, duration: {len(audio)/16000:.2f}s")

# Extract features
print("\n🔧 Extracting mel features...")
feature_extractor = FeatureExtractor()
start = time.time()
features = feature_extractor(audio)
preprocessing_time = (time.time() - start) * 1000
print(f"✅ Features shape: {features.shape}")
print(f"⏱️  Preprocessing: {preprocessing_time:.1f}ms")

# Load ONNX models
print("\n📦 Loading ONNX models...")
encoder_path = "/Users/amraboelela/develop/android/AndroidArabicWhisper/onnx/models/custom-whisper-ar-quran-onnx-simplified/encoder_model.onnx"
decoder_path = "/Users/amraboelela/develop/android/AndroidArabicWhisper/onnx/models/custom-whisper-ar-quran-onnx-simplified/decoder_model.onnx"

sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 4
sess_options.inter_op_num_threads = 4

encoder_session = ort.InferenceSession(encoder_path, sess_options)
decoder_session = ort.InferenceSession(decoder_path, sess_options)

print(f"✅ Encoder providers: {encoder_session.get_providers()}")
print(f"✅ Decoder providers: {decoder_session.get_providers()}")

# Run encoder
print("\n🔧 Running ONNX encoder...")
# Pad/truncate to 3000 frames (30 seconds)
if features.shape[1] > 3000:
    features = features[:, :3000]
elif features.shape[1] < 3000:
    padded = np.zeros((80, 3000), dtype=features.dtype)
    padded[:, :features.shape[1]] = features
    features = padded

encoder_input = features[np.newaxis, :, :].astype(np.float32)
print(f"   Encoder input shape: {encoder_input.shape}")

start = time.time()
encoder_outputs = encoder_session.run(None, {"input_features": encoder_input})
encoder_time = (time.time() - start) * 1000

encoder_hidden_states = encoder_outputs[0]
print(f"✅ Encoder output shape: {encoder_hidden_states.shape}")
print(f"⏱️  Encoder time: {encoder_time:.1f}ms")

# Run decoder (autoregressive)
print("\n🔧 Running ONNX decoder...")
decoder_start_token = 50258  # <|startoftranscript|>
lang_token = 50272           # <|ar|>
task_token = 50359           # <|transcribe|>
no_timestamps_token = 50363  # <|notimestamps|>
eos_token = 50257            # <|endoftext|>

generated_tokens = [decoder_start_token, lang_token, task_token, no_timestamps_token]
max_length = 200

start = time.time()
for step in range(max_length):
    input_ids = np.array([generated_tokens], dtype=np.int64)

    decoder_outputs = decoder_session.run(
        None,
        {
            "input_ids": input_ids,
            "encoder_hidden_states": encoder_hidden_states
        }
    )

    logits = decoder_outputs[0]
    next_token = np.argmax(logits[0, -1, :])

    if next_token == eos_token:
        print(f"   🛑 EOS token reached at step {step}")
        break

    generated_tokens.append(int(next_token))

decoder_time = (time.time() - start) * 1000
print(f"✅ Decoder generated {len(generated_tokens)} tokens")
print(f"⏱️  Decoder time: {decoder_time:.1f}ms")

# Total inference time
total_inference = encoder_time + decoder_time
total_time = preprocessing_time + total_inference

print("\n" + "="*60)
print("TIMING SUMMARY (Python on Mac)")
print("="*60)
print(f"Preprocessing:  {preprocessing_time:>8.1f}ms")
print(f"Encoder:        {encoder_time:>8.1f}ms")
print(f"Decoder:        {decoder_time:>8.1f}ms")
print(f"Total Inference:{total_inference:>8.1f}ms")
print(f"TOTAL:          {total_time:>8.1f}ms")

print("\n" + "="*60)
print("COMPARISON WITH ANDROID (Kotlin on Emulator)")
print("="*60)
print(f"{'':20} {'Python (Mac)':<15} {'Kotlin (Emulator)':<20} {'Ratio':<10}")
print(f"{'-'*60}")
print(f"{'Preprocessing:':<20} {preprocessing_time:>8.1f}ms     {675:>8.1f}ms           {675/preprocessing_time:>6.1f}x")
print(f"{'Encoder:':<20} {encoder_time:>8.1f}ms     {374:>8.1f}ms           {374/encoder_time:>6.1f}x")
print(f"{'Decoder:':<20} {decoder_time:>8.1f}ms     {12115:>8.1f}ms         {12115/decoder_time:>6.1f}x")
print(f"{'TOTAL:':<20} {total_time:>8.1f}ms     {13164:>8.1f}ms         {13164/total_time:>6.1f}x")
