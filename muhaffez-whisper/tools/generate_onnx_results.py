#!/usr/bin/env python3
"""
Generate 002-04-onnx.txt with transcriptions from ONNX model
"""

import os
import numpy as np
import onnxruntime as ort
import torchaudio
import torch
from transformers import WhisperProcessor

def transcribe_audio_onnx(encoder_session, decoder_session, processor, audio_path, max_length=200):
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    audio_array = waveform.squeeze().numpy()

    # Extract features
    input_features = processor(audio_array, sampling_rate=16000, return_tensors='pt').input_features
    input_features_np = input_features.numpy()

    # Run encoder
    encoder_outputs = encoder_session.run(None, {'input_features': input_features_np})
    encoder_hidden_states = encoder_outputs[0]

    # Decoder tokens
    generated_tokens = [50258, 50272, 50359, 50363]
    decoder_input_ids = np.array([generated_tokens], dtype=np.int64)

    for step in range(max_length):
        decoder_outputs = decoder_session.run(None, {
            'input_ids': decoder_input_ids,
            'encoder_hidden_states': encoder_hidden_states
        })
        logits = decoder_outputs[0]
        next_token = int(np.argmax(logits[0, -1, :]))
        if next_token == 50257:
            break
        generated_tokens.append(next_token)
        decoder_input_ids = np.array([generated_tokens], dtype=np.int64)

    return processor.decode(generated_tokens, skip_special_tokens=True)

def main():
    print("="*70)
    print("Generating 002-04-onnx.txt from ONNX Model")
    print("="*70)

    # Load model
    model_dir = 'models/whisper-base-ar-quran-onnx-simplified'
    print(f"\nLoading model from {model_dir}...")
    processor = WhisperProcessor.from_pretrained(model_dir)
    encoder_path = os.path.join(model_dir, 'encoder_model.onnx')
    decoder_path = os.path.join(model_dir, 'decoder_model.onnx')
    encoder_session = ort.InferenceSession(encoder_path)
    decoder_session = ort.InferenceSession(decoder_path)
    print("  ✓ Model loaded")

    # Process all files
    audio_dir = 'datasets/base/audio'
    audio_files = sorted([f for f in os.listdir(audio_dir) if f.startswith('002-04-') and f.endswith('.wav')])

    print(f"\nTranscribing {len(audio_files)} audio files...")

    results = []
    for i, audio_file in enumerate(audio_files):
        audio_path = os.path.join(audio_dir, audio_file)
        print(f'[{i+1}/{len(audio_files)}] {audio_file}', end='')
        try:
            transcription = transcribe_audio_onnx(encoder_session, decoder_session, processor, audio_path)
            results.append(transcription)
            print(f' ✓')
        except Exception as e:
            print(f' ✗ Error: {e}')
            results.append('')

    # Write results
    output_file = 'datasets/base/text/002-04-onnx.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in results:
            f.write(line + '\n')

    print(f"\n{'='*70}")
    print(f"✓ Wrote {len(results)} lines to {output_file}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
