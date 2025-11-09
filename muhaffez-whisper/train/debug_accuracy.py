#!/usr/bin/env python3
"""Debug script to check accuracy per surah part"""
import sys
sys.path.append("..")
import torch
import json
import glob
import os
from train_all_full import calculate_accuracy
from tools.encoder_decoder_transformer import EncoderDecoderTransformer

# Setup
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
vocab_path = '../models/vocabulary.json'
model_path = '../models/muhaffez_whisper.pt'
dataset_name = 'Quran-A'
datasets_dir = f'../datasets/{dataset_name}'

# Load vocab
with open(vocab_path, 'r', encoding='utf-8') as f:
    vocab = json.load(f)

# Load model
model = EncoderDecoderTransformer(
    vocab_size=len(vocab),
    d_model=128,
    n_encoder_layers=4,
    n_decoder_layers=4,
    n_heads=4,
    d_ff=512,
    dropout=0.1
)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model = model.to(device)

# Test each surah part separately
text_files = sorted(glob.glob(f'{datasets_dir}/text/*.txt'))
print(f"Testing accuracy per surah part:\n")

all_segment_files = []
all_transcriptions = []

for text_file in text_files:
    surah_part = os.path.splitext(os.path.basename(text_file))[0]
    surah_num = surah_part.split('-')[0]
    audio_dir = f'{datasets_dir}/audio/{surah_num}'

    with open(text_file, 'r', encoding='utf-8') as f:
        transcriptions = [line.strip() for line in f if line.strip()]

    segment_files = sorted(glob.glob(f'{audio_dir}/{surah_part}-*.wav'))

    if len(segment_files) != len(transcriptions):
        print(f"⚠️  Skipping {surah_part}: mismatch")
        continue

    # Calculate accuracy for this part
    acc = calculate_accuracy(model, segment_files, transcriptions, vocab, device)
    print(f'{surah_part}: {acc:.1f}% ({len(segment_files)} segments)')

    # Collect for overall
    all_segment_files.extend(segment_files)
    all_transcriptions.extend(transcriptions)

# Calculate overall accuracy
print(f"\n{'='*60}")
overall_acc = calculate_accuracy(model, all_segment_files, all_transcriptions, vocab, device)
print(f'OVERALL: {overall_acc:.1f}% ({len(all_segment_files)} segments)')
print(f"{'='*60}")
