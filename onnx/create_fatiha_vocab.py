#!/usr/bin/env python3
"""
Create a minimal vocabulary for Al-Fatiha only
"""
import json

# Read Al-Fatiha text
with open("segments/001.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

# Collect all unique words
words_set = set()
for line in lines:
    words = line.split()
    words_set.update(words)

# Sort words alphabetically
words = sorted(list(words_set))

# Create vocabulary with special tokens
vocab = ["<unk>", "<s>", "</s>"] + words

print(f"Al-Fatiha Vocabulary ({len(vocab)} tokens):")
print("="*60)
print("Special tokens:")
for i in range(3):
    print(f"  {i}: {vocab[i]}")

print(f"\nAl-Fatiha words ({len(words)} unique):")
for i, word in enumerate(words, start=3):
    print(f"  {i}: {word}")

# Save vocabulary
output_path = "vocabulary_fatiha.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)

print(f"\n✓ Vocabulary saved to: {output_path}")
print(f"  Total tokens: {len(vocab)}")
