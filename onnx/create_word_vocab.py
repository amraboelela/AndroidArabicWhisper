#!/usr/bin/env python3
import json
from collections import Counter

def create_word_vocabulary():
    """Create vocabulary from unique words in normalized Quran text"""

    input_file = "quran-simple-norm.txt"
    output_file = "vocabulary_words.json"

    # Read all text
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Split into words and count frequency
    words = text.split()
    word_freq = Counter(words)

    # Sort by frequency (most common first)
    sorted_words = [word for word, count in word_freq.most_common()]

    # Create vocabulary with special tokens at the beginning
    vocab = ["<unk>", "<s>", "</s>"] + sorted_words

    # Save to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"Word vocabulary created")
    print(f"Total unique words: {len(sorted_words)}")
    print(f"Total vocabulary size (with special tokens): {len(vocab)}")
    print(f"Output file: {output_file}")
    print(f"\nTop 20 most frequent words:")
    for i, (word, count) in enumerate(word_freq.most_common(20), 1):
        print(f"  {i}. '{word}' - {count} times")

if __name__ == "__main__":
    create_word_vocabulary()
