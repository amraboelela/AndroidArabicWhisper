#!/usr/bin/env python3
import sentencepiece as spm
import json

def convert_vocab_to_json():
    """Convert SentencePiece vocabulary to JSON format as an array"""

    model_file = "muhaffez_bpe_7k.model"
    output_file = "vocabulary.json"

    # Load the SentencePiece model
    sp = spm.SentencePieceProcessor()
    sp.load(model_file)

    # Create vocabulary array (index = token ID)
    vocab = []

    # Get all pieces in order of their IDs
    for idx in range(sp.get_piece_size()):
        piece = sp.id_to_piece(idx)
        vocab.append(piece)

    # Save to JSON as array
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"Vocabulary converted to JSON")
    print(f"Total tokens: {len(vocab)}")
    print(f"Output file: {output_file}")

if __name__ == "__main__":
    convert_vocab_to_json()
