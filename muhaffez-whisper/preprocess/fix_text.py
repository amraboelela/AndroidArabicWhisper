#!/usr/bin/env python3
"""
Check transcribed text against vocabulary and auto-fix words with closest matches
Also uses context matching with reference surah text for difficult cases
Usage: python3 fix_text.py <dataset_name> <segment_name>
       python3 fix_text.py Quran-A 002-02
"""
import os
import sys
import json
from difflib import get_close_matches

def load_vocabulary():
    """Load vocabulary from JSON file"""
    vocab_path = os.path.join(os.path.dirname(__file__), "..", "models", "vocabulary.json")
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_list = json.load(f)
    return set(vocab_list), vocab_list

def load_surah_text(dataset_name, surah_number):
    """Load complete surah reference text"""
    surah_file = os.path.join(os.path.dirname(__file__), "..", "datasets", "quran-text", f"{surah_number}.txt")

    if not os.path.exists(surah_file):
        print(f"⚠  Warning: Surah reference file not found: {surah_file}")
        return "", set()

    with open(surah_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Extract all words from the surah
    words = set(text.split())
    return text, words

def find_word_by_context(unknown_word, words_in_line, surah_text, word_index, prev_line_words=None):
    """
    Find correct word using context from reference surah text
    - For words with 3 or 2 words before in current line: uses them
    - For first word in line: uses last 3 or 2 words from previous line (if available)
    - Otherwise: uses 2 words AFTER

    Args:
        unknown_word: The word we don't recognize
        words_in_line: All words in the current line
        surah_text: Full reference surah text
        word_index: Index of unknown_word in words_in_line
        prev_line_words: Words from the previous line (for context at line start)

    Returns:
        The correct word from reference text, or None if not found
    """
    # If this is the first word in line and we have previous line, use its ending as context
    if word_index == 0 and prev_line_words:
        for context_size in [3, 2]:
            if len(prev_line_words) >= context_size:
                # Use last N words from previous line as BEFORE context
                before_words = prev_line_words[-context_size:]
                result = _find_with_before_context(unknown_word, before_words, surah_text)
                if result:
                    return result

    # Try with cascading context sizes: 3, then 2 words BEFORE
    for context_size in [3, 2]:
        result = _find_with_context_size(unknown_word, words_in_line, surah_text, word_index, context_size)
        if result:
            return result

    # If only 1 word or no words before, use 2 words AFTER instead
    after_words = words_in_line[word_index + 1:min(len(words_in_line), word_index + 3)]
    if len(after_words) >= 2:
        return _find_with_after_context(unknown_word, after_words, surah_text)

    return None

def _find_with_before_context(unknown_word, before_words, surah_text):
    """Helper function to find word using explicit BEFORE context words"""
    surah_words = surah_text.split()

    # Search for this context in the surah text
    for i in range(len(before_words), len(surah_words)):
        # Check if we have a match for before context
        before_match = True
        start_idx = i - len(before_words)

        for j, word in enumerate(before_words):
            if surah_words[start_idx + j] != word:
                before_match = False
                break

        # If before context matches, check if candidate is similar to unknown_word
        if before_match:
            candidate = surah_words[i]
            # At least 55% character overlap based on unique chars
            common_chars = set(candidate) & set(unknown_word)
            unique_unknown = len(set(unknown_word))
            if unique_unknown > 0 and len(common_chars) >= unique_unknown * 0.55:
                return candidate

    return None

def _find_with_context_size(unknown_word, words_in_line, surah_text, word_index, context_size):
    """Helper function to find word with specific context size (BEFORE only)"""
    # Get words before unknown word
    before_words = words_in_line[max(0, word_index - context_size):word_index]

    # Special case: if this is the first word (no before context), use 2 words AFTER
    if not before_words and word_index == 0:
        after_words = words_in_line[1:min(len(words_in_line), 3)]  # Get 2 words after
        if after_words:
            return _find_with_after_context(unknown_word, after_words, surah_text)

    if not before_words:
        return None

    # Search for this context in the surah text
    surah_words = surah_text.split()

    # Try to find matching context in reference text
    for i in range(len(before_words), len(surah_words)):
        # Check if we have a match for before context
        before_match = True
        start_idx = i - len(before_words)

        for j, word in enumerate(before_words):
            if surah_words[start_idx + j] != word:
                before_match = False
                break

        # If before context matches, check if candidate is similar to unknown_word
        if before_match:
            candidate = surah_words[i]
            # At least 55% character overlap based on unique chars
            common_chars = set(candidate) & set(unknown_word)
            unique_unknown = len(set(unknown_word))
            if unique_unknown > 0 and len(common_chars) >= unique_unknown * 0.55:
                return candidate

    return None

def _find_with_after_context(unknown_word, after_words, surah_text):
    """Helper function to find first word using AFTER context (2 words after)"""
    surah_words = surah_text.split()

    # Search for the after context in surah
    for i in range(len(surah_words) - len(after_words)):
        # Check if the words after match
        after_match = True
        for j, word in enumerate(after_words):
            if surah_words[i + 1 + j] != word:
                after_match = False
                break

        # If after context matches, check if candidate is similar to unknown_word
        if after_match:
            candidate = surah_words[i]
            # At least 55% character overlap based on unique chars
            common_chars = set(candidate) & set(unknown_word)
            unique_unknown = len(set(unknown_word))
            if unique_unknown > 0 and len(common_chars) >= unique_unknown * 0.55:
                return candidate

    return None

def fix_text_file(text_file, vocab, vocab_list, surah_text, surah_words, segment_name):
    """Fix words in text file using context matching only:
    - For words in middle/end of line: uses 3→2 words BEFORE (cascading)
    - For first word in line: uses last 3→2 words from PREVIOUS line
    - Otherwise: uses 2 words AFTER
    - Multi-pass: keeps running until no more fixes are made

    Note: Fuzzy matching is disabled because it can give wrong matches.
          Context matching is more reliable for finding the correct word.
    """
    fixed_count = 0
    context_fixed_count = 0
    unfixed_words = {}

    with open(text_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    prev_line_words = None  # Track previous line's words for context
    for line_num, line in enumerate(lines, 1):
        original_line = line.strip()
        if not original_line:
            new_lines.append(line)
            continue

        words = original_line.split()
        new_words = []

        for word_idx, word in enumerate(words):
            if word in vocab:
                new_words.append(word)
            else:
                # Skip fuzzy matching - go straight to context matching
                # Fuzzy matching can give wrong matches even if they're in the surah
                best_match = None
                match_method = ""

                # Use context-based matching (pass previous line for first word context)
                context_match = find_word_by_context(word, words, surah_text, word_idx, prev_line_words)
                if context_match and context_match in vocab:
                    best_match = context_match
                    match_method = f"(context match in surah {segment_name.split('-')[0]})"
                    context_fixed_count += 1

                if best_match:
                    new_words.append(best_match)
                    fixed_count += 1
                    print(f"Line {line_num}: '{word}' → '{best_match}' {match_method}")
                else:
                    # Keep original if no close match
                    new_words.append(word)
                    if word not in unfixed_words:
                        unfixed_words[word] = [line_num]
                    else:
                        unfixed_words[word].append(line_num)

        new_lines.append(" ".join(new_words) + "\n")
        # Update previous line words for next iteration
        prev_line_words = new_words

    # Write back
    with open(text_file, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    return fixed_count, context_fixed_count, unfixed_words

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 fix_text.py <dataset_name> <segment_name>")
        print("Example: python3 fix_text.py Quran-A 002-02")
        sys.exit(1)

    dataset_name = sys.argv[1]
    segment_name = sys.argv[2]

    # Extract surah number from segment name (e.g., "002-02" -> "002")
    surah_number = segment_name.split('-')[0]

    # Load vocabulary
    vocab, vocab_list = load_vocabulary()

    # Load reference surah text
    surah_text, surah_words = load_surah_text(dataset_name, surah_number)
    if surah_words:
        print(f"✓ Loaded {len(surah_words)} unique words from surah {surah_number} reference")

    # Fix text file
    text_file = os.path.join(os.path.dirname(__file__), "..", "datasets", dataset_name, "text", f"{segment_name}.txt")

    if not os.path.exists(text_file):
        print(f"✗ Text file not found: {text_file}")
        sys.exit(1)

    # Multi-pass fixing: keep running until no more fixes are made
    pass_num = 1
    total_fixed = 0

    while True:
        print(f"\n{'='*60}")
        print(f"Pass {pass_num}: Fixing words in {segment_name}.txt")
        print('='*60)

        fixed_count, context_fixed_count, unfixed_words = fix_text_file(text_file, vocab, vocab_list, surah_text, surah_words, segment_name)

        if fixed_count > 0:
            total_fixed += fixed_count
            print(f"✓ Pass {pass_num} fixed {fixed_count} words ({context_fixed_count} using context matching)")
            pass_num += 1
        else:
            # No more fixes made, stop
            break

    print(f"\n{'='*60}")
    print(f"✓ Completed {pass_num - 1} pass(es), fixed {total_fixed} total words")
    print('='*60)

    if unfixed_words:
        print(f"\n⚠  {len(unfixed_words)} words remain unfixed (no close match):")
        for word, lines in sorted(unfixed_words.items()):
            lines_str = ", ".join(map(str, lines))
            print(f"   '{word}' (lines: {lines_str})")
        print(f"\n💡 Tip: Ask Claude to fix the remaining {len(unfixed_words)} word(s) in {text_file}")
        return 0  # Don't fail, just warn
    else:
        print("\n✓ All words match vocabulary")

    return 0

if __name__ == "__main__":
    sys.exit(main())
