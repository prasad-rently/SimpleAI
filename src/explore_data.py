"""
explore_data.py — Look at our training data before feeding it to the AI.

Before training any model, it's important to understand your data.
This script answers basic questions:
  - How much text do we have?
  - What characters appear in it?
  - What does the text look like?

This is a habit used by every AI/ML engineer: always inspect your data first.

Usage:
    python src/explore_data.py
"""


def main():
    # ------------------------------------------------------------------
    # 1. LOAD THE TEXT FILE
    # ------------------------------------------------------------------
    # open() reads a file from disk into memory as a string.
    # encoding="utf-8" ensures special characters are read correctly.
    filepath = "data/input.txt"

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    print("=" * 50)
    print("DATA EXPLORATION")
    print("=" * 50)
    print(f"File            : {filepath}")
    print()

    # ------------------------------------------------------------------
    # 2. BASIC STATISTICS
    # ------------------------------------------------------------------
    # These numbers tell us how "big" our dataset is.
    # More data = more patterns for the model to learn.

    total_chars = len(text)
    lines = text.strip().split("\n")
    total_lines = len(lines)
    words = text.split()
    total_words = len(words)

    print("--- Size ---")
    print(f"Total characters: {total_chars}")
    print(f"Total words     : {total_words}")
    print(f"Total lines     : {total_lines}")
    print()

    # ------------------------------------------------------------------
    # 3. CHARACTER ANALYSIS
    # ------------------------------------------------------------------
    # Find every UNIQUE character in the text. This is our "vocabulary".
    # The model can only learn characters it has seen — if the letter 'z'
    # never appears in the training data, the model won't know about it.

    unique_chars = sorted(set(text))
    vocab_size = len(unique_chars)

    print("--- Character vocabulary ---")
    print(f"Unique characters: {vocab_size}")
    print()

    # Show each character. We use repr() to make invisible characters
    # visible (like \n for newline, ' ' for space).
    print("All characters found:")
    for i, char in enumerate(unique_chars):
        # repr() shows 'a', '\n', ' ' etc. — makes whitespace visible
        display = repr(char)
        print(f"  [{i:2d}] {display}")
    print()

    # ------------------------------------------------------------------
    # 4. PREVIEW THE DATA
    # ------------------------------------------------------------------
    # Always look at a sample of your data. This catches problems early:
    # weird characters, encoding issues, unexpected formatting, etc.

    print("--- First 5 lines ---")
    for i, line in enumerate(lines[:5]):
        print(f"  {i + 1}: {line}")
    print()

    print("--- Last 3 lines ---")
    for line in lines[-3:]:
        print(f"  {line}")
    print()

    # ------------------------------------------------------------------
    # 5. CHARACTER FREQUENCY
    # ------------------------------------------------------------------
    # Which characters appear most often? This matters because:
    # - Common characters are easier for the model to learn
    # - Rare characters might not get enough training examples
    # - The distribution tells us about the nature of our text

    print("--- Top 15 most common characters ---")
    char_counts = {}
    for char in text:
        char_counts[char] = char_counts.get(char, 0) + 1

    # Sort by count, highest first
    sorted_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)

    for char, count in sorted_chars[:15]:
        percentage = (count / total_chars) * 100
        bar = "#" * int(percentage * 2)  # Simple visual bar
        print(f"  {repr(char):6s} : {count:4d} ({percentage:5.1f}%) {bar}")
    print()

    # ------------------------------------------------------------------
    # 6. WHAT THIS MEANS FOR OUR MODEL
    # ------------------------------------------------------------------
    print("=" * 50)
    print("WHAT THIS MEANS FOR OUR AI MODEL")
    print("=" * 50)
    print(f"""
Our model will:
  - Learn a vocabulary of {vocab_size} characters
  - Train on {total_chars} characters of text
  - Try to predict the next character given previous ones

The more text we have, the better the model can learn patterns.
With {total_lines} lines of quotes, the model should learn common
English patterns like "th", "the", "ing", "tion", etc.

Next step: convert these characters into numbers (Step 04).
""")


if __name__ == "__main__":
    main()
