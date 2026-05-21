"""
vocabulary.py — Convert text to numbers and back (Tokenization).

PURPOSE:
    Neural networks can ONLY work with numbers. They cannot process
    letters, words, or sentences directly. So we need a way to convert
    text into numbers before feeding it to the model, and convert the
    model's number outputs back into text.

    This is called "TOKENIZATION" — breaking text into "tokens" and
    assigning each token a unique number.

    In our case, each TOKEN is a single CHARACTER. So:
      'a' → 22     (character to number — "encoding")
      22  → 'a'    (number to character — "decoding")

    Real LLMs (like ChatGPT) use more sophisticated tokenization
    where tokens are word pieces (e.g., "running" → "run" + "ning"),
    but character-level is simpler and perfect for learning.

WHAT THIS FILE PROVIDES:
    Two classes:
    1. Vocabulary  — builds and stores the character↔number mappings
    2. (used by other files via import)

    Two standalone demo functions:
    3. demonstrate_encoding()  — shows encoding in action
    4. demonstrate_decoding()  — shows decoding in action

THE BIG IDEA:
    ┌──────────┐    encode()    ┌──────────┐
    │  "Hello" │ ──────────────▶│ [13,26,  │
    │  (text)  │                │  33,33,  │
    │          │    decode()    │  36]     │
    │          │ ◀──────────────│ (numbers)│
    └──────────┘                └──────────┘

INPUT:  A text file (data/input.txt) to build vocabulary from
OUTPUT: Character↔number mappings, plus encoding/decoding demos

Usage:
    python src/vocabulary.py
"""


class Vocabulary:
    """
    Builds a two-way mapping between characters and numbers.

    WHAT IT STORES:
        chars       — sorted list of all unique characters
        vocab_size  — how many unique characters there are
        char_to_idx — dictionary mapping each character to a number
        idx_to_char — dictionary mapping each number back to a character

    WHY TWO DICTIONARIES?
        We need to go BOTH directions:
        - Encoding (text → numbers): to feed text into the model
        - Decoding (numbers → text): to read the model's output

    EXAMPLE:
        If our text contains only "abc", the mappings would be:
            char_to_idx = {'a': 0, 'b': 1, 'c': 2}
            idx_to_char = {0: 'a', 1: 'b', 2: 'c'}

    HOW IT'S USED IN THE PROJECT:
        1. Step 04 (this file): Build the vocabulary from training text
        2. Step 05 (dataset.py): Use encode() to convert training text to numbers
        3. Step 07 (model.py): vocab_size tells the model how many characters exist
        4. Step 12 (generate.py): Use decode() to convert model output to text
    """

    def __init__(self, text):
        """
        Build the vocabulary from a text string.

        HOW IT WORKS:
            1. set(text)    — find all unique characters
            2. sorted(...)  — put them in consistent order
            3. enumerate    — assign each character a number (0, 1, 2, ...)

        WHY sorted()?
            Without sorting, set() returns characters in arbitrary order.
            Sorting ensures the SAME text always produces the SAME mappings.
            This is important because:
            - A model saved today must use the same mappings when loaded tomorrow
            - Different runs should be reproducible

        PARAMETERS:
            text (str): The full training text. Every unique character in
                        this string becomes part of the vocabulary.

        EXAMPLE:
            >>> vocab = Vocabulary("hello world")
            >>> vocab.chars
            [' ', 'd', 'e', 'h', 'l', 'o', 'r', 'w']
            >>> vocab.vocab_size
            8
            >>> vocab.char_to_idx
            {' ': 0, 'd': 1, 'e': 2, 'h': 3, 'l': 4, 'o': 5, 'r': 6, 'w': 7}
        """

        # ---- Step A: Find all unique characters ----
        # set(text) removes duplicates. "hello" → {'h', 'e', 'l', 'o'}
        # sorted() puts them in a consistent alphabetical/ASCII order.
        self.chars = sorted(set(text))

        # ---- Step B: Count the vocabulary size ----
        # This number is used later when building the model.
        # The model needs to know: "how many possible characters are there?"
        # For our dataset, vocab_size = 48.
        self.vocab_size = len(self.chars)

        # ---- Step C: Build character → number mapping ----
        # enumerate() gives us (index, character) pairs:
        #   enumerate(['a', 'b', 'c']) → (0, 'a'), (1, 'b'), (2, 'c')
        #
        # We flip it to {character: index} for the dictionary:
        #   {'a': 0, 'b': 1, 'c': 2}
        #
        # This is the ENCODING direction: text → numbers.
        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}

        # ---- Step D: Build number → character mapping ----
        # The reverse: {0: 'a', 1: 'b', 2: 'c'}
        #
        # This is the DECODING direction: numbers → text.
        self.idx_to_char = {idx: char for idx, char in enumerate(self.chars)}

    def encode(self, text):
        """
        Convert a text string into a list of numbers.

        This is the "text → numbers" direction. Every character in the
        input string is looked up in the char_to_idx dictionary and
        replaced with its number.

        PARAMETERS:
            text (str): Any text string to encode. Every character in it
                        must exist in the vocabulary (i.e., must have been
                        in the original training text).

        RETURNS:
            list[int]: A list of integers, one per character.

        EXAMPLE:
            >>> vocab = Vocabulary("hello world")
            >>> vocab.encode("hello")
            [3, 2, 4, 4, 5]

            Breakdown:
                'h' → 3  (looked up in char_to_idx)
                'e' → 2
                'l' → 4
                'l' → 4  (same character, same number)
                'o' → 5

        WHAT HAPPENS IF A CHARACTER IS NOT IN THE VOCABULARY?
            It raises a KeyError. For example, if 'z' wasn't in the
            training text, encode("quiz") would crash on 'z'.
            This is why the training data defines the vocabulary —
            the model literally cannot handle characters it hasn't seen.

        WHY A LIST AND NOT A TENSOR?
            We return a plain Python list for simplicity. In Step 05
            (dataset.py), we'll convert these lists to PyTorch tensors
            right before feeding them to the model.
        """
        return [self.char_to_idx[char] for char in text]

    def decode(self, indices):
        """
        Convert a list of numbers back into a text string.

        This is the "numbers → text" direction — the reverse of encode().
        Used after the model makes predictions (which are numbers) to
        turn them back into readable text.

        PARAMETERS:
            indices (list[int]): A list of integers to decode. Each integer
                                 must be a valid index (0 to vocab_size-1).

        RETURNS:
            str: The decoded text string.

        EXAMPLE:
            >>> vocab = Vocabulary("hello world")
            >>> vocab.decode([3, 2, 4, 4, 5])
            'hello'

            Breakdown:
                3 → 'h'  (looked up in idx_to_char)
                2 → 'e'
                4 → 'l'
                4 → 'l'
                5 → 'o'
            Then join them: 'h' + 'e' + 'l' + 'l' + 'o' = "hello"

        THE FULL ROUND-TRIP:
            text → encode() → numbers → [model processes] → numbers → decode() → text

            "The"  →  [19, 29, 26]  →  [...model...]  →  [1, 27, 42]  →  " fu"
            (This is how the model "reads" and "writes" text!)
        """
        return "".join([self.idx_to_char[idx] for idx in indices])


def demonstrate_encoding(vocab):
    """
    Show how encoding works with step-by-step examples.

    Takes a Vocabulary object and encodes several example strings,
    printing each character alongside its number.

    PARAMETERS:
        vocab (Vocabulary): A built vocabulary to use for encoding.

    OUTPUT: Printed examples of text → number conversion.
    """

    print("=" * 60)
    print("ENCODING: Text → Numbers")
    print("=" * 60)
    print()
    print("This is what happens to your text BEFORE the model sees it.")
    print("The model never sees letters — only these numbers.")
    print()

    # ---- Example 1: A simple word ----
    example1 = "The"
    encoded1 = vocab.encode(example1)

    print(f"  Text:    \"{example1}\"")
    print(f"  Encoded: {encoded1}")
    print()

    # Show the character-by-character mapping
    print("  Step by step:")
    for char in example1:
        idx = vocab.char_to_idx[char]
        print(f"    '{char}' → char_to_idx['{char}'] → {idx}")
    print()

    # ---- Example 2: A full sentence ----
    example2 = "Life is short."
    encoded2 = vocab.encode(example2)

    print(f"  Text:    \"{example2}\"")
    print(f"  Encoded: {encoded2}")
    print()

    # Show that spaces and punctuation are also encoded
    print("  Notice: spaces and punctuation get numbers too!")
    for char in example2:
        idx = vocab.char_to_idx[char]
        print(f"    {repr(char):5s} → {idx}")
    print()

    # ---- Example 3: Show the encoding is deterministic ----
    # Same text always produces the same numbers.
    example3 = "the"
    encoded3a = vocab.encode(example3)
    encoded3b = vocab.encode(example3)

    print(f"  Encoding \"{example3}\" twice:")
    print(f"    First time:  {encoded3a}")
    print(f"    Second time: {encoded3b}")
    print(f"    Same? {encoded3a == encoded3b}  ← encoding is deterministic!")
    print()


def demonstrate_decoding(vocab):
    """
    Show how decoding works — converting numbers back to text.

    Demonstrates the reverse direction: numbers → text.
    Also shows the full round-trip: text → numbers → text.

    PARAMETERS:
        vocab (Vocabulary): A built vocabulary to use for decoding.

    OUTPUT: Printed examples of number → text conversion.
    """

    print("=" * 60)
    print("DECODING: Numbers → Text")
    print("=" * 60)
    print()
    print("This is what happens to the model's output AFTER it predicts.")
    print("The model outputs numbers — we decode them back to text.")
    print()

    # ---- Example 1: Decode a sequence ----
    # These numbers come from encoding "The" earlier
    numbers = [19, 29, 26]
    decoded = vocab.decode(numbers)

    print(f"  Numbers: {numbers}")
    print(f"  Decoded: \"{decoded}\"")
    print()

    print("  Step by step:")
    for idx in numbers:
        char = vocab.idx_to_char[idx]
        print(f"    {idx} → idx_to_char[{idx}] → '{char}'")
    print()

    # ---- Example 2: Full round-trip ----
    original = "Dream big."
    print(f"  ROUND TRIP DEMO:")
    print(f"    Original text  : \"{original}\"")

    encoded = vocab.encode(original)
    print(f"    After encode() : {encoded}")

    decoded = vocab.decode(encoded)
    print(f"    After decode() : \"{decoded}\"")

    print(f"    Match?         : {original == decoded}  ← perfect round-trip!")
    print()
    print("  This proves our encoding loses NO information.")
    print("  encode() and decode() are perfect inverses of each other.")
    print()


def demonstrate_vocabulary_details(vocab):
    """
    Show the complete vocabulary mapping and useful statistics.

    Prints every character and its assigned number, plus insights
    about the vocabulary structure.

    PARAMETERS:
        vocab (Vocabulary): A built vocabulary to inspect.

    OUTPUT: Printed vocabulary table and statistics.
    """

    print("=" * 60)
    print("COMPLETE VOCABULARY MAPPING")
    print("=" * 60)
    print()
    print(f"Vocabulary size: {vocab.vocab_size} unique characters")
    print()

    # Print the mapping in a nice table format
    # We show 4 mappings per line to keep it compact
    print("  Character → Number mappings:")
    print("  " + "-" * 50)

    items = list(vocab.char_to_idx.items())
    for i in range(0, len(items), 4):
        row_items = items[i:i + 4]
        row_str = "  "
        for char, idx in row_items:
            row_str += f"  {repr(char):6s}→ {idx:<4d}"
        print(row_str)
    print()

    # ---- Interesting observations ----
    print("  Observations:")
    print(f"    - Newline ('\\n') is character #{vocab.char_to_idx.get(chr(10), 'N/A')}")
    print(f"    - Space (' ') is character #{vocab.char_to_idx.get(' ', 'N/A')}")
    print(f"    - Lowercase 'a' is #{vocab.char_to_idx.get('a', 'N/A')}, "
          f"uppercase 'A' is #{vocab.char_to_idx.get('A', 'N/A')}")
    print(f"    - The model treats 'A' and 'a' as DIFFERENT characters")
    print(f"    - Numbers go from 0 to {vocab.vocab_size - 1}")
    print()

    # ---- Why this matters for the model ----
    print("  WHY THIS MATTERS:")
    print(f"    The model's output layer will have {vocab.vocab_size} neurons —")
    print(f"    one for each possible character. When predicting the next")
    print(f"    character, it outputs {vocab.vocab_size} probabilities (one per")
    print(f"    character) and picks the most likely one.")
    print()


def main():
    """
    Main function — builds vocabulary from training data and runs all demos.

    FLOW:
        1. Read data/input.txt
        2. Build Vocabulary object (creates all mappings)
        3. Show encoding examples (text → numbers)
        4. Show decoding examples (numbers → text)
        5. Show complete vocabulary mapping
        6. Preview what comes next (Step 05)
    """

    # ---- Load the training text ----
    filepath = "data/input.txt"
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    print("=" * 60)
    print("STEP 04: CHARACTER VOCABULARY (TOKENIZATION)")
    print("=" * 60)
    print()
    print(f"Building vocabulary from: {filepath}")
    print()

    # ---- Build the vocabulary ----
    # This single line does all the work:
    # - Finds unique characters
    # - Sorts them
    # - Creates both mapping dictionaries
    vocab = Vocabulary(text)

    print(f"Vocabulary built successfully!")
    print(f"  Total characters in text : {len(text)}")
    print(f"  Unique characters (vocab): {vocab.vocab_size}")
    print()

    # ---- Run demonstrations ----
    demonstrate_encoding(vocab)
    demonstrate_decoding(vocab)
    demonstrate_vocabulary_details(vocab)

    # ---- What comes next ----
    print("=" * 60)
    print("WHAT COMES NEXT (Step 05)")
    print("=" * 60)
    print("""
Now that we can convert text ↔ numbers, the next step is to
create "training pairs" — sequences where:

  INPUT:  "The only way to do grea"   (a chunk of text)
  TARGET: "he only way to do great"   (the same chunk, shifted by 1)

The model learns: given these input characters, the next character
should be the corresponding target character.

  Input:  T  h  e     o  n  l  y  ...
  Target: h  e     o  n  l  y     ...
          ↑  ↑  ↑  ↑
          For each input char, predict the target char (next one)

This is how the model learns to predict "what comes next".
""")


# ======================================================================
# ENTRY POINT
# ======================================================================
# Run this file directly to see the vocabulary demos:
#   python src/vocabulary.py
#
# Or import the Vocabulary class in other files:
#   from src.vocabulary import Vocabulary
if __name__ == "__main__":
    main()
