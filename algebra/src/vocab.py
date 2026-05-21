"""
vocab.py — Character-level vocabulary with special tokens for seq2seq.

PURPOSE:
    Converts characters to numbers and back, just like in the text
    generator — but with three important additions:

    SPECIAL TOKENS:
        <PAD> (index 0) — Padding. Fills empty space when batching
                          sequences of different lengths together.
                          "x = 2<PAD><PAD>" makes a short answer the
                          same length as a longer one in the same batch.

        <SOS> (index 1) — Start Of Sequence. Tells the decoder "begin
                          generating now." Always the first input to
                          the decoder. The decoder never predicts this
                          token — it only receives it.

        <EOS> (index 2) — End Of Sequence. Tells the decoder "stop
                          generating." Appended to every target during
                          training. During inference, when the decoder
                          predicts <EOS>, we stop.

    WHY DO WE NEED THESE?
        The text generator didn't need them because:
        - It always generated a fixed number of characters (no stop signal)
        - It processed one sequence at a time (no padding needed)
        - It continued from the seed (no start signal needed)

        The algebra solver needs them because:
        - Answers have different lengths ("x = 2" vs "x = -35")
        - The decoder needs to know when to start and stop
        - Batching requires equal-length sequences

HOW IT DIFFERS FROM THE TEXT GENERATOR'S VOCABULARY:
    Text Generator:
        chars = ['\\n', ' ', ',', '.', ... 'z']  (48 chars)
        No special tokens. No padding.

    Algebra Solver:
        tokens = ['<PAD>', '<SOS>', '<EOS>', ' ', '+', '-', ... 'x']
        3 special tokens + 16 data characters = 19 tokens total.

WHAT THIS FILE PROVIDES:
    1. AlgebraVocab class — vocabulary with special tokens
       - encode(text)     — "2x + 3 = 7" → [5, 16, 3, 8, 3, 6, 3, 12, 3, 10]
       - decode(indices)  — [16, 3, 12, 3, 5] → "x = 2"
       - encode_with_eos(text) — adds <EOS> at the end
       - vocab_size       — total number of tokens (19)
    2. build_vocab_from_data() — scan data file to find all characters

INPUT:  algebra/data/equations.txt (to discover character set)
OUTPUT: Vocabulary object (used by dataset and model)

Usage:
    from vocab import AlgebraVocab, build_vocab_from_data
    vocab = build_vocab_from_data("algebra/data/equations.txt")
    encoded = vocab.encode("2x + 3 = 7")
"""


# ====================================================================
# SPECIAL TOKEN CONSTANTS
# ====================================================================
# These are defined at module level so any file can import them.
# Using constants prevents typos ("PAD_IDX" vs magic number 0).

PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2


class AlgebraVocab:
    """
    Character-level vocabulary with special tokens for seq2seq models.

    This maps each character (and special token) to a unique integer
    index, and back. The model works with indices, not characters.

    SPECIAL TOKENS ARE ALWAYS AT INDICES 0, 1, 2:
        0 = <PAD>  — padding (ignored during loss calculation)
        1 = <SOS>  — start of decoder input
        2 = <EOS>  — end of target sequence

    Data characters start at index 3.

    ATTRIBUTES:
        token_to_idx (dict): Maps token/char → integer index
        idx_to_token (dict): Maps integer index → token/char
        tokens (list):       All tokens in order [<PAD>, <SOS>, <EOS>, ' ', '+', ...]
        vocab_size (int):    Total count of tokens
    """

    def __init__(self, characters):
        """
        Build the vocabulary from a list of data characters.

        The special tokens are always prepended at positions 0, 1, 2.
        Data characters are sorted and appended starting at position 3.
        Sorting ensures the vocabulary is deterministic — same characters
        always produce the same mapping, regardless of input order.

        PARAMETERS:
            characters (list or set): The data characters to include
                                       (e.g., [' ', '+', '-', '0', ..., 'x'])
        """

        # ---- Sort data characters for deterministic ordering ----
        sorted_chars = sorted(set(characters))

        # ---- Build token list: special tokens first, then data ----
        self.tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN] + sorted_chars

        # ---- Build bidirectional mappings ----
        self.token_to_idx = {token: idx for idx, token in enumerate(self.tokens)}
        self.idx_to_token = {idx: token for idx, token in enumerate(self.tokens)}

        # ---- Store size ----
        self.vocab_size = len(self.tokens)

    def encode(self, text):
        """
        Convert a string to a list of integer indices.

        Each character in the text is looked up in token_to_idx.
        Special tokens (<PAD>, <SOS>, <EOS>) are NOT added — this
        method only converts the raw text characters.

        PARAMETERS:
            text (str): The text to encode, e.g., "2x + 3 = 7"

        RETURNS:
            list[int]: Integer indices, e.g., [5, 16, 3, 8, 3, 6, 3, 12, 3, 10]

        RAISES:
            KeyError: If the text contains a character not in the vocabulary
        """

        return [self.token_to_idx[ch] for ch in text]

    def encode_with_eos(self, text):
        """
        Convert a string to indices and append <EOS> at the end.

        Used for target sequences during training. The <EOS> token
        teaches the decoder when to stop generating.

        EXAMPLE:
            "x = 2" → [16, 3, 12, 3, 5, 2]
                                          ^-- EOS_IDX = 2

        PARAMETERS:
            text (str): The text to encode

        RETURNS:
            list[int]: Integer indices ending with EOS_IDX
        """

        return self.encode(text) + [EOS_IDX]

    def decode(self, indices):
        """
        Convert a list of integer indices back to a string.

        Special tokens (<PAD>, <SOS>, <EOS>) are skipped — they
        are control signals, not real characters to display.

        PARAMETERS:
            indices (list[int]): The indices to decode

        RETURNS:
            str: The decoded text with special tokens removed
        """

        result = []
        for idx in indices:
            token = self.idx_to_token[idx]
            # ---- Skip special tokens ----
            if token in (PAD_TOKEN, SOS_TOKEN, EOS_TOKEN):
                continue
            result.append(token)
        return ''.join(result)

    def decode_until_eos(self, indices):
        """
        Decode indices, stopping at the first <EOS> token.

        During inference, the decoder generates tokens one at a time.
        When it predicts <EOS>, everything after that is garbage
        (the model wasn't "thinking" past the end signal). So we
        stop decoding at the first <EOS>.

        EXAMPLE:
            [16, 3, 12, 3, 5, 2, 9, 9, 0]  → "x = 2"
             x   =     2  EOS ← stop here, ignore 9, 9, 0

        PARAMETERS:
            indices (list[int]): The indices to decode

        RETURNS:
            str: The decoded text up to (not including) <EOS>
        """

        result = []
        for idx in indices:
            if idx == EOS_IDX:
                break
            token = self.idx_to_token[idx]
            if token in (PAD_TOKEN, SOS_TOKEN):
                continue
            result.append(token)
        return ''.join(result)

    def __len__(self):
        """Return the vocabulary size (for convenience)."""
        return self.vocab_size

    def __repr__(self):
        """Readable string representation for debugging."""
        return (f"AlgebraVocab(size={self.vocab_size}, "
                f"tokens={self.tokens})")


def build_vocab_from_data(filepath="algebra/data/equations.txt"):
    """
    Scan the data file and build a vocabulary from all characters found.

    This ensures the vocabulary covers exactly the characters that
    appear in the data — no more, no less.

    PROCESS:
        1. Read every line from the data file
        2. Collect all unique characters from equations and solutions
        3. Create an AlgebraVocab with those characters

    WHY SCAN THE DATA?
        We could hardcode the character set, but scanning is safer:
        - If we add new equation types with new characters, the vocab
          automatically includes them
        - We can verify that no unexpected characters snuck in

    PARAMETERS:
        filepath (str): Path to the equations file

    RETURNS:
        AlgebraVocab: Vocabulary built from the data
    """

    all_chars = set()

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # ---- Split on tab to get equation and solution ----
            parts = line.split('\t')
            equation = parts[0]
            solution = parts[1]
            # ---- Collect characters from both ----
            all_chars.update(equation)
            all_chars.update(solution)

    return AlgebraVocab(all_chars)


# ====================================================================
# MAIN — demo the vocabulary
# ====================================================================

def main():
    """
    Demo the vocabulary: build it, show the mappings, encode/decode examples.
    """

    print("=" * 60)
    print("STEP 03: VOCABULARY WITH SPECIAL TOKENS")
    print("=" * 60)
    print()

    # ---- Build vocabulary from data ----
    vocab = build_vocab_from_data("algebra/data/equations.txt")

    print(f"  Vocabulary size: {vocab.vocab_size} tokens")
    print()

    # ---- Show all tokens and their indices ----
    print("  Token → Index mapping:")
    print("  " + "-" * 40)
    for idx, token in enumerate(vocab.tokens):
        if token in (PAD_TOKEN, SOS_TOKEN, EOS_TOKEN):
            print(f"    {token:<8} → {idx}  (special)")
        else:
            print(f"    \"{token}\"      → {idx}")

    # ---- Encoding examples ----
    print()
    print("  Encoding examples:")
    print("  " + "-" * 40)

    examples = [
        "2x + 3 = 7",
        "x = 2",
        "-5x = 15",
        "x / 4 = -3",
        "3x + 2 = x + 8",
    ]

    for text in examples:
        encoded = vocab.encode(text)
        print(f"    \"{text}\"")
        print(f"      → {encoded}")
        print()

    # ---- Encode with EOS ----
    print("  Encoding with <EOS> (for target sequences):")
    print("  " + "-" * 40)
    for sol in ["x = 2", "x = -35", "x = 0"]:
        encoded = vocab.encode_with_eos(sol)
        print(f"    \"{sol}\"  → {encoded}")
        print(f"      last index {encoded[-1]} = <EOS> (stop signal)")

    # ---- Roundtrip test ----
    print()
    print("  Roundtrip test (encode → decode):")
    print("  " + "-" * 40)
    for text in examples:
        encoded = vocab.encode(text)
        decoded = vocab.decode(encoded)
        match = "✓" if decoded == text else "✗"
        print(f"    \"{text}\" → {encoded} → \"{decoded}\" {match}")

    # ---- Decode with EOS stopping ----
    print()
    print("  Decode until <EOS> (simulating inference):")
    print("  " + "-" * 40)
    test_indices = vocab.encode("x = 2") + [EOS_IDX, 9, 9, 0]
    print(f"    Indices: {test_indices}")
    print(f"    Full decode:      \"{vocab.decode(test_indices)}\"")
    print(f"    Decode until EOS: \"{vocab.decode_until_eos(test_indices)}\"")
    print(f"    (Stops at <EOS>, ignoring garbage after it)")

    print()
    print("  " + "=" * 40)
    print("  Vocabulary ready!")


if __name__ == "__main__":
    main()
