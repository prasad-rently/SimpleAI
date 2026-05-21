"""
dataset.py — Turn text into training pairs the model can learn from.

PURPOSE:
    The model learns by seeing examples of "given THESE characters,
    the NEXT character should be THIS." We need to chop our text into
    many such examples.

    This is like making flashcards:
      Front of card (input):   "The only way to do grea"
      Back of card (target):   "he only way to do great"

    The input and target are the SAME text, just shifted by one character.
    For every position, the model must predict the next character:

      Position 0:  input='T'  →  target='h'   (after 'T' comes 'h')
      Position 1:  input='h'  →  target='e'   (after 'h' comes 'e')
      Position 2:  input='e'  →  target=' '   (after 'e' comes ' ')
      ...and so on for every position in the sequence.

THE BIG IDEA:
    ┌────────────────────────────────────────────────┐
    │  Full text: "The only way to do great work..."  │
    └──────────────────┬─────────────────────────────┘
                       │
                       │  (slice into chunks of seq_length characters)
                       ▼
    ┌──────────────────────────┐
    │  Chunk 1: "The only way" │ ← characters 0-11
    │  Chunk 2: " to do great" │ ← characters 12-23
    │  Chunk 3: " work is to " │ ← characters 24-35
    │  ...                     │
    └────────────┬─────────────┘
                 │
                 │  (for each chunk, split into input + target)
                 ▼
    ┌──────────────────────────────────────────┐
    │  input:  "The only wa"  (chars 0 to 10)  │
    │  target: "he only way"  (chars 1 to 11)  │
    │                                          │
    │  input is target shifted LEFT by 1 char  │
    └──────────────────────────────────────────┘

WHAT THIS FILE PROVIDES:
    1. TextDataset class — a PyTorch Dataset that creates these pairs
    2. Demo showing example training pairs with numbers and text

WHY PyTorch Dataset?
    PyTorch has a standard way to handle training data: the Dataset class.
    By following this convention, we get automatic batching, shuffling,
    and parallel data loading for free (via DataLoader in Step 06).

INPUT:  data/input.txt (via Vocabulary from Step 04)
OUTPUT: Training pairs of (input_sequence, target_sequence) as tensors

Usage:
    python src/dataset.py
"""

import torch
from torch.utils.data import Dataset

# Import our Vocabulary class from Step 04.
# This lets us reuse the encode/decode functionality.
from vocabulary import Vocabulary


class TextDataset(Dataset):
    """
    A PyTorch Dataset that converts text into training pairs.

    WHAT IS A PyTorch Dataset?
        It's any class that implements two methods:
        - __len__()        → how many examples do we have?
        - __getitem__(idx) → give me the i-th example

        PyTorch's DataLoader (Step 06) calls these methods to
        automatically batch, shuffle, and serve training data.
        It's like a vending machine: you put in an index, you get
        out a training example.

    HOW IT WORKS:
        1. Take the full training text
        2. Encode it into numbers using the Vocabulary
        3. Convert to a PyTorch tensor (one long list of numbers)
        4. When asked for example 'i':
           - Slice out characters [i*seq_len] to [i*seq_len + seq_len]     → input
           - Slice out characters [i*seq_len + 1] to [i*seq_len + seq_len + 1] → target
           - Return both as tensors

    PARAMETERS:
        text (str):     The full training text
        vocab (Vocabulary): A built vocabulary for encoding
        seq_length (int):   How many characters per training example
                            (default: 50). Longer = more context for
                            the model, but slower training.

    EXAMPLE:
        If text = "The only way" and seq_length = 5:
            Example 0: input="The o"  target="he on"
            Example 1: input="nly w"  target="ly wa"
            Example 2: input="ay..."  (and so on)

    HOW IT'S USED IN THE PROJECT:
        1. Step 05 (this file): Create the dataset
        2. Step 06 (also this file): Wrap in DataLoader for batching
        3. Step 09 (train.py): DataLoader feeds batches to the training loop
    """

    def __init__(self, text, vocab, seq_length=50):
        """
        Initialize the dataset by encoding the full text.

        WHAT HAPPENS HERE:
            1. Save the vocab and seq_length for later use
            2. Encode the ENTIRE text into a list of numbers
            3. Convert that list into a PyTorch tensor

        WHY ENCODE EVERYTHING UPFRONT?
            It's much faster to encode once and then slice, rather
            than encoding small pieces every time __getitem__ is called.
            Since __getitem__ is called thousands of times during
            training, this optimization matters.

        PARAMETERS:
            text (str):         The full training text to learn from
            vocab (Vocabulary): The vocabulary for encoding characters
            seq_length (int):   Characters per training example (default: 50)

        EXAMPLE:
            >>> vocab = Vocabulary("hello world")
            >>> dataset = TextDataset("hello world", vocab, seq_length=5)
            >>> len(dataset)
            1
            >>> dataset.data
            tensor([3, 2, 4, 4, 5, 0, 7, 5, 6, 4, 1])
            #        h  e  l  l  o     w  o  r  l  d
        """

        # Save references for use in __getitem__ and __len__
        self.vocab = vocab
        self.seq_length = seq_length

        # ---- Encode the entire text into numbers ----
        # vocab.encode("hello") → [3, 2, 4, 4, 5]
        # This converts every character in the training text to its number.
        encoded = vocab.encode(text)

        # ---- Convert to a PyTorch tensor ----
        # torch.tensor() wraps the Python list in a PyTorch tensor.
        # dtype=torch.long means 64-bit integers (standard for indices).
        #
        # WHY torch.long?
        #   The model's embedding layer (Step 07) expects integer indices.
        #   torch.long = torch.int64 = the default integer type in PyTorch.
        #
        # EXAMPLE:
        #   [3, 2, 4, 4, 5] → tensor([3, 2, 4, 4, 5])
        #   Now PyTorch can do fast math on this data.
        self.data = torch.tensor(encoded, dtype=torch.long)

        # ---- Calculate how many complete examples we can make ----
        # We need (seq_length + 1) characters per example because:
        #   - seq_length characters for the INPUT
        #   - 1 more character to get the LAST target
        #
        # EXAMPLE with seq_length=5, text="hello world" (11 chars):
        #   We need 6 chars per example (5 input + 1 for final target)
        #   11 chars ÷ 6 = 1 complete example (with 5 chars left over)
        #
        # The leftover characters at the end are simply discarded.
        # For our dataset (6201 chars, seq_length=50):
        #   6201 ÷ 51 = 121 complete training examples
        self.num_examples = (len(self.data) - 1) // seq_length

    def __len__(self):
        """
        Return the total number of training examples.

        PyTorch's DataLoader calls this to know how many examples exist.
        This determines:
        - How many batches per epoch
        - When an epoch is "complete"

        RETURNS:
            int: Number of complete (input, target) pairs.

        EXAMPLE:
            >>> len(dataset)
            121  (for our 6201-char text with seq_length=50)
        """
        return self.num_examples

    def __getitem__(self, idx):
        """
        Return the i-th training example as (input, target) tensors.

        This is the heart of the dataset. PyTorch's DataLoader calls
        this method with different idx values to build batches.

        HOW THE SLICING WORKS:
            start = idx * seq_length

            input  = data[start : start + seq_length]       ← seq_length chars
            target = data[start + 1 : start + seq_length + 1] ← shifted by 1

        VISUAL EXAMPLE (seq_length=5, text="The only way"):
            data = [19, 29, 26, 1, 36, 35, 33, 46, 1, 44, 22, 46]
                    T   h   e      o   n   l   y      w   a   y

            idx=0, start=0:
              input  = data[0:5]  = [19, 29, 26,  1, 36]  → "The o"
              target = data[1:6]  = [29, 26,  1, 36, 35]  → "he on"
                                     ↑
                                     For position 0: input='T'(19), target='h'(29)
                                     The model should learn: after 'T', predict 'h'

            idx=1, start=5:
              input  = data[5:10] = [35, 33, 46,  1, 44]  → "nly w"
              target = data[6:11] = [33, 46,  1, 44, 22]  → "ly wa"

        PARAMETERS:
            idx (int): Index of the training example (0 to len-1)

        RETURNS:
            tuple[Tensor, Tensor]: (input_sequence, target_sequence)
            Both tensors have shape (seq_length,) and dtype torch.long.

        EXAMPLE:
            >>> input_seq, target_seq = dataset[0]
            >>> input_seq.shape
            torch.Size([50])
            >>> target_seq.shape
            torch.Size([50])
        """

        # Calculate where this example starts in the encoded data
        start = idx * self.seq_length

        # Slice out input and target sequences
        # input:  positions [start] to [start + seq_length - 1]
        # target: positions [start + 1] to [start + seq_length]
        # The target is the input shifted RIGHT by one position.
        input_seq = self.data[start:start + self.seq_length]
        target_seq = self.data[start + 1:start + self.seq_length + 1]

        return input_seq, target_seq


def demonstrate_training_pairs(dataset, vocab, num_examples=3):
    """
    Show what training pairs look like — both as numbers and text.

    This function helps you understand the fundamental pattern that
    the model will learn from: for each input character, predict the
    next character (the target).

    PARAMETERS:
        dataset (TextDataset): The built dataset to pull examples from
        vocab (Vocabulary):    The vocabulary for decoding numbers back to text
        num_examples (int):    How many examples to display (default: 3)

    OUTPUT: Printed examples showing input/target alignment.
    """

    print("=" * 60)
    print("TRAINING PAIRS: What the model learns from")
    print("=" * 60)
    print()
    print(f"Sequence length : {dataset.seq_length} characters")
    print(f"Total examples  : {len(dataset)}")
    print()

    for i in range(min(num_examples, len(dataset))):
        input_seq, target_seq = dataset[i]

        # Decode numbers back to text for readability
        input_text = vocab.decode(input_seq.tolist())
        target_text = vocab.decode(target_seq.tolist())

        print(f"--- Example {i} ---")
        print(f"  Input  (text): \"{input_text}\"")
        print(f"  Target (text): \"{target_text}\"")
        print()

        # Show first few positions in detail
        print("  Character-by-character (first 8 positions):")
        print(f"  {'Position':<10} {'Input char':<12} {'Target char':<12} {'Meaning'}")
        print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*30}")

        for pos in range(min(8, dataset.seq_length)):
            in_char = vocab.idx_to_char[input_seq[pos].item()]
            tgt_char = vocab.idx_to_char[target_seq[pos].item()]
            print(f"  {pos:<10} {repr(in_char):<12} {repr(tgt_char):<12} "
                  f"after {repr(in_char)}, predict {repr(tgt_char)}")

        print(f"  ... ({dataset.seq_length - 8} more positions)")
        print()


def demonstrate_data_shapes(dataset):
    """
    Show the tensor shapes and data types — what PyTorch actually sees.

    Understanding shapes is crucial because shape mismatches are the
    #1 cause of bugs in neural network code. This demo helps you
    build intuition for what each dimension means.

    PARAMETERS:
        dataset (TextDataset): The built dataset to inspect

    OUTPUT: Printed shape and dtype information for the raw data
            and individual training examples.
    """

    print("=" * 60)
    print("DATA SHAPES: What PyTorch sees")
    print("=" * 60)
    print()

    # ---- The full encoded dataset ----
    print("Full encoded dataset:")
    print(f"  Shape: {dataset.data.shape}")
    # Shape: (6201,) — one big 1D tensor with all characters as numbers
    print(f"  dtype: {dataset.data.dtype}")
    # dtype: torch.int64 — 64-bit integers (needed for embedding lookup)
    print(f"  First 20 values: {dataset.data[:20].tolist()}")
    print()

    # ---- A single training example ----
    input_seq, target_seq = dataset[0]

    print("Single training example (dataset[0]):")
    print(f"  input_seq  shape: {input_seq.shape}  dtype: {input_seq.dtype}")
    # Shape: (50,) — a 1D tensor of 50 integers
    print(f"  target_seq shape: {target_seq.shape}  dtype: {target_seq.dtype}")
    # Shape: (50,) — a 1D tensor of 50 integers (shifted by 1)
    print()

    # ---- Verify the shift ----
    # The target should be the input shifted by one position.
    # So target[0] should equal the character AFTER input[0] in the original text.
    print("Verify the shift (target = input shifted by 1):")
    print(f"  input  values [0:5]: {input_seq[:5].tolist()}")
    print(f"  target values [0:5]: {target_seq[:5].tolist()}")
    print(f"  data   values [0:6]: {dataset.data[:6].tolist()}")
    print()
    print("  Notice: target[0]=data[1], target[1]=data[2], etc.")
    print("  The target IS the input, shifted one position to the right.")
    print()

    # ---- Summary ----
    print("  DIMENSIONS CHEAT SHEET:")
    print(f"  ┌─────────────────────────────────────────────────┐")
    print(f"  │ Full data:  ({len(dataset.data)},)          "
          f"← all chars as numbers │")
    print(f"  │ One input:  ({dataset.seq_length},)            "
          f"← {dataset.seq_length} numbers (one example)  │")
    print(f"  │ One target: ({dataset.seq_length},)            "
          f"← {dataset.seq_length} numbers (shifted by 1) │")
    print(f"  │ Num examples: {len(dataset)}                "
          f"← total training pairs     │")
    print(f"  └─────────────────────────────────────────────────┘")
    print()


def main():
    """
    Main function — builds the dataset and runs all demonstrations.

    FLOW:
        1. Load data/input.txt
        2. Build Vocabulary (from Step 04)
        3. Create TextDataset (chop text into training pairs)
        4. Show example training pairs (text + numbers)
        5. Show tensor shapes (what PyTorch sees)
        6. Preview what comes next (Step 06: batching)
    """

    # ---- Load the training text ----
    filepath = "data/input.txt"
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # ---- Build vocabulary (reusing Step 04) ----
    vocab = Vocabulary(text)

    print("=" * 60)
    print("STEP 05: TRAINING SEQUENCES")
    print("=" * 60)
    print()
    print(f"Text file       : {filepath}")
    print(f"Total characters: {len(text)}")
    print(f"Vocabulary size : {vocab.vocab_size}")
    print()

    # ---- Create the dataset ----
    # seq_length=50 means each training example is 50 characters long.
    # The model sees 50 characters of context and tries to predict
    # the next character for each position.
    #
    # WHY 50?
    #   - Long enough to capture patterns like "The only way to"
    #   - Short enough to keep training fast
    #   - Real LLMs use seq_lengths of 2048, 4096, or even 128K!
    seq_length = 50
    dataset = TextDataset(text, vocab, seq_length=seq_length)

    print(f"Sequence length : {seq_length} characters per example")
    print(f"Total examples  : {len(dataset)}")
    print(f"  ({len(text)} chars ÷ {seq_length + 1} chars per example "
          f"= {len(dataset)} examples)")
    print()

    # ---- Run demonstrations ----
    demonstrate_training_pairs(dataset, vocab, num_examples=3)
    demonstrate_data_shapes(dataset)

    # ---- What comes next ----
    print("=" * 60)
    print("WHAT COMES NEXT (Step 06)")
    print("=" * 60)
    print("""
Right now, the model would see examples ONE AT A TIME:
  dataset[0] → first example
  dataset[1] → second example
  ...

This is slow! In Step 06, we'll use a DataLoader to group
examples into BATCHES:

  Batch 1: [example 0, example 1, example 2, example 3]  ← 4 at once
  Batch 2: [example 4, example 5, example 6, example 7]
  ...

Processing examples in batches is MUCH faster because:
  - GPUs/CPUs can do math on many examples simultaneously
  - The model updates its weights once per batch, not once per example
  - It's more stable (averages out noise from individual examples)
""")


# ======================================================================
# ENTRY POINT
# ======================================================================
# Run this file directly to see the training pair demos:
#   python src/dataset.py
#
# Or import the TextDataset class in other files:
#   from src.dataset import TextDataset
#
# NOTE: This script must be run from the SimpleAI/ root directory,
# and you must run it as: python src/dataset.py
# (so that "data/input.txt" path and "from vocabulary import" work)
if __name__ == "__main__":
    main()
