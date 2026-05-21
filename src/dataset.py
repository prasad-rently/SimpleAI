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
    2. create_dataloader() — wraps the dataset in a DataLoader for batching
    3. Demos showing training pairs, batch shapes, and shuffling

WHY PyTorch Dataset + DataLoader?
    PyTorch has a two-part system for handling training data:
    - Dataset:    stores the data and serves individual examples
    - DataLoader: groups examples into batches, shuffles them, etc.

    By following this convention, we get automatic batching, shuffling,
    and parallel data loading — the standard approach in all PyTorch projects.

    ANALOGY:
      Dataset    = a deck of flashcards (one card = one example)
      DataLoader = a study system that shuffles the deck and hands you
                   groups of 4 cards at a time (one batch = 4 cards)

INPUT:  data/input.txt (via Vocabulary from Step 04)
OUTPUT: Batched training pairs ready for the model to consume

Usage:
    python src/dataset.py
"""

import torch
from torch.utils.data import Dataset, DataLoader

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


def create_dataloader(dataset, batch_size=16, shuffle=True):
    """
    Wrap a TextDataset in a PyTorch DataLoader for batched training.

    WHAT IS A DataLoader?
        A DataLoader takes a Dataset and adds superpowers:
        1. BATCHING  — groups examples together (e.g., 16 at a time)
        2. SHUFFLING — randomizes the order each epoch
        3. PARALLEL  — can load data in background threads

        Think of it as a conveyor belt in a factory:
        - The Dataset is the warehouse of parts (individual examples)
        - The DataLoader is the conveyor belt that delivers parts to
          the assembly line (the model) in organized groups (batches)

    WHY BATCHES?
        Processing one example at a time is like washing dishes one by
        one under the tap. Batching is like loading the dishwasher —
        you process many at once, which is MUCH more efficient.

        Technical reasons:
        1. HARDWARE EFFICIENCY: CPUs and GPUs are designed to do math
           on many numbers simultaneously (SIMD/parallel processing).
           Batching lets us use this capability.
        2. STABLE GRADIENTS: Averaging the loss over a batch gives a
           smoother learning signal than any single example.
        3. SPEED: One batch of 16 is faster than 16 individual passes
           because of reduced overhead (function calls, memory access).

    WHY SHUFFLE?
        If the model always sees examples in the same order, it might
        "memorize" the order instead of learning general patterns.
        Shuffling ensures each epoch presents examples in a different
        random order, which leads to better learning.

        EXAMPLE without shuffle (same order every epoch):
          Epoch 1: [ex0, ex1, ex2, ex3, ex4, ...]
          Epoch 2: [ex0, ex1, ex2, ex3, ex4, ...]  ← same!

        With shuffle (random order each epoch):
          Epoch 1: [ex3, ex0, ex4, ex1, ex2, ...]
          Epoch 2: [ex1, ex4, ex2, ex0, ex3, ...]  ← different!

    PARAMETERS:
        dataset (TextDataset): The dataset to batch
        batch_size (int):      How many examples per batch (default: 16)
                               Common values: 8, 16, 32, 64
                               Larger = faster but uses more memory
        shuffle (bool):        Whether to randomize order (default: True)
                               Always True for training, False for evaluation

    RETURNS:
        DataLoader: An iterable that yields batches of (input, target) tensors.
                    Each batch has shape (batch_size, seq_length).

    EXAMPLE:
        >>> loader = create_dataloader(dataset, batch_size=4)
        >>> for batch_inputs, batch_targets in loader:
        ...     print(batch_inputs.shape)  # torch.Size([4, 50])
        ...     break
        >>>
        >>> # 4 examples per batch, each 50 chars long
        >>> # Total batches: 124 examples ÷ 4 per batch = 31 batches

    HOW IT'S USED IN THE PROJECT:
        In train.py (Step 09), the training loop iterates over the DataLoader:

        for epoch in range(num_epochs):
            for batch_inputs, batch_targets in dataloader:
                # Process the batch through the model
                # Compute loss, backpropagate, update weights
                pass
    """

    # ---- Create the DataLoader ----
    # DataLoader wraps our Dataset and handles batching + shuffling.
    #
    # drop_last=True: If the last batch has fewer than batch_size examples
    # (because the total doesn't divide evenly), drop it. This keeps all
    # batches the same size, which simplifies the training code.
    #
    # EXAMPLE: 124 examples ÷ 16 per batch = 7 full batches + 12 leftover
    #   drop_last=True:  7 batches (the 12 leftovers are dropped)
    #   drop_last=False: 8 batches (last batch has only 12 examples)
    #
    # Dropping the last partial batch is standard practice because:
    # - Some model operations expect a fixed batch size
    # - The dropped examples still get used when shuffle=True
    #   (they'll likely be in a different position next epoch)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
    )

    return dataloader


def demonstrate_batching(dataloader, vocab, dataset):
    """
    Show how the DataLoader groups examples into batches.

    Visualizes the shape transformation from individual examples
    to stacked batches, and shows what a batch looks like in both
    numbers and text.

    PARAMETERS:
        dataloader (DataLoader): The DataLoader to demonstrate
        vocab (Vocabulary):      For decoding numbers back to text
        dataset (TextDataset):   For reference info (seq_length, etc.)

    OUTPUT: Printed batch shapes, contents, and explanation.
    """

    print("=" * 60)
    print("BATCHING: Grouping examples for efficient training")
    print("=" * 60)
    print()

    # ---- Batch configuration ----
    batch_size = dataloader.batch_size
    num_batches = len(dataloader)

    print(f"  Batch size     : {batch_size} examples per batch")
    print(f"  Total examples : {len(dataset)}")
    print(f"  Total batches  : {num_batches}")
    print(f"  Examples used  : {num_batches * batch_size} "
          f"({len(dataset) - num_batches * batch_size} dropped from last partial batch)")
    print()

    # ---- Shape transformation ----
    # Before batching (individual examples):
    #   input:  shape (50,)     ← one sequence of 50 characters
    #   target: shape (50,)
    #
    # After batching (grouped examples):
    #   input:  shape (16, 50)  ← 16 sequences of 50 characters each
    #   target: shape (16, 50)
    #
    # The new first dimension is the BATCH dimension.
    # batch_inputs[0] is the first example, batch_inputs[1] is the second, etc.

    print("  SHAPE TRANSFORMATION:")
    print(f"  ┌─────────────────────────────────────────────────────┐")
    print(f"  │ Before batching (one example):                      │")
    print(f"  │   input  shape: ({dataset.seq_length},)             │")
    print(f"  │   target shape: ({dataset.seq_length},)             │")
    print(f"  │                                                     │")
    print(f"  │ After batching ({batch_size} examples stacked):     │")
    print(f"  │   input  shape: ({batch_size}, {dataset.seq_length})"
          f"                    │")
    print(f"  │   target shape: ({batch_size}, {dataset.seq_length})"
          f"                    │")
    print(f"  │                  ↑    ↑                             │")
    print(f"  │            batch_dim  seq_dim                       │")
    print(f"  └─────────────────────────────────────────────────────┘")
    print()

    # ---- Grab and inspect the first batch ----
    # iter() creates an iterator from the DataLoader.
    # next() gets the first item from that iterator.
    # This is equivalent to doing one step of a for-loop.
    batch_inputs, batch_targets = next(iter(dataloader))

    print(f"  First batch (actual tensors):")
    print(f"    batch_inputs  shape: {batch_inputs.shape}  dtype: {batch_inputs.dtype}")
    print(f"    batch_targets shape: {batch_targets.shape}  dtype: {batch_targets.dtype}")
    print()

    # ---- Show a few examples from the batch ----
    # Decode the first 3 examples to show what text is in this batch.
    print(f"  First 3 examples in this batch:")
    for i in range(min(3, batch_size)):
        input_text = vocab.decode(batch_inputs[i].tolist())
        # Show first 40 chars for readability
        preview = input_text[:40] + "..." if len(input_text) > 40 else input_text
        print(f"    [{i}] \"{preview}\"")
    print()

    # ---- Explain how indexing works ----
    print("  HOW TO ACCESS BATCH DATA:")
    print(f"    batch_inputs[0]    → first example in batch   shape: ({dataset.seq_length},)")
    print(f"    batch_inputs[0][0] → first char of first ex   value: {batch_inputs[0][0].item()}"
          f" = '{vocab.idx_to_char[batch_inputs[0][0].item()]}'")
    print(f"    batch_inputs[:, 0] → first char of ALL examples  shape: ({batch_size},)")
    print()


def demonstrate_shuffling(dataset, vocab):
    """
    Show that shuffling randomizes the batch order each epoch.

    Creates two DataLoaders from the same dataset — one shuffled,
    one not — and compares the first batch from each.

    PARAMETERS:
        dataset (TextDataset): The dataset to demonstrate with
        vocab (Vocabulary):    For decoding numbers back to text

    OUTPUT: Printed comparison showing shuffled vs unshuffled batches.
    """

    print("=" * 60)
    print("SHUFFLING: Why random order helps learning")
    print("=" * 60)
    print()

    # ---- Without shuffling ----
    # Same order every time — the first batch always contains the
    # same examples (the first 4 in the dataset).
    loader_no_shuffle = DataLoader(dataset, batch_size=4, shuffle=False)

    batch1_inputs, _ = next(iter(loader_no_shuffle))
    batch1_again, _ = next(iter(loader_no_shuffle))

    text1 = vocab.decode(batch1_inputs[0].tolist())[:40]
    text1_again = vocab.decode(batch1_again[0].tolist())[:40]

    print("  WITHOUT shuffle (shuffle=False):")
    print(f"    First call, batch[0]: \"{text1}...\"")
    print(f"    Second call, batch[0]: \"{text1_again}...\"")
    print(f"    Same? {text1 == text1_again} ← always the same order")
    print()

    # ---- With shuffling ----
    # Different order each time — the first batch will contain
    # different examples on each pass.
    loader_shuffled_1 = DataLoader(dataset, batch_size=4, shuffle=True)
    loader_shuffled_2 = DataLoader(dataset, batch_size=4, shuffle=True)

    batch_s1, _ = next(iter(loader_shuffled_1))
    batch_s2, _ = next(iter(loader_shuffled_2))

    text_s1 = vocab.decode(batch_s1[0].tolist())[:40]
    text_s2 = vocab.decode(batch_s2[0].tolist())[:40]

    print("  WITH shuffle (shuffle=True):")
    print(f"    First loader, batch[0]:  \"{text_s1}...\"")
    print(f"    Second loader, batch[0]: \"{text_s2}...\"")
    print(f"    Same? {text_s1 == text_s2} ← different order (usually)")
    print()
    print("  Shuffling prevents the model from memorizing the ORDER of")
    print("  examples. It must learn PATTERNS, not sequences of batches.")
    print()


def demonstrate_full_epoch(dataloader):
    """
    Iterate through ALL batches to show what one full epoch looks like.

    In training, one "epoch" = processing every batch exactly once.
    This demo counts the batches and total examples seen.

    PARAMETERS:
        dataloader (DataLoader): The DataLoader to iterate through

    OUTPUT: Printed batch count and epoch summary.
    """

    print("=" * 60)
    print("ONE FULL EPOCH: Iterating through all batches")
    print("=" * 60)
    print()

    # ---- Iterate through all batches ----
    # This is EXACTLY what the training loop does (Step 09).
    # The only difference is that during training, we also compute
    # the loss and update the model's weights for each batch.

    total_examples_seen = 0
    batch_count = 0

    for batch_idx, (batch_inputs, batch_targets) in enumerate(dataloader):
        batch_count += 1
        total_examples_seen += batch_inputs.shape[0]

        # Show the first 3 and last batch
        if batch_idx < 3 or batch_idx == len(dataloader) - 1:
            print(f"  Batch {batch_idx:3d}: "
                  f"input shape {tuple(batch_inputs.shape)}, "
                  f"target shape {tuple(batch_targets.shape)}")

        if batch_idx == 3:
            print(f"  ... ({len(dataloader) - 4} more batches) ...")

    print()
    print(f"  Epoch complete!")
    print(f"    Batches processed   : {batch_count}")
    print(f"    Examples seen       : {total_examples_seen}")
    print(f"    Batch size          : {dataloader.batch_size}")
    print()

    # ---- This is the training loop skeleton ----
    print("  IN TRAINING (Step 09), each batch will be processed like:")
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │  for batch_inputs, batch_targets in dataloader:     │")
    print("  │      predictions = model(batch_inputs)   # forward  │")
    print("  │      loss = loss_fn(predictions, targets) # score   │")
    print("  │      loss.backward()                      # grads   │")
    print("  │      optimizer.step()                     # update  │")
    print("  └─────────────────────────────────────────────────────┘")
    print()


def main():
    """
    Main function — builds dataset + dataloader and runs all demonstrations.

    FLOW:
        1. Load data/input.txt
        2. Build Vocabulary (from Step 04)
        3. Create TextDataset (chop text into training pairs)
        4. Show example training pairs (text + numbers)
        5. Show tensor shapes (what PyTorch sees)
        6. Create DataLoader (group examples into batches)
        7. Show batch shapes and contents
        8. Show shuffling effect
        9. Walk through one full epoch
    """

    # ---- Load the training text ----
    filepath = "data/input.txt"
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # ---- Build vocabulary (reusing Step 04) ----
    vocab = Vocabulary(text)

    print("=" * 60)
    print("STEP 05-06: TRAINING SEQUENCES + BATCHING")
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

    # ---- Run Step 05 demonstrations ----
    demonstrate_training_pairs(dataset, vocab, num_examples=2)
    demonstrate_data_shapes(dataset)

    # ---- Create the DataLoader (Step 06) ----
    # batch_size=16 means the model sees 16 examples at once.
    #
    # WHY 16?
    #   - Small enough to fit in memory on any machine
    #   - Large enough to give stable gradient estimates
    #   - Common starting point; can be tuned later
    #
    # COMMON BATCH SIZES AND TRADE-OFFS:
    #   batch_size=1:   Very noisy gradients, slow training
    #   batch_size=16:  Good balance of speed and stability (our choice)
    #   batch_size=32:  Faster, slightly less noise
    #   batch_size=128: Very fast, but needs more memory
    #   batch_size=4096: What large companies use with big GPUs
    batch_size = 16
    dataloader = create_dataloader(dataset, batch_size=batch_size, shuffle=True)

    print(f"DataLoader created: batch_size={batch_size}, shuffle=True")
    print(f"  Batches per epoch: {len(dataloader)}")
    print()

    # ---- Run Step 06 demonstrations ----
    demonstrate_batching(dataloader, vocab, dataset)
    demonstrate_shuffling(dataset, vocab)
    demonstrate_full_epoch(dataloader)

    # ---- What comes next ----
    print("=" * 60)
    print("WHAT COMES NEXT (Step 07)")
    print("=" * 60)
    print("""
Now we have data flowing in organized batches. Next, we build
the NEURAL NETWORK itself — the model that will learn to predict
the next character.

The model will have three layers:
  1. EMBEDDING  — converts character numbers into rich vectors
  2. RNN        — processes the sequence and builds up context
  3. OUTPUT     — predicts probabilities for the next character

This is where the magic happens!
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
