"""
dataset.py — PyTorch Dataset and DataLoader for equation-solution pairs.

PURPOSE:
    Loads the generated equations, encodes them using the vocabulary,
    and serves them as padded batches for training.

    This is more complex than the text generator's dataset because:
    1. We have INPUT-OUTPUT PAIRS (equation → solution), not just sequences
    2. Sequences have DIFFERENT LENGTHS ("3x = 9" vs "3x + 2 = x + 8")
    3. We need PADDING to make all sequences in a batch the same length
    4. We need TRAIN/TEST SPLIT to evaluate on unseen data

WHY PADDING?
    Neural networks process data in batches (groups of examples).
    But equations have different lengths:

        "3x = 9"           →  6 characters
        "2x + 3 = 7"       → 10 characters
        "3x + 2 = x + 8"   → 14 characters

    PyTorch tensors must be rectangular (every row same length), so
    we pad shorter sequences with <PAD> tokens:

        "3x = 9"           → [10, 18, 3, 17, 3, 16,  2,  0,  0,  0]
        "2x + 3 = 7"       → [ 9, 18, 3,  4, 3, 10,  3, 17, 3, 14]
                                                             pad ──^

    The loss function ignores <PAD> positions (ignore_index=0), so
    padding doesn't affect training — it's purely structural.

WHY TRAIN/TEST SPLIT?
    We split the 50,000 equations into:
    - Training set (40,000): the model learns from these
    - Test set (10,000): we evaluate on these AFTER training

    The test set must NEVER be seen during training. If the model
    memorized training data and we tested on the same data, we'd get
    a misleadingly high accuracy. The test set measures how well the
    model GENERALIZES to new, unseen equations.

WHAT THIS FILE PROVIDES:
    1. AlgebraDataset     — PyTorch Dataset for equation-solution pairs
    2. collate_fn()       — pads variable-length sequences into uniform batches
    3. load_pairs()       — reads equation\\tsolution lines from file
    4. create_dataloaders()— builds train and test DataLoaders with split
    5. main()             — demo showing shapes and sample batches

INPUT:  algebra/data/equations.txt + AlgebraVocab
OUTPUT: DataLoader yielding batches of (encoder_input, decoder_input, decoder_target)

Usage:
    from dataset import AlgebraDataset, create_dataloaders
    from vocab import build_vocab_from_data
    vocab = build_vocab_from_data("algebra/data/equations.txt")
    train_loader, test_loader = create_dataloaders(vocab)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from vocab import PAD_IDX, SOS_IDX, EOS_IDX


# ====================================================================
# DATA LOADING
# ====================================================================

def load_pairs(filepath="algebra/data/equations.txt"):
    """
    Load equation-solution pairs from the tab-separated data file.

    Each line is: equation<TAB>solution
    Example: "2x + 3 = 7\\tx = 2"

    PARAMETERS:
        filepath (str): Path to the data file

    RETURNS:
        list[tuple]: List of (equation_str, solution_str) tuples
    """

    pairs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) == 2:
                pairs.append((parts[0], parts[1]))
    return pairs


# ====================================================================
# DATASET
# ====================================================================

class AlgebraDataset(Dataset):
    """
    PyTorch Dataset for equation-solution pairs.

    Each item contains three tensors:
        1. encoder_input:  the equation, encoded + <EOS>
        2. decoder_input:  <SOS> + the solution, encoded
        3. decoder_target: the solution, encoded + <EOS>

    WHY THREE TENSORS?
        - encoder_input: what the encoder reads (the equation)
        - decoder_input: what the decoder receives at each step
          (starts with <SOS>, then each character of the answer)
        - decoder_target: what the decoder should predict at each step
          (the answer shifted by 1, ending with <EOS>)

    EXAMPLE for "2x + 3 = 7" → "x = 2":
        encoder_input:  [9, 18, 3, 4, 3, 10, 3, 17, 3, 14, 2]
                         2  x     +     3     =     7     EOS

        decoder_input:  [1, 18, 3, 17, 3, 9]
                        SOS x     =     2

        decoder_target: [18, 3, 17, 3, 9, 2]
                         x     =     2  EOS

    The decoder input and target are offset by 1 position:
        input:  [<SOS>, x,  " ", =,  " ", 2  ]
        target: [x,     " ", =,  " ", 2,  <EOS>]

    At each step, the decoder receives one token and must predict
    the next one. This is how it learns the answer character by character.

    ATTRIBUTES:
        pairs (list):    Raw (equation, solution) string pairs
        vocab:           AlgebraVocab for encoding
        encoded (list):  Pre-encoded (enc_input, dec_input, dec_target) tuples
    """

    def __init__(self, pairs, vocab):
        """
        Encode all pairs upfront for fast access during training.

        PRE-ENCODING:
            We encode everything once in __init__ rather than encoding
            on-the-fly in __getitem__. This is faster because:
            - Each pair is encoded exactly once (not re-encoded every epoch)
            - __getitem__ just returns a pre-computed tensor

        PARAMETERS:
            pairs (list): List of (equation_str, solution_str) tuples
            vocab (AlgebraVocab): Vocabulary for encoding
        """

        self.pairs = pairs
        self.vocab = vocab
        self.encoded = []

        for equation, solution in pairs:
            # ---- Encode the equation (encoder input) ----
            # Append <EOS> so the encoder knows where the equation ends
            enc_input = vocab.encode_with_eos(equation)

            # ---- Encode the solution (decoder input and target) ----
            sol_encoded = vocab.encode(solution)

            # Decoder input: <SOS> + solution
            # (the decoder receives <SOS> first, then each answer character)
            dec_input = [SOS_IDX] + sol_encoded

            # Decoder target: solution + <EOS>
            # (the decoder must predict each answer character, then <EOS> to stop)
            dec_target = sol_encoded + [EOS_IDX]

            self.encoded.append((
                torch.tensor(enc_input, dtype=torch.long),
                torch.tensor(dec_input, dtype=torch.long),
                torch.tensor(dec_target, dtype=torch.long),
            ))

    def __len__(self):
        """Return the number of equation-solution pairs."""
        return len(self.encoded)

    def __getitem__(self, idx):
        """
        Return one encoded example.

        PARAMETERS:
            idx (int): Index of the example

        RETURNS:
            tuple: (encoder_input, decoder_input, decoder_target) tensors
        """
        return self.encoded[idx]


# ====================================================================
# COLLATE FUNCTION — custom batch assembly with padding
# ====================================================================

def collate_fn(batch):
    """
    Pad variable-length sequences to form uniform batches.

    PyTorch's default collate function requires all tensors in a batch
    to have the same shape. Our sequences have different lengths, so
    we need a CUSTOM collate function that pads shorter sequences.

    HOW PADDING WORKS:
        Given a batch of 3 examples:
            encoder_inputs: lengths [10, 14, 8]
            decoder_inputs: lengths [6, 8, 5]
            decoder_targets: lengths [6, 8, 5]

        We pad each group to the max length in that group:
            encoder_inputs: all padded to length 14
            decoder_inputs: all padded to length 8
            decoder_targets: all padded to length 8

    PADDING VALUE:
        We use PAD_IDX (0) for padding. The loss function will be
        configured with ignore_index=0 to skip these positions.

    PARAMETERS:
        batch (list): List of (enc_input, dec_input, dec_target) tuples
                      from AlgebraDataset.__getitem__

    RETURNS:
        tuple: (enc_padded, dec_input_padded, dec_target_padded)
               Each is a tensor of shape (batch_size, max_length)
    """

    # ---- Separate the three components ----
    enc_inputs, dec_inputs, dec_targets = zip(*batch)

    # ---- Pad each group to its max length ----
    # torch.nn.utils.rnn.pad_sequence pads a list of tensors.
    # batch_first=True makes shape (batch, seq_len) instead of (seq_len, batch).
    # padding_value=PAD_IDX fills with 0 (our <PAD> token).
    enc_padded = torch.nn.utils.rnn.pad_sequence(
        enc_inputs, batch_first=True, padding_value=PAD_IDX
    )
    dec_input_padded = torch.nn.utils.rnn.pad_sequence(
        dec_inputs, batch_first=True, padding_value=PAD_IDX
    )
    dec_target_padded = torch.nn.utils.rnn.pad_sequence(
        dec_targets, batch_first=True, padding_value=PAD_IDX
    )

    return enc_padded, dec_input_padded, dec_target_padded


# ====================================================================
# DATALOADERS WITH TRAIN/TEST SPLIT
# ====================================================================

def create_dataloaders(vocab, filepath="algebra/data/equations.txt",
                       batch_size=64, train_ratio=0.8, seed=42):
    """
    Load data, split into train/test, and create DataLoaders.

    PROCESS:
        1. Load all 50,000 equation-solution pairs
        2. Shuffle with a fixed seed (reproducible)
        3. Split: first 80% for training, last 20% for testing
        4. Create AlgebraDataset for each split
        5. Wrap in DataLoaders with our custom collate function

    WHY FIXED SEED FOR SPLITTING?
        Using the same seed every time ensures:
        - Training and test sets are always the same
        - Results are reproducible across runs
        - We can compare models fairly (same test set)

    PARAMETERS:
        vocab (AlgebraVocab): Vocabulary for encoding
        filepath (str):       Path to the data file
        batch_size (int):     Batch size (default 64)
        train_ratio (float):  Fraction for training (default 0.8 = 80%)
        seed (int):           Random seed for reproducible splitting

    RETURNS:
        tuple: (train_loader, test_loader) DataLoader objects
    """

    import random

    # ---- Load all pairs ----
    all_pairs = load_pairs(filepath)
    print(f"  Loaded {len(all_pairs):,} equation-solution pairs")

    # ---- Shuffle with fixed seed ----
    rng = random.Random(seed)
    rng.shuffle(all_pairs)

    # ---- Split into train and test ----
    split_idx = int(len(all_pairs) * train_ratio)
    train_pairs = all_pairs[:split_idx]
    test_pairs = all_pairs[split_idx:]

    print(f"  Train set: {len(train_pairs):,} pairs")
    print(f"  Test set:  {len(test_pairs):,} pairs")

    # ---- Create datasets ----
    train_dataset = AlgebraDataset(train_pairs, vocab)
    test_dataset = AlgebraDataset(test_pairs, vocab)

    # ---- Create dataloaders ----
    # shuffle=True for training (see different order each epoch)
    # shuffle=False for testing (consistent evaluation)
    # drop_last=True for training (skip incomplete last batch)
    # drop_last=False for testing (evaluate every example)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )

    return train_loader, test_loader


# ====================================================================
# MAIN — demo the dataset and dataloader
# ====================================================================

def main():
    """
    Demo: load data, build dataset, show shapes and sample batches.
    """

    print("=" * 60)
    print("STEP 04: DATASET AND DATALOADER")
    print("=" * 60)
    print()

    # ---- Build vocabulary ----
    from vocab import build_vocab_from_data
    vocab = build_vocab_from_data("algebra/data/equations.txt")
    print(f"  Vocabulary: {vocab.vocab_size} tokens")
    print()

    # ---- Create dataloaders ----
    train_loader, test_loader = create_dataloaders(vocab, batch_size=64)

    # ---- Show batch counts ----
    print()
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches:  {len(test_loader)}")
    print()

    # ---- Show one batch ----
    print("  Sample batch shapes:")
    print("  " + "-" * 50)

    enc_input, dec_input, dec_target = next(iter(train_loader))

    print(f"    Encoder input:  {enc_input.shape}")
    print(f"      (batch_size={enc_input.shape[0]}, "
          f"max_equation_len={enc_input.shape[1]})")
    print(f"    Decoder input:  {dec_input.shape}")
    print(f"      (batch_size={dec_input.shape[0]}, "
          f"max_solution_len={dec_input.shape[1]})")
    print(f"    Decoder target: {dec_target.shape}")
    print(f"      (batch_size={dec_target.shape[0]}, "
          f"max_solution_len={dec_target.shape[1]})")

    # ---- Show one example decoded ----
    print()
    print("  First example in batch (decoded):")
    print("  " + "-" * 50)

    # Encoder input (the equation)
    enc_tokens = enc_input[0].tolist()
    equation_decoded = vocab.decode_until_eos(enc_tokens)
    print(f"    Equation:       \"{equation_decoded}\"")
    print(f"    Encoder input:  {enc_tokens}")

    # Decoder input (<SOS> + solution)
    dec_in_tokens = dec_input[0].tolist()
    print(f"    Decoder input:  {dec_in_tokens}")
    print(f"      Starts with SOS (index {SOS_IDX}), "
          f"then solution characters")

    # Decoder target (solution + <EOS>)
    dec_tgt_tokens = dec_target[0].tolist()
    solution_decoded = vocab.decode_until_eos(dec_tgt_tokens)
    print(f"    Decoder target: {dec_tgt_tokens}")
    print(f"    Solution:       \"{solution_decoded}\"")

    # ---- Show the input/target offset ----
    print()
    print("  Decoder input vs target alignment:")
    print("  " + "-" * 50)
    print(f"    Input:  {dec_in_tokens[:8]}")
    print(f"    Target: {dec_tgt_tokens[:8]}")
    print(f"    (Target is input shifted left by 1 position)")
    print(f"    At each step, decoder receives one token and")
    print(f"    must predict the next token (the target).")

    # ---- Show padding ----
    print()
    print("  Padding example (last 5 tokens of 3 encoder inputs):")
    print("  " + "-" * 50)
    for i in range(3):
        tokens = enc_input[i].tolist()
        eq = vocab.decode_until_eos(tokens)
        last5 = tokens[-5:]
        pad_count = tokens.count(PAD_IDX)
        print(f"    \"{eq}\"")
        print(f"      Last 5 tokens: {last5}  "
              f"({pad_count} padding tokens)")

    # ---- Verify no data leakage ----
    print()
    print("  Data leakage check:")
    print("  " + "-" * 50)
    all_pairs = load_pairs("algebra/data/equations.txt")
    import random
    rng = random.Random(42)
    rng.shuffle(all_pairs)
    split_idx = int(len(all_pairs) * 0.8)
    train_eqs = {eq for eq, _ in all_pairs[:split_idx]}
    test_eqs = {eq for eq, _ in all_pairs[split_idx:]}
    overlap = train_eqs & test_eqs
    print(f"    Train equations: {len(train_eqs):,}")
    print(f"    Test equations:  {len(test_eqs):,}")
    print(f"    Overlap:         {len(overlap)} {'✓ (no leakage)' if len(overlap) == 0 else '✗ LEAKAGE!'}")

    print()
    print("  " + "=" * 50)
    print("  Dataset and DataLoader ready!")


if __name__ == "__main__":
    main()
