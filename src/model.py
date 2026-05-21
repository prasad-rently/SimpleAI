"""
model.py — The neural network that learns to predict the next character.

PURPOSE:
    This file defines the actual AI model — a small neural network called
    TinyLanguageModel. It takes character numbers as input and outputs
    predictions for what the next character should be.

    This is the BRAIN of the project. Everything before this (Steps 01-06)
    was preparing the data. Everything after this (Steps 08-14) is training
    the brain and using it to generate text.

HOW A NEURAL NETWORK WORKS (SIMPLIFIED):
    A neural network is a series of math operations that transform input
    data into output predictions. Our model has three layers:

    ┌──────────────────────────────────────────────────────────────────┐
    │                    TinyLanguageModel                             │
    │                                                                  │
    │   Input          Layer 1         Layer 2         Layer 3         │
    │   (numbers)      EMBEDDING       RNN             OUTPUT          │
    │                                                                  │
    │   [19, 29, 26]  ──▶ [vectors] ──▶ [context] ──▶ [48 scores]    │
    │   "T"  "h"  "e"     (rich        (understands   (one per        │
    │                       meaning)     sequences)     character)     │
    │                                                                  │
    │   Each layer is a matrix of LEARNABLE NUMBERS (weights).        │
    │   Training adjusts these weights until predictions are good.     │
    └──────────────────────────────────────────────────────────────────┘

    LAYER 1 — EMBEDDING:
        Converts each character number into a "rich" vector of floats.
        Why? The number 19 (for 'T') doesn't carry meaning. But a
        vector like [0.3, -0.7, 0.1, ...] can encode relationships:
        similar characters get similar vectors.

        Think of it like this:
          Before: 'T' = just the number 19 (meaningless)
          After:  'T' = [0.3, -0.7, 0.1, 0.5, ...] (128 numbers that
                        capture things like "uppercase", "consonant",
                        "often starts sentences", etc.)

    LAYER 2 — RNN (Recurrent Neural Network):
        Processes the sequence LEFT TO RIGHT, building up "context".
        After seeing 'T', 'h', 'e', the RNN has a hidden state that
        represents "I've seen 'The' so far". This context helps predict
        what comes next.

        Think of it like reading a sentence word by word — each new
        word adds to your understanding of the whole sentence.

    LAYER 3 — OUTPUT (Linear):
        Takes the RNN's context and produces 48 scores (one per character
        in our vocabulary). The highest score = the model's best guess
        for the next character.

        Example output for position after "Th":
          'e': 8.5   ← highest score! Model predicts 'e'
          ' ': 3.2
          'a': 2.1
          'i': 1.5
          ... (44 more characters with lower scores)

WHAT THIS FILE PROVIDES:
    1. TinyLanguageModel class — the neural network
    2. Demo that creates the model, passes fake data through it,
       and shows the output shapes

INPUT:  Batched tensors of character indices, shape (batch_size, seq_length)
OUTPUT: Prediction scores for next character, shape (batch_size, seq_length, vocab_size)

Usage:
    python src/model.py
"""

import torch
import torch.nn as nn


class TinyLanguageModel(nn.Module):
    """
    A small character-level language model using an RNN.

    ARCHITECTURE:
        Input (batch_size, seq_length)
          │
          ▼
        Embedding (batch_size, seq_length, embed_size)
          │         Converts numbers → rich vectors
          ▼
        RNN (batch_size, seq_length, hidden_size)
          │   Processes sequence, builds context
          ▼
        Linear (batch_size, seq_length, vocab_size)
          │     Predicts next character scores
          ▼
        Output: 48 scores per position (one per character)

    WHAT IS nn.Module?
        All PyTorch neural networks inherit from nn.Module.
        This gives us:
        - Automatic tracking of all learnable parameters (weights)
        - A standard forward() method that defines the computation
        - Built-in save/load functionality
        - Integration with optimizers for training

        Think of nn.Module as a "smart container" that knows about
        all the numbers (weights) inside it and how to update them.

    PARAMETERS:
        vocab_size (int):  Number of unique characters (48 for our data).
                           Determines the size of the embedding table and
                           the output layer.
        embed_size (int):  Size of character embedding vectors (default: 128).
                           How many numbers represent each character.
                           Larger = more expressive but slower.
        hidden_size (int): Size of the RNN hidden state (default: 256).
                           The "memory capacity" of the model.
                           Larger = can remember more context but slower.
        num_layers (int):  Number of stacked RNN layers (default: 2).
                           More layers = deeper understanding but slower.
                           Think of it as layers of analysis:
                           Layer 1 learns simple patterns ("th", "ing")
                           Layer 2 learns higher patterns ("the meaning of")

    EXAMPLE:
        >>> model = TinyLanguageModel(vocab_size=48)
        >>> # Count total learnable parameters:
        >>> total_params = sum(p.numel() for p in model.parameters())
        >>> print(f"Total parameters: {total_params:,}")
        Total parameters: 469,040
        >>>
        >>> # Compare: GPT-4 has ~1,800,000,000,000 parameters (1.8 trillion!)
        >>> # Our model is about 4 BILLION times smaller.
    """

    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=2):
        """
        Initialize the model's layers and their weights.

        WHAT HAPPENS HERE:
            1. Call super().__init__() — required by PyTorch to set up
               the nn.Module internals (parameter tracking, etc.)
            2. Create three layers, each initialized with random weights
            3. Store hyperparameters for later use

        EACH LAYER EXPLAINED:

            nn.Embedding(vocab_size, embed_size):
                Creates a lookup table of shape (48, 128).
                Each row is a 128-number vector for one character.
                Initially random — training will make similar characters
                have similar vectors.

                INPUT:  tensor of ints, e.g., [19, 29, 26]
                OUTPUT: tensor of floats, shape (3, 128)
                        Each int is replaced by its 128-number row.

            nn.RNN(embed_size, hidden_size, num_layers, batch_first=True):
                A Recurrent Neural Network that processes sequences.
                "Recurrent" = it has a loop: the output of processing
                character N feeds into the processing of character N+1.
                This is how it builds up context.

                INPUT:  embedded vectors, shape (batch, seq_len, 128)
                OUTPUT: context vectors, shape (batch, seq_len, 256)
                        Plus a hidden state that carries context forward.

                batch_first=True means the batch dimension comes first
                in the tensor shape, which is the convention we've been
                using throughout this project.

            nn.Linear(hidden_size, vocab_size):
                A simple matrix multiplication + bias.
                Transforms the 256-number context into 48 scores
                (one per character in the vocabulary).

                INPUT:  context vectors, shape (batch, seq_len, 256)
                OUTPUT: raw scores, shape (batch, seq_len, 48)
                        These scores are called "logits" — they will be
                        converted to probabilities during training/generation.

        PARAMETERS:
            vocab_size (int):  Number of unique characters (48)
            embed_size (int):  Embedding vector size (default: 128)
            hidden_size (int): RNN hidden state size (default: 256)
            num_layers (int):  Number of stacked RNN layers (default: 2)
        """

        # ---- Required PyTorch initialization ----
        # super().__init__() tells PyTorch to set up the Module internals.
        # Without this line, PyTorch won't track our layers or their weights.
        # This is a Python pattern: when a child class (TinyLanguageModel)
        # inherits from a parent class (nn.Module), it must call the
        # parent's __init__() to properly initialize.
        super().__init__()

        # ---- Save hyperparameters ----
        # We save these so other methods (forward, etc.) can use them.
        # "Hyperparameters" = settings that define the model's structure.
        # They are NOT learned — we choose them before training.
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # ---- Layer 1: Embedding ----
        # Creates a table of shape (vocab_size, embed_size) = (48, 128).
        # Each of the 48 characters gets its own 128-number vector.
        #
        # HOW IT WORKS:
        #   self.embedding(tensor([19])) → looks up row 19 of the table
        #   Returns a tensor of shape (1, 128) — the embedding for 'T'.
        #
        # Initially, all 48 × 128 = 6,144 numbers are random.
        # Training adjusts them so similar characters get similar vectors.
        #
        # ANALOGY: It's like a dictionary where each word has a definition.
        # The "definition" is a 128-number vector that captures the
        # character's properties. Training writes these definitions.
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # ---- Layer 2: RNN ----
        # A Recurrent Neural Network with 2 stacked layers.
        # Processes the sequence position by position, building context.
        #
        # HOW IT WORKS (simplified):
        #   For each position in the sequence:
        #     hidden = f(embedding[position], hidden_from_previous_position)
        #   Where f() is a learned function (matrix multiplications + activation).
        #
        # The "hidden state" is a vector of `hidden_size` numbers that acts
        # as the model's "working memory". After seeing 'T', 'h', 'e', the
        # hidden state encodes "I've seen 'The' and the next char is likely
        # a space or continuation of a word."
        #
        # num_layers=2: Two RNNs stacked. Output of layer 1 feeds into layer 2.
        #   Layer 1: learns basic patterns (character pairs, common combos)
        #   Layer 2: learns higher-order patterns (word structures, phrases)
        #
        # batch_first=True: input shape is (batch, seq_len, embed_size)
        #   instead of (seq_len, batch, embed_size). Matches our DataLoader.
        self.rnn = nn.RNN(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # ---- Layer 3: Output (Linear) ----
        # A fully-connected layer: matrix of shape (hidden_size, vocab_size).
        # Transforms the 256-number context into 48 prediction scores.
        #
        # HOW IT WORKS:
        #   output = context @ weight_matrix + bias
        #   Where @ is matrix multiplication.
        #
        # Each of the 48 output numbers is a "score" for that character.
        # Higher score = model thinks this character is more likely next.
        # These raw scores are called "logits".
        #
        # EXAMPLE output at one position:
        #   [ 2.1, -0.5,  0.3,  0.8, ..., 8.5, ..., -1.2 ]
        #     'a'    'b'   'c'   'd'        'e'       'z'
        #   'e' has the highest score (8.5) → model predicts 'e'
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        """
        Process input through all three layers to get predictions.

        This method defines the "forward pass" — the computation that
        transforms input character indices into prediction scores.

        In PyTorch, you define forward() and PyTorch handles the
        backward pass (computing gradients) automatically. This is
        called "autograd" and is one of PyTorch's killer features.

        THE FORWARD PASS STEP BY STEP:

            Step 1: Embedding lookup
              [19, 29, 26] → [[0.3, -0.7, ...], [0.1, 0.5, ...], [-0.2, 0.8, ...]]
              Shape: (batch, seq_len) → (batch, seq_len, embed_size)
              Shape: (16, 50) → (16, 50, 128)

            Step 2: RNN processing
              Processes the embedded sequence, building context.
              Shape: (batch, seq_len, embed_size) → (batch, seq_len, hidden_size)
              Shape: (16, 50, 128) → (16, 50, 256)
              Also outputs a hidden state of shape (num_layers, batch, hidden_size)

            Step 3: Output projection
              Converts context to prediction scores.
              Shape: (batch, seq_len, hidden_size) → (batch, seq_len, vocab_size)
              Shape: (16, 50, 256) → (16, 50, 48)

        PARAMETERS:
            x (Tensor):      Input character indices.
                             Shape: (batch_size, seq_length)
                             dtype: torch.long (int64)
                             Example: tensor([[19, 29, 26, ...], ...])

            hidden (Tensor): Optional initial hidden state for the RNN.
                             Shape: (num_layers, batch_size, hidden_size)
                             If None, defaults to zeros (no prior context).
                             Used during generation to carry context between
                             calls (Step 12).

        RETURNS:
            tuple of:
            - logits (Tensor): Prediction scores for each position.
                               Shape: (batch_size, seq_length, vocab_size)
                               Example shape: (16, 50, 48)
                               Each of the 48 values is a raw score for
                               that character being the next one.

            - hidden (Tensor): The final RNN hidden state.
                               Shape: (num_layers, batch_size, hidden_size)
                               Carries the context built up from the sequence.
                               Passed back in during generation for continuity.

        EXAMPLE:
            >>> model = TinyLanguageModel(vocab_size=48)
            >>> x = torch.randint(0, 48, (16, 50))  # fake batch
            >>> logits, hidden = model(x)
            >>> logits.shape
            torch.Size([16, 50, 48])
            >>> hidden.shape
            torch.Size([2, 16, 256])
        """

        # ---- Step 1: Embedding lookup ----
        # Convert integer indices to rich float vectors.
        #
        # INPUT:  x has shape (batch_size, seq_length) = (16, 50)
        #         Each value is an int from 0-47 (a character index).
        #
        # OUTPUT: embedded has shape (batch_size, seq_length, embed_size)
        #         = (16, 50, 128)
        #         Each int has been replaced by its 128-number embedding vector.
        #
        # VISUAL:
        #   x[0] = [19,  29,  26,  ...]
        #            ↓    ↓    ↓
        #   emb[0] = [[0.3, -0.7, ...(128 numbers)],   ← embedding for 'T'
        #             [0.1,  0.5, ...(128 numbers)],   ← embedding for 'h'
        #             [-0.2, 0.8, ...(128 numbers)],   ← embedding for 'e'
        #             ...]
        embedded = self.embedding(x)

        # ---- Step 2: RNN processing ----
        # Process the embedded sequence through the RNN.
        #
        # INPUT:  embedded shape (16, 50, 128) — batch of embedded sequences
        #         hidden shape (2, 16, 256) — initial hidden state
        #           (2 = num_layers, 16 = batch_size, 256 = hidden_size)
        #           If hidden is None, PyTorch uses zeros.
        #
        # OUTPUT: rnn_out shape (16, 50, 256) — context vectors for every position
        #         hidden shape (2, 16, 256) — final hidden state after full sequence
        #
        # WHAT THE RNN DOES (for one example in the batch):
        #   Position 0 ('T'): combines embedding of 'T' with initial hidden state
        #                      → produces context vector for position 0
        #   Position 1 ('h'): combines embedding of 'h' with hidden from position 0
        #                      → produces context that "knows" about 'T' and 'h'
        #   Position 2 ('e'): combines embedding of 'e' with hidden from position 1
        #                      → produces context that "knows" about 'T', 'h', 'e'
        #   ...and so on for all 50 positions.
        #
        # By position 49, the context vector encodes information about
        # the entire 50-character input sequence.
        rnn_out, hidden = self.rnn(embedded, hidden)

        # ---- Step 3: Output projection ----
        # Convert 256-number context vectors to 48-number prediction scores.
        #
        # INPUT:  rnn_out shape (16, 50, 256)
        # OUTPUT: logits shape (16, 50, 48)
        #
        # For each of the 50 positions in each of the 16 examples,
        # we get 48 scores — one per character in the vocabulary.
        #
        # logits[0][2] = 48 scores predicting what comes after position 2
        #                in the first example. The highest score is the
        #                model's best guess.
        #
        # These are called "logits" (raw scores). They'll be converted to
        # probabilities using softmax during generation (Step 12) or fed
        # directly to CrossEntropyLoss during training (Step 08) which
        # applies softmax internally.
        logits = self.output_layer(rnn_out)

        return logits, hidden


def demonstrate_model_creation(vocab_size):
    """
    Create the model and display its architecture and parameter counts.

    PARAMETERS:
        vocab_size (int): Number of unique characters

    OUTPUT: Printed model architecture, layer details, and parameter count.
    """

    print("=" * 60)
    print("MODEL CREATION: Building the neural network")
    print("=" * 60)
    print()

    # ---- Create the model ----
    model = TinyLanguageModel(vocab_size=vocab_size)

    # ---- Show the architecture ----
    # Printing a PyTorch model shows all its layers and their configurations.
    print("Model architecture:")
    print(model)
    print()

    # ---- Count parameters per layer ----
    # model.named_parameters() yields (name, parameter_tensor) pairs.
    # p.numel() counts how many individual numbers in each parameter.
    print("Parameter breakdown:")
    print(f"  {'Layer':<30s} {'Shape':<25s} {'Parameters':>12s}")
    print(f"  {'-'*30} {'-'*25} {'-'*12}")

    total_params = 0
    for name, param in model.named_parameters():
        num = param.numel()
        total_params += num
        print(f"  {name:<30s} {str(tuple(param.shape)):<25s} {num:>12,}")

    print(f"  {'-'*30} {'-'*25} {'-'*12}")
    print(f"  {'TOTAL':<30s} {'':<25s} {total_params:>12,}")
    print()

    # ---- Compare to real models ----
    print("  FOR PERSPECTIVE:")
    print(f"    Our model       : {total_params:>15,} parameters")
    print(f"    GPT-2 (small)   : {'117,000,000':>15s} parameters")
    print(f"    GPT-3           : {'175,000,000,000':>15s} parameters")
    print(f"    GPT-4 (rumored) : {'1,800,000,000,000':>15s} parameters")
    print()
    print(f"    Our model is about {175_000_000_000 // total_params:,}x smaller than GPT-3!")
    print()

    return model


def demonstrate_forward_pass(model, vocab_size):
    """
    Pass fake data through the model and show input/output shapes.

    This demonstrates the complete forward pass — the path data takes
    through all three layers of the network.

    PARAMETERS:
        model (TinyLanguageModel): The model to test
        vocab_size (int):          For creating fake input data

    OUTPUT: Printed shapes at each stage of the forward pass.
    """

    print("=" * 60)
    print("FORWARD PASS: Data flowing through the model")
    print("=" * 60)
    print()

    # ---- Create fake input data ----
    # torch.randint(low, high, shape) creates random integers.
    # We simulate a batch of 4 examples, each 50 characters long.
    # In real training, this data comes from the DataLoader (Step 06).
    batch_size = 4
    seq_length = 50

    fake_input = torch.randint(0, vocab_size, (batch_size, seq_length))

    print(f"  Input (fake batch of character indices):")
    print(f"    Shape: {fake_input.shape}  →  ({batch_size} examples, {seq_length} chars each)")
    print(f"    dtype: {fake_input.dtype}")
    print(f"    First example, first 10 chars: {fake_input[0, :10].tolist()}")
    print()

    # ---- Run the forward pass ----
    # model(fake_input) calls model.forward(fake_input) internally.
    # PyTorch convention: you call the model object directly, not .forward().
    logits, hidden = model(fake_input)

    print(f"  Output (prediction scores):")
    print(f"    logits shape: {logits.shape}  →  ({batch_size} examples, "
          f"{seq_length} positions, {vocab_size} chars)")
    print(f"    hidden shape: {hidden.shape}  →  "
          f"({model.num_layers} layers, {batch_size} examples, {model.hidden_size} hidden)")
    print()

    # ---- Explain what the output means ----
    print(f"  WHAT THE OUTPUT MEANS:")
    print(f"    logits[0][0] = scores for 'what comes after position 0 in example 0'")
    print(f"    logits[0][0] has {vocab_size} values — one score per character")
    print()

    # Show the actual scores for one position
    scores = logits[0][0]
    print(f"    Example: logits[0][0] (first position, first example):")
    print(f"    Top 5 predicted characters:")

    # Get the top 5 highest scores
    top5_values, top5_indices = torch.topk(scores, 5)
    for val, idx in zip(top5_values, top5_indices):
        print(f"      Character index {idx.item():2d}  score: {val.item():+.3f}")
    print()

    print(f"    (These scores are random because the model is UNTRAINED.)")
    print(f"    After training, the highest score should match the correct")
    print(f"    next character most of the time.)")
    print()

    # ---- Show the shape transformation journey ----
    print(f"  SHAPE JOURNEY THROUGH THE MODEL:")
    print(f"  ┌────────────────────────────────────────────────────────┐")
    print(f"  │ Input:     ({batch_size}, {seq_length})"
          f"              ← char indices    │")
    print(f"  │     ↓ Embedding                                       │")
    print(f"  │ Embedded:  ({batch_size}, {seq_length}, 128)"
          f"          ← float vectors   │")
    print(f"  │     ↓ RNN                                             │")
    print(f"  │ Context:   ({batch_size}, {seq_length}, 256)"
          f"          ← with context    │")
    print(f"  │     ↓ Linear                                          │")
    print(f"  │ Logits:    ({batch_size}, {seq_length}, {vocab_size})"
          f"           ← 48 scores/pos  │")
    print(f"  └────────────────────────────────────────────────────────┘")
    print()


def main():
    """
    Main function — creates the model, shows architecture, runs a forward pass.

    FLOW:
        1. Set vocab_size (from our dataset: 48 characters)
        2. Create TinyLanguageModel
        3. Show architecture and parameter counts
        4. Run a forward pass with fake data
        5. Preview what comes next (Step 08: loss + optimizer)
    """

    # ---- Our vocabulary size (from Steps 03-04) ----
    # In a real pipeline, this comes from the Vocabulary object.
    # Here we hardcode it for simplicity since this demo doesn't
    # need actual text data — just the model architecture.
    vocab_size = 48

    print("=" * 60)
    print("STEP 07: THE NEURAL NETWORK")
    print("=" * 60)
    print()
    print(f"Building a TinyLanguageModel for {vocab_size} characters...")
    print()

    # ---- Create and inspect the model ----
    model = demonstrate_model_creation(vocab_size)

    # ---- Run a forward pass ----
    demonstrate_forward_pass(model, vocab_size)

    # ---- What comes next ----
    print("=" * 60)
    print("WHAT COMES NEXT (Step 08)")
    print("=" * 60)
    print("""
The model exists, but it's full of RANDOM weights — it knows nothing.
To teach it, we need two things:

  1. LOSS FUNCTION — measures how wrong the model's predictions are.
     "Your predicted 'z' but the correct answer was 'e'. That's very wrong."
     Gives us a single number: the "loss". Lower = better.

  2. OPTIMIZER — takes the loss and figures out how to adjust the weights
     to make the model less wrong next time.
     "Okay, increase weight #3 by 0.01, decrease weight #7 by 0.005..."

Together, these two components let the model LEARN from its mistakes.
""")


# ======================================================================
# ENTRY POINT
# ======================================================================
# Run this file directly to see the model architecture demo:
#   python src/model.py
#
# Or import the model class in other files:
#   from model import TinyLanguageModel
if __name__ == "__main__":
    main()
