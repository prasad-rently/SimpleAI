"""
encoder.py — The encoder network that reads and understands equations.

PURPOSE:
    The encoder is the first half of our seq2seq model. Its job is to
    read an equation like "2x + 3 = 7" character by character and
    compress its meaning into a fixed-size vector (the "context vector").

    Think of the encoder as a student reading a math problem:
    - It reads each character from left to right (and right to left)
    - It builds up an understanding of what the equation is asking
    - At the end, it has a "summary" of the entire problem in its head

HOW IT DIFFERS FROM THE TEXT GENERATOR'S RNN:
    Text Generator:
        - Simple RNN (one direction, left→right only)
        - Output at every position was used (for next-char prediction)
        - Hidden state carried forward for generation

    Encoder:
        - GRU instead of RNN (better at remembering important info)
        - BIDIRECTIONAL (reads left→right AND right→left)
        - Only the final hidden state matters (the "context vector")
        - Output passed to decoder, not used for direct prediction

WHY GRU INSTEAD OF SIMPLE RNN?
    The simple RNN from the text generator has a "forgetting" problem:
    by the time it reads the end of "2x + 3 = 7", it's forgotten details
    about the "2x" at the start. This is called the "vanishing gradient"
    problem.

    GRU (Gated Recurrent Unit) fixes this with GATES — learned switches
    that control what information to keep and what to throw away:

        reset gate:  "How much of the old memory should I forget?"
        update gate: "How much of the new input should I remember?"

    This lets the GRU selectively remember important information
    (like coefficients and constants) even in longer sequences.

WHY BIDIRECTIONAL?
    A regular RNN reads only left→right:
        "2x + 3 = 7"  →  at position "7", it knows "2x + 3 = "

    A bidirectional RNN reads BOTH directions:
        Forward:  "2" → "x" → "+" → "3" → "=" → "7"
        Backward: "7" → "=" → "3" → "+" → "x" → "2"

    This means at every position, the model sees context from BOTH sides.
    For "2x + 3 = 7", the "=" sign knows what's on BOTH its left and right.
    This is critical for math where the right side (= 7) is essential
    for understanding the left side (2x + 3).

WHAT THIS FILE PROVIDES:
    1. Encoder class — bidirectional GRU that reads equations
       - forward(x) → outputs, hidden
       - outputs: context at every position (useful for attention later)
       - hidden: final hidden state = "context vector" for decoder

INPUT:  Encoded equation tensor, shape (batch_size, seq_len)
OUTPUT: outputs (batch, seq_len, hidden*2), hidden (num_layers, batch, hidden*2)

Usage:
    from encoder import Encoder
    encoder = Encoder(vocab_size=19, embed_size=64, hidden_size=128, num_layers=2)
    outputs, hidden = encoder(encoder_input)
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Bidirectional GRU encoder that reads an equation and produces
    a context vector summarizing its meaning.

    ARCHITECTURE:
        Input indices → Embedding → Bidirectional GRU → Context vector

        Embedding(19, 64):
            Each character index → 64-dimensional vector.
            19 = vocab size (16 chars + 3 special tokens).
            64 = embedding dimension (richer than needed for 19 tokens,
                 but gives the model room to learn nuanced representations).

        GRU(64, 128, 2 layers, bidirectional=True):
            Reads the embedded sequence in both directions.
            128 = hidden size per direction.
            Output per position = 256 (128 forward + 128 backward).
            2 layers = stacked for deeper processing.

        Hidden state transform:
            The bidirectional GRU produces 4 hidden states
            (2 layers × 2 directions). We reshape and project these
            to match what the decoder expects (2 layers × 256 dims).

    PARAMETER COUNT:
        Embedding:      19 × 64         =     1,216
        GRU Layer 1:    2 × 3 × (64+128) × 128 = 147,456
        GRU Layer 2:    2 × 3 × (256+128) × 128 = 295,296  (input is 256 from bidir)
        Hidden bridge:  256 × 256 + 256  =    65,792
        Total:                           ≈   509,760
    """

    def __init__(self, vocab_size, embed_size=64, hidden_size=128,
                 num_layers=2, dropout=0.1):
        """
        Initialize the encoder.

        PARAMETERS:
            vocab_size (int):  Number of tokens in vocabulary (19)
            embed_size (int):  Embedding dimension (64)
            hidden_size (int): GRU hidden size per direction (128)
            num_layers (int):  Number of stacked GRU layers (2)
            dropout (float):   Dropout between GRU layers (0.1)
        """

        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # ---- Embedding layer ----
        # padding_idx=0 tells PyTorch that index 0 (<PAD>) should always
        # map to a zero vector and NOT be updated during training.
        # This prevents padding tokens from affecting the model.
        self.embedding = nn.Embedding(
            vocab_size, embed_size, padding_idx=0
        )

        # ---- Bidirectional GRU ----
        # batch_first=True: input shape is (batch, seq_len, embed_size)
        # bidirectional=True: reads in both directions, doubling output size
        # dropout: applied between layers (not after the last layer)
        self.gru = nn.GRU(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # ---- Hidden state bridge ----
        # The bidirectional GRU produces hidden states of shape:
        #   (num_layers * 2, batch, hidden_size)
        # But the decoder (unidirectional) expects:
        #   (num_layers, batch, hidden_size * 2)
        #
        # We use a linear layer to transform each layer's combined
        # forward+backward hidden state into the decoder's format.
        # This is the "bridge" between encoder and decoder.
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size * 2)

    def forward(self, x):
        """
        Read an equation and produce a context vector.

        PROCESS:
            1. Embed the input indices → vectors
            2. Run bidirectional GRU → outputs at every position
            3. Reshape hidden state → context vector for decoder

        PARAMETERS:
            x (Tensor): Encoded equation, shape (batch_size, seq_len)

        RETURNS:
            outputs (Tensor): GRU output at every position
                              shape (batch, seq_len, hidden_size * 2)
            hidden (Tensor):  Final hidden state for decoder
                              shape (num_layers, batch, hidden_size * 2)
        """

        # ---- Step 1: Embed ----
        # (batch, seq_len) → (batch, seq_len, embed_size)
        embedded = self.embedding(x)

        # ---- Step 2: Bidirectional GRU ----
        # outputs: (batch, seq_len, hidden_size * 2)
        #   → context at every position (forward + backward concatenated)
        # hidden: (num_layers * 2, batch, hidden_size)
        #   → final hidden state from each layer and direction
        outputs, hidden = self.gru(embedded)

        # ---- Step 3: Reshape hidden for decoder ----
        # hidden shape is (num_layers * 2, batch, hidden_size)
        # which is interleaved: [layer0_fwd, layer0_bwd, layer1_fwd, layer1_bwd]
        #
        # We need to combine forward and backward for each layer:
        # layer0: cat(layer0_fwd, layer0_bwd) → (batch, hidden_size * 2)
        # layer1: cat(layer1_fwd, layer1_bwd) → (batch, hidden_size * 2)
        #
        # Result: (num_layers, batch, hidden_size * 2)

        batch_size = x.shape[0]

        # Reshape: (num_layers * 2, batch, hidden) → (num_layers, 2, batch, hidden)
        hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_size)

        # Concatenate forward and backward: (num_layers, batch, hidden * 2)
        hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=2)

        # Apply bridge transform with tanh activation
        # tanh squashes values to [-1, 1], preventing extreme values
        hidden = torch.tanh(self.fc_hidden(hidden))

        return outputs, hidden


# ====================================================================
# MAIN — demo the encoder
# ====================================================================

def main():
    """Demo: create encoder, run forward pass, show shapes."""

    print("=" * 60)
    print("STEP 05: ENCODER (BIDIRECTIONAL GRU)")
    print("=" * 60)
    print()

    # ---- Create encoder ----
    vocab_size = 19
    encoder = Encoder(vocab_size=vocab_size, embed_size=64,
                      hidden_size=128, num_layers=2)

    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"  Encoder created: {total_params:,} parameters")
    print()

    # ---- Show architecture ----
    print("  Architecture:")
    print(f"    {encoder}")
    print()

    # ---- Parameter breakdown ----
    print("  Parameter breakdown:")
    print("  " + "-" * 50)
    for name, param in encoder.named_parameters():
        print(f"    {name:<30} {str(list(param.shape)):<20} "
              f"{param.numel():>8,}")
    print(f"    {'TOTAL':<30} {'':20} {total_params:>8,}")
    print()

    # ---- Forward pass with sample input ----
    print("  Forward pass with sample batch:")
    print("  " + "-" * 50)

    # Simulate a batch of 4 equations, each 12 tokens long
    sample_input = torch.randint(3, vocab_size, (4, 12))
    print(f"    Input shape:  {sample_input.shape}  (4 equations, 12 tokens each)")

    outputs, hidden = encoder(sample_input)
    print(f"    Output shape: {outputs.shape}  "
          f"(4 equations, 12 positions, {outputs.shape[2]} dims)")
    print(f"    Hidden shape: {hidden.shape}  "
          f"({hidden.shape[0]} layers, 4 equations, {hidden.shape[2]} dims)")
    print()
    print(f"    Output dim = {outputs.shape[2]} = 128 forward + 128 backward")
    print(f"    Hidden dim = {hidden.shape[2]} = 128 forward + 128 backward")
    print(f"    Hidden is the 'context vector' passed to the decoder")

    print()
    print("  " + "=" * 50)
    print("  Encoder ready!")


if __name__ == "__main__":
    main()
