"""
decoder.py — The decoder network that generates solutions.

PURPOSE:
    The decoder is the second half of our seq2seq model. It receives
    the "context vector" from the encoder and generates the solution
    one character at a time.

    Think of the decoder as a student writing an answer:
    - It receives a summary of the problem (context vector)
    - It starts writing with a blank page (<SOS> token)
    - At each step, it writes one character and uses it to decide
      what to write next
    - It stops when it writes the "done" signal (<EOS> token)

HOW IT WORKS — TWO MODES:
    TRAINING MODE (teacher forcing):
        The decoder receives the CORRECT previous character at each step,
        even if it predicted wrong. This is like having a teacher who
        says "write x next" regardless of what the student wrote.

        Why? Without teacher forcing, one wrong character derails
        everything. The model can't learn from completely wrong sequences.
        Teacher forcing gives clean learning signal at every position.

    INFERENCE MODE (autoregressive):
        The decoder uses its OWN predictions. No teacher, no help.
        It predicts a character, feeds it back as input, and repeats.
        This is how the model works in the real world.

ARCHITECTURE:
    <SOS> or previous character
         │
         ▼
    Embedding(19, 64)     ← same embedding size as encoder
         │
         ▼
    GRU(64, 256, 2 layers) ← unidirectional (left→right only)
         │                    hidden_size=256 to match encoder's output
         ▼
    Linear(256, 19)        ← produce score for each possible token
         │
         ▼
    19 logits (one per token in vocabulary)

WHY UNIDIRECTIONAL?
    The encoder was bidirectional because it reads the whole equation.
    The decoder is unidirectional because it generates LEFT TO RIGHT —
    when writing "x = 2", it can't peek at the "2" while writing "x".
    It must generate each character knowing only what came before it.

WHAT THIS FILE PROVIDES:
    1. Decoder class — GRU decoder with two forward modes
       - forward(x, hidden, target)  → training mode with teacher forcing
       - forward(x, hidden)          → inference mode (autoregressive)

INPUT:  <SOS> token + encoder hidden state (context vector)
OUTPUT: Predicted token indices forming the solution

Usage:
    from decoder import Decoder
    decoder = Decoder(vocab_size=19, embed_size=64, hidden_size=256, num_layers=2)
    outputs, hidden = decoder(sos_input, encoder_hidden, target_sequence)
"""

import torch
import torch.nn as nn
import random

from vocab import SOS_IDX, EOS_IDX


class Decoder(nn.Module):
    """
    GRU decoder that generates solutions character by character.

    Takes the encoder's context vector as its initial hidden state
    and generates the answer one token at a time.

    ARCHITECTURE:
        Embedding(19, 64)    →  token index to vector
        GRU(64, 256, 2)      →  process sequence with context
        Linear(256, 19)      →  predict next token

    The hidden_size (256) matches the encoder's output dimension
    (128 forward + 128 backward = 256). This is necessary because
    the encoder's final hidden state initializes the decoder's hidden state.

    ATTRIBUTES:
        embedding (Embedding): Token → vector lookup
        gru (GRU):             Recurrent layer
        output_layer (Linear): Hidden state → token logits
        hidden_size (int):     GRU hidden dimension
        vocab_size (int):      Number of output tokens
        max_length (int):      Maximum generation length (safety limit)
    """

    def __init__(self, vocab_size, embed_size=64, hidden_size=256,
                 num_layers=2, dropout=0.1, max_length=20):
        """
        Initialize the decoder.

        PARAMETERS:
            vocab_size (int):  Number of tokens (19)
            embed_size (int):  Embedding dimension (64)
            hidden_size (int): GRU hidden size (256, must match encoder output)
            num_layers (int):  Stacked GRU layers (2, must match encoder)
            dropout (float):   Dropout between layers (0.1)
            max_length (int):  Max tokens to generate during inference (20)
        """

        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_length = max_length

        # ---- Embedding ----
        # Same structure as encoder's embedding.
        # padding_idx=0 for <PAD> consistency.
        self.embedding = nn.Embedding(
            vocab_size, embed_size, padding_idx=0
        )

        # ---- GRU (unidirectional) ----
        # NOT bidirectional — decoder generates left to right only.
        # hidden_size=256 matches encoder's bidirectional output (128*2).
        self.gru = nn.GRU(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # ---- Output projection ----
        # Converts 256-dim hidden state to 19 logits (one per token).
        # The highest logit = the predicted next token.
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, encoder_hidden, target=None, teacher_forcing_ratio=0.5):
        """
        Generate a solution, with optional teacher forcing.

        TWO MODES:
            1. Training (target is provided):
               Uses teacher forcing — at each step, with probability
               teacher_forcing_ratio, feeds the correct target token
               instead of the model's own prediction.

            2. Inference (target is None):
               Pure autoregressive — always feeds own predictions.
               Stops at <EOS> or max_length.

        PARAMETERS:
            encoder_hidden (Tensor): Context from encoder
                                      shape (num_layers, batch, hidden_size)
            target (Tensor or None):  Target sequence for teacher forcing
                                      shape (batch, target_len)
                                      None for inference mode
            teacher_forcing_ratio (float): Probability of using correct token
                                           during training (0.0 to 1.0)

        RETURNS:
            outputs (Tensor): Logits at each step
                              shape (batch, seq_len, vocab_size)
            hidden (Tensor):  Final hidden state
                              shape (num_layers, batch, hidden_size)
        """

        batch_size = encoder_hidden.shape[1]
        hidden = encoder_hidden

        # ---- Determine sequence length ----
        if target is not None:
            # Training: generate as many steps as the target
            max_steps = target.shape[1]
        else:
            # Inference: generate up to max_length
            max_steps = self.max_length

        # ---- Start with <SOS> token for every sequence in batch ----
        # Shape: (batch_size, 1) — one token per sequence
        current_input = torch.full(
            (batch_size, 1), SOS_IDX, dtype=torch.long,
            device=encoder_hidden.device
        )

        # ---- Collect outputs at each step ----
        all_outputs = []

        for step in range(max_steps):
            # ---- Embed the current input token ----
            # (batch, 1) → (batch, 1, embed_size)
            embedded = self.embedding(current_input)

            # ---- Run one GRU step ----
            # output: (batch, 1, hidden_size)
            # hidden: (num_layers, batch, hidden_size)
            output, hidden = self.gru(embedded, hidden)

            # ---- Project to vocabulary ----
            # (batch, 1, hidden_size) → (batch, 1, vocab_size)
            logits = self.output_layer(output)

            # ---- Store this step's output ----
            all_outputs.append(logits)

            # ---- Decide next input ----
            if target is not None and random.random() < teacher_forcing_ratio:
                # Teacher forcing: use the correct target token
                # target[:, step:step+1] extracts column 'step' keeping dim
                current_input = target[:, step:step+1]
            else:
                # Autoregressive: use our own prediction
                # argmax picks the most likely token
                current_input = logits.argmax(dim=2)  # (batch, 1)

        # ---- Concatenate all steps ----
        # List of (batch, 1, vocab_size) → (batch, max_steps, vocab_size)
        outputs = torch.cat(all_outputs, dim=1)

        return outputs, hidden

    def generate(self, encoder_hidden):
        """
        Generate a solution in pure inference mode (no teacher forcing).

        Convenience method that calls forward with target=None and
        teacher_forcing_ratio=0.

        PARAMETERS:
            encoder_hidden (Tensor): Context from encoder
                                      shape (num_layers, batch, hidden_size)

        RETURNS:
            predicted_indices (Tensor): Predicted token indices
                                         shape (batch, generated_len)
            outputs (Tensor):           Raw logits
                                         shape (batch, generated_len, vocab_size)
        """

        # ---- Generate without teacher forcing ----
        outputs, _ = self.forward(
            encoder_hidden, target=None, teacher_forcing_ratio=0.0
        )

        # ---- Extract predicted indices ----
        predicted = outputs.argmax(dim=2)  # (batch, generated_len)

        return predicted, outputs


# ====================================================================
# MAIN — demo the decoder
# ====================================================================

def main():
    """Demo: create decoder, run forward pass, show shapes."""

    print("=" * 60)
    print("STEP 06: DECODER (GRU WITH TEACHER FORCING)")
    print("=" * 60)
    print()

    # ---- Create decoder ----
    vocab_size = 19
    decoder = Decoder(vocab_size=vocab_size, embed_size=64,
                      hidden_size=256, num_layers=2)

    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"  Decoder created: {total_params:,} parameters")
    print()

    # ---- Show architecture ----
    print("  Architecture:")
    print(f"    {decoder}")
    print()

    # ---- Parameter breakdown ----
    print("  Parameter breakdown:")
    print("  " + "-" * 50)
    for name, param in decoder.named_parameters():
        print(f"    {name:<30} {str(list(param.shape)):<20} "
              f"{param.numel():>8,}")
    print(f"    {'TOTAL':<30} {'':20} {total_params:>8,}")
    print()

    # ---- Training mode (with teacher forcing) ----
    print("  Training mode (teacher forcing):")
    print("  " + "-" * 50)

    # Simulate encoder hidden state: (2 layers, batch=4, hidden=256)
    fake_hidden = torch.randn(2, 4, 256)
    # Simulate target: (batch=4, target_len=7) — "x = 123" = 7 chars
    fake_target = torch.randint(3, vocab_size, (4, 7))

    outputs, final_hidden = decoder(fake_hidden, target=fake_target,
                                     teacher_forcing_ratio=0.5)
    print(f"    Encoder hidden: {fake_hidden.shape}")
    print(f"    Target:         {fake_target.shape}")
    print(f"    Output:         {outputs.shape}  "
          f"(batch, {outputs.shape[1]} steps, {outputs.shape[2]} token scores)")
    print(f"    Final hidden:   {final_hidden.shape}")
    print()

    # ---- Inference mode (autoregressive) ----
    print("  Inference mode (autoregressive):")
    print("  " + "-" * 50)

    predicted, logits = decoder.generate(fake_hidden)
    print(f"    Encoder hidden: {fake_hidden.shape}")
    print(f"    Predicted:      {predicted.shape}  "
          f"(batch, up to {decoder.max_length} tokens)")
    print(f"    Logits:         {logits.shape}")
    print(f"    Sample prediction (first in batch): {predicted[0].tolist()}")

    print()
    print("  " + "=" * 50)
    print("  Decoder ready!")


if __name__ == "__main__":
    main()
