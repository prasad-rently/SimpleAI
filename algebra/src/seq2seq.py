"""
seq2seq.py — Combined encoder-decoder model for solving equations.

PURPOSE:
    This wraps the Encoder and Decoder into a single model that takes
    an equation as input and produces a solution as output.

    It's the "glue" between the two halves:
        Encoder reads "2x + 3 = 7" → context vector → Decoder writes "x = 2"

    By wrapping them in a single nn.Module, we get:
    1. One model to save/load (both encoder and decoder weights together)
    2. One optimizer that updates both networks simultaneously
    3. A clean interface: model(equation) → solution

WHAT THIS FILE PROVIDES:
    1. Seq2Seq class — combines Encoder + Decoder
       - forward(src, target, teacher_forcing_ratio) → training output
       - solve(src) → inference output (predicted solution)
    2. create_model() — factory function with default hyperparameters

INPUT:  Encoded equation tensor
OUTPUT: Predicted solution (logits during training, indices during inference)

Usage:
    from seq2seq import create_model
    model = create_model(vocab_size=19)
    # Training:  outputs = model(equations, targets, teacher_forcing_ratio=0.5)
    # Inference: predicted = model.solve(equations)
"""

import torch
import torch.nn as nn

from encoder import Encoder
from decoder import Decoder
from vocab import EOS_IDX


class Seq2Seq(nn.Module):
    """
    Sequence-to-sequence model: Encoder + Decoder.

    This is the complete algebra solver. It:
    1. Encodes the equation into a context vector (Encoder)
    2. Decodes the context vector into a solution (Decoder)

    ARCHITECTURE:
        ┌────────────────────────────────────────────────────────┐
        │                    Seq2Seq                              │
        │                                                        │
        │   "2x + 3 = 7"  ──→  [Encoder]  ──→  context vector  │
        │                                           │            │
        │                                           ▼            │
        │                        [Decoder]  ──→  "x = 2"        │
        │                                                        │
        └────────────────────────────────────────────────────────┘

    ATTRIBUTES:
        encoder (Encoder): Bidirectional GRU encoder
        decoder (Decoder): GRU decoder with teacher forcing
    """

    def __init__(self, encoder, decoder):
        """
        Initialize with pre-built encoder and decoder.

        PARAMETERS:
            encoder (Encoder): The encoder network
            decoder (Decoder): The decoder network
        """

        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, target=None, teacher_forcing_ratio=0.5):
        """
        Full forward pass: encode equation, decode solution.

        TRAINING MODE (target provided):
            Uses teacher forcing to help the decoder learn.
            Returns logits for loss computation.

        INFERENCE MODE (target=None):
            Decoder generates autoregressively.
            Returns logits for the predicted sequence.

        PARAMETERS:
            src (Tensor):    Encoder input (equations)
                             shape (batch, src_len)
            target (Tensor): Decoder target (solutions), optional
                             shape (batch, target_len)
            teacher_forcing_ratio (float): Probability of teacher forcing

        RETURNS:
            outputs (Tensor): Predicted logits
                              shape (batch, seq_len, vocab_size)
        """

        # ---- Step 1: Encode the equation ----
        # encoder_outputs: (batch, src_len, hidden*2) — not used yet
        # hidden: (num_layers, batch, hidden*2) — the context vector
        encoder_outputs, hidden = self.encoder(src)

        # ---- Step 2: Decode the solution ----
        # The decoder uses the encoder's hidden state as its initial state.
        # This is how the "meaning" of the equation transfers to the decoder.
        outputs, _ = self.decoder(
            hidden, target=target,
            teacher_forcing_ratio=teacher_forcing_ratio
        )

        return outputs

    def solve(self, src):
        """
        Solve equations in inference mode (no teacher forcing).

        This is the method used after training to actually solve
        new equations. No target is provided — the decoder must
        generate the answer entirely on its own.

        PARAMETERS:
            src (Tensor): Encoded equations, shape (batch, src_len)

        RETURNS:
            predictions (Tensor): Predicted token indices
                                   shape (batch, max_length)
            logits (Tensor):      Raw logits
                                   shape (batch, max_length, vocab_size)
        """

        # ---- Encode ----
        _, hidden = self.encoder(src)

        # ---- Decode without teacher forcing ----
        predictions, logits = self.decoder.generate(hidden)

        return predictions, logits


def create_model(vocab_size, embed_size=64, encoder_hidden=128,
                 decoder_hidden=256, num_layers=2, dropout=0.1,
                 max_length=20):
    """
    Factory function to create a Seq2Seq model with default settings.

    This handles the slightly tricky coordination between encoder
    and decoder dimensions:
    - Encoder hidden = 128, but bidirectional makes output 256
    - Decoder hidden must be 256 to accept encoder's output
    - Both must have the same number of layers

    PARAMETERS:
        vocab_size (int):      Number of tokens (19)
        embed_size (int):      Embedding dimension for both (64)
        encoder_hidden (int):  Encoder GRU hidden size per direction (128)
        decoder_hidden (int):  Decoder GRU hidden size (256 = encoder_hidden * 2)
        num_layers (int):      GRU layers for both (2)
        dropout (float):       Dropout rate (0.1)
        max_length (int):      Max generation length (20)

    RETURNS:
        Seq2Seq: The complete model
    """

    encoder = Encoder(
        vocab_size=vocab_size,
        embed_size=embed_size,
        hidden_size=encoder_hidden,
        num_layers=num_layers,
        dropout=dropout,
    )

    decoder = Decoder(
        vocab_size=vocab_size,
        embed_size=embed_size,
        hidden_size=decoder_hidden,
        num_layers=num_layers,
        dropout=dropout,
        max_length=max_length,
    )

    return Seq2Seq(encoder, decoder)


# ====================================================================
# MAIN — demo the full seq2seq model
# ====================================================================

def main():
    """Demo: create model, run training and inference forward passes."""

    print("=" * 60)
    print("STEP 07: SEQ2SEQ MODEL (ENCODER + DECODER)")
    print("=" * 60)
    print()

    # ---- Create model ----
    vocab_size = 19
    model = create_model(vocab_size=vocab_size)

    enc_params = sum(p.numel() for p in model.encoder.parameters())
    dec_params = sum(p.numel() for p in model.decoder.parameters())
    total_params = sum(p.numel() for p in model.parameters())

    print(f"  Model created:")
    print(f"    Encoder:  {enc_params:>10,} parameters")
    print(f"    Decoder:  {dec_params:>10,} parameters")
    print(f"    Total:    {total_params:>10,} parameters")
    print()

    # ---- Architecture ----
    print("  Architecture:")
    print(f"    {model}")
    print()

    # ---- Training forward pass ----
    print("  Training forward pass (with teacher forcing):")
    print("  " + "-" * 50)

    # Simulate batch: 8 equations of length 15, targets of length 7
    src = torch.randint(3, vocab_size, (8, 15))
    target = torch.randint(3, vocab_size, (8, 7))

    model.train()
    outputs = model(src, target=target, teacher_forcing_ratio=0.5)

    print(f"    Source (equations): {src.shape}")
    print(f"    Target (solutions): {target.shape}")
    print(f"    Output (logits):    {outputs.shape}")
    print(f"      → {outputs.shape[1]} steps × {outputs.shape[2]} token scores")
    print()

    # ---- Verify gradients flow ----
    print("  Gradient flow check:")
    print("  " + "-" * 50)

    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    loss = loss_fn(outputs.view(-1, vocab_size), target.view(-1))
    loss.backward()

    enc_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.encoder.parameters())
    dec_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.decoder.parameters())

    print(f"    Loss: {loss.item():.4f}")
    print(f"    Encoder gradients: {'✓ flowing' if enc_grad else '✗ NOT flowing'}")
    print(f"    Decoder gradients: {'✓ flowing' if dec_grad else '✗ NOT flowing'}")
    print()

    # ---- Inference forward pass ----
    print("  Inference forward pass (no teacher forcing):")
    print("  " + "-" * 50)

    model.eval()
    with torch.no_grad():
        predictions, logits = model.solve(src)

    print(f"    Source (equations):  {src.shape}")
    print(f"    Predictions:        {predictions.shape}")
    print(f"    Logits:             {logits.shape}")
    print(f"    Sample prediction:  {predictions[0].tolist()}")
    print()

    # ---- Now test with REAL data ----
    print("  Test with real equation data:")
    print("  " + "-" * 50)

    from vocab import build_vocab_from_data
    vocab = build_vocab_from_data("algebra/data/equations.txt")

    # Encode a real equation
    eq_text = "2x + 3 = 7"
    eq_encoded = vocab.encode_with_eos(eq_text)
    eq_tensor = torch.tensor([eq_encoded])

    with torch.no_grad():
        preds, _ = model.solve(eq_tensor)

    pred_text = vocab.decode_until_eos(preds[0].tolist())

    print(f"    Equation: \"{eq_text}\"")
    print(f"    Encoded:  {eq_encoded}")
    print(f"    Predicted: {preds[0].tolist()}")
    print(f"    Decoded:  \"{pred_text}\"")
    print(f"    (Random output expected — model is untrained!)")

    print()
    print("  " + "=" * 50)
    print("  Seq2Seq model ready for training!")


if __name__ == "__main__":
    main()
