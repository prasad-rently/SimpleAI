"""
train.py — Training loop for the algebra solver seq2seq model.

PURPOSE:
    Train the encoder-decoder model on 40,000 equations so it learns
    to solve linear equations like "2x + 3 = 7" → "x = 2".

    This is where the model goes from producing random gibberish
    (untrained) to correctly solving 90%+ of equations (trained).

HOW TRAINING WORKS (seq2seq edition):
    The training loop is similar to the text generator, but with
    important differences:

    TEXT GENERATOR:
        1. Feed text sequence → predict next character at each position
        2. One model, one loss per sequence

    SEQ2SEQ:
        1. Encoder reads equation → context vector
        2. Decoder generates solution using context + teacher forcing
        3. Loss computed on decoder output vs correct solution
        4. Gradients flow backward through BOTH decoder AND encoder
        5. Both networks updated simultaneously

    ┌─────────────────────────────────────────────────────────┐
    │  For each batch of 64 equations:                        │
    │                                                         │
    │  1. FORWARD PASS                                        │
    │     equation → Encoder → context → Decoder → prediction │
    │                                                         │
    │  2. LOSS                                                │
    │     Compare prediction vs correct answer (ignore <PAD>) │
    │                                                         │
    │  3. BACKWARD PASS                                       │
    │     Compute gradients for ALL parameters                │
    │     (both encoder and decoder)                          │
    │                                                         │
    │  4. GRADIENT CLIPPING                                   │
    │     Cap gradients at 1.0 to prevent exploding gradients │
    │                                                         │
    │  5. OPTIMIZER STEP                                      │
    │     Update weights to reduce loss                       │
    └─────────────────────────────────────────────────────────┘

TEACHER FORCING SCHEDULE:
    We gradually reduce teacher forcing during training:

    Epoch  1-10:  ratio = 1.0   → always feed correct answer
    Epoch 11-20:  ratio = 0.75  → 75% correct, 25% own prediction
    Epoch 21-30:  ratio = 0.5   → 50/50 mix

    This is like removing training wheels gradually:
    - Early: model needs full help to learn the output format
    - Middle: model starts using its own predictions sometimes
    - Late: model practices generating on its own (closer to inference)

GRADIENT CLIPPING:
    RNNs can suffer from "exploding gradients" — when gradients become
    extremely large during backpropagation, causing wild weight updates
    that destabilize training. Gradient clipping caps the total gradient
    norm at 1.0, preventing this.

    Think of it like speed limiting: the model can only change its
    weights by a bounded amount per step, preventing overshooting.

WHAT THIS FILE PROVIDES:
    1. get_teacher_forcing_ratio() — schedule for teacher forcing
    2. train_one_epoch()           — one pass through training data
    3. evaluate_accuracy()         — test accuracy on held-out data
    4. train()                     — full training loop
    5. save_model()                — save model weights
    6. save_vocab()                — save vocabulary
    7. main()                      — run everything

INPUT:  algebra/data/equations.txt
OUTPUT: algebra/outputs/model.pth, vocab.pth, loss_history.pth

Usage:
    PYTHONPATH=algebra/src python algebra/src/train.py
"""

import os
import time
import torch
import torch.nn as nn

from vocab import build_vocab_from_data, PAD_IDX, EOS_IDX
from dataset import create_dataloaders
from seq2seq import create_model


# ====================================================================
# TEACHER FORCING SCHEDULE
# ====================================================================

def get_teacher_forcing_ratio(epoch, num_epochs):
    """
    Compute teacher forcing ratio for a given epoch.

    Linearly decreases from 1.0 to 0.5 over training:
        epoch 0:          ratio = 1.0
        epoch num_epochs:  ratio = 0.5

    PARAMETERS:
        epoch (int):      Current epoch (0-indexed)
        num_epochs (int): Total number of epochs

    RETURNS:
        float: Teacher forcing ratio between 0.5 and 1.0
    """

    # Linear decrease from 1.0 to 0.3
    ratio = 1.0 - 0.7 * (epoch / max(num_epochs - 1, 1))
    return max(ratio, 0.3)


# ====================================================================
# TRAINING ONE EPOCH
# ====================================================================

def train_one_epoch(model, dataloader, loss_fn, optimizer, vocab_size,
                    teacher_forcing_ratio, grad_clip=1.0):
    """
    Train for one epoch (one full pass through the training data).

    PROCESS for each batch:
        1. Forward pass: equation → encoder → decoder → predictions
        2. Compute loss (predictions vs targets, ignoring <PAD>)
        3. Backward pass: compute gradients
        4. Clip gradients (prevent exploding gradients)
        5. Optimizer step: update weights
        6. Zero gradients for next batch

    PARAMETERS:
        model (Seq2Seq):       The model to train
        dataloader (DataLoader): Training data batches
        loss_fn (CrossEntropyLoss): Loss function (ignores PAD)
        optimizer (Adam):      Optimizer
        vocab_size (int):      Vocabulary size (for reshaping)
        teacher_forcing_ratio (float): Teacher forcing probability
        grad_clip (float):     Maximum gradient norm (1.0)

    RETURNS:
        float: Average loss over all batches in this epoch
    """

    model.train()
    total_loss = 0.0
    num_batches = 0

    for enc_input, dec_input, dec_target in dataloader:
        # ---- Forward pass ----
        # enc_input: (batch, src_len) — the equations
        # dec_target: (batch, target_len) — what decoder should predict
        # We pass dec_target as the target for teacher forcing
        outputs = model(
            enc_input, target=dec_target,
            teacher_forcing_ratio=teacher_forcing_ratio
        )

        # ---- Compute loss ----
        # Reshape for CrossEntropyLoss:
        #   outputs: (batch, seq_len, vocab_size) → (batch*seq_len, vocab_size)
        #   targets: (batch, seq_len) → (batch*seq_len,)
        # CrossEntropyLoss(ignore_index=PAD_IDX) skips padding positions
        output_flat = outputs.view(-1, vocab_size)
        target_flat = dec_target.view(-1)
        loss = loss_fn(output_flat, target_flat)

        # ---- Backward pass ----
        loss.backward()

        # ---- Gradient clipping ----
        # Prevents exploding gradients in RNNs.
        # clip_grad_norm_ modifies gradients in-place, capping
        # the total L2 norm of all parameters at grad_clip.
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # ---- Optimizer step ----
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


# ====================================================================
# EVALUATION — accuracy on test set
# ====================================================================

def evaluate_accuracy(model, dataloader, vocab):
    """
    Compute exact-match accuracy on a dataset.

    For each equation, we run inference (no teacher forcing) and
    check if the predicted solution exactly matches the correct one.

    EXACT MATCH means every character must be identical:
        "x = 2" == "x = 2"   ✓
        "x = 3" == "x = 2"   ✗ (wrong number)
        "x = 2 " == "x = 2"  ✗ (trailing space)

    PARAMETERS:
        model (Seq2Seq):       The trained model
        dataloader (DataLoader): Test data batches
        vocab (AlgebraVocab):  Vocabulary for decoding

    RETURNS:
        float: Accuracy between 0.0 and 1.0
    """

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for enc_input, dec_input, dec_target in dataloader:
            # ---- Run inference ----
            predictions, _ = model.solve(enc_input)

            # ---- Compare each prediction to target ----
            for i in range(enc_input.shape[0]):
                # Decode predicted and target sequences
                pred_text = vocab.decode_until_eos(predictions[i].tolist())
                target_text = vocab.decode_until_eos(dec_target[i].tolist())

                if pred_text == target_text:
                    correct += 1
                total += 1

    return correct / total if total > 0 else 0.0


# ====================================================================
# FULL TRAINING LOOP
# ====================================================================

def train(model, train_loader, test_loader, vocab, num_epochs=30,
          learning_rate=0.001, grad_clip=1.0, print_every=5,
          eval_every=5):
    """
    Full training loop with teacher forcing schedule and evaluation.

    PROCESS:
        For each epoch:
        1. Compute teacher forcing ratio (decreasing over time)
        2. Train one epoch (forward, loss, backward, update)
        3. Every eval_every epochs: check accuracy on test set
        4. Print progress

    PARAMETERS:
        model (Seq2Seq):         The model to train
        train_loader (DataLoader): Training data
        test_loader (DataLoader):  Test data (for evaluation)
        vocab (AlgebraVocab):    Vocabulary
        num_epochs (int):        Number of training epochs (30)
        learning_rate (float):   Adam learning rate (0.001)
        grad_clip (float):       Gradient clip value (1.0)
        print_every (int):       Print progress every N epochs
        eval_every (int):        Evaluate accuracy every N epochs

    RETURNS:
        dict: Training history with 'losses', 'accuracies', 'epochs'
    """

    vocab_size = vocab.vocab_size

    # ---- Loss function ----
    # ignore_index=PAD_IDX tells CrossEntropyLoss to skip padding
    # positions when computing the loss. Without this, the model
    # would be penalized for not predicting <PAD> correctly, which
    # is meaningless.
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # ---- Optimizer ----
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ---- Learning rate scheduler ----
    # Reduces learning rate by half every 15 epochs.
    # Early epochs use large steps (fast learning), later epochs
    # use small steps (fine-tuning). This helps the model converge
    # to a better solution than using a fixed learning rate.
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=15, gamma=0.5
    )

    # ---- History tracking ----
    history = {
        'losses': [],
        'accuracies': [],
        'tf_ratios': [],
    }

    print(f"  Starting training: {num_epochs} epochs, "
          f"{len(train_loader)} batches/epoch")
    print()

    start_time = time.time()

    for epoch in range(num_epochs):
        # ---- Get teacher forcing ratio ----
        tf_ratio = get_teacher_forcing_ratio(epoch, num_epochs)

        # ---- Train one epoch ----
        avg_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer,
            vocab_size, tf_ratio, grad_clip
        )

        history['losses'].append(avg_loss)
        history['tf_ratios'].append(tf_ratio)

        # ---- Step the learning rate scheduler ----
        scheduler.step()

        # ---- Evaluate accuracy ----
        if (epoch + 1) % eval_every == 0 or epoch == num_epochs - 1:
            accuracy = evaluate_accuracy(model, test_loader, vocab)
            history['accuracies'].append((epoch, accuracy))

            elapsed = time.time() - start_time
            print(f"  Epoch {epoch:>3}/{num_epochs} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Accuracy: {accuracy:.1%} | "
                  f"TF: {tf_ratio:.2f} | "
                  f"Time: {elapsed:.0f}s")
        elif (epoch + 1) % print_every == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch:>3}/{num_epochs} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"TF: {tf_ratio:.2f} | "
                  f"Time: {elapsed:.0f}s")

    total_time = time.time() - start_time
    print()
    print(f"  Training complete in {total_time:.1f}s "
          f"({total_time/60:.1f} minutes)")

    final_accuracy = history['accuracies'][-1][1] if history['accuracies'] else 0
    print(f"  Final loss:     {history['losses'][-1]:.4f}")
    print(f"  Final accuracy: {final_accuracy:.1%}")

    return history


# ====================================================================
# SAVE
# ====================================================================

def save_model(model, filepath="algebra/outputs/model.pth"):
    """
    Save the trained model weights to disk.

    We save model.state_dict() (just the learned weights), not the
    entire model object. This is the PyTorch best practice because:
    - Smaller file size
    - More portable (doesn't depend on exact class definition)
    - Safer (weights_only=True when loading)

    PARAMETERS:
        model (Seq2Seq): The trained model
        filepath (str):   Where to save
    """

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)

    size_kb = os.path.getsize(filepath) / 1024
    print(f"  Model saved to {filepath} ({size_kb:.1f} KB)")


def save_vocab(vocab, filepath="algebra/outputs/vocab.pth"):
    """
    Save the vocabulary to disk.

    We save the token list and mappings so we can reconstruct
    the vocabulary during inference without re-reading the data file.

    PARAMETERS:
        vocab (AlgebraVocab): The vocabulary
        filepath (str):        Where to save
    """

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save({
        'tokens': vocab.tokens,
        'token_to_idx': vocab.token_to_idx,
        'idx_to_token': vocab.idx_to_token,
        'vocab_size': vocab.vocab_size,
    }, filepath)

    size_kb = os.path.getsize(filepath) / 1024
    print(f"  Vocabulary saved to {filepath} ({size_kb:.1f} KB)")


def save_history(history, filepath="algebra/outputs/loss_history.pth"):
    """
    Save training history (losses and accuracies) for plotting.

    PARAMETERS:
        history (dict): Training history from train()
        filepath (str):  Where to save
    """

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(history, filepath)
    print(f"  History saved to {filepath}")


# ====================================================================
# MAIN
# ====================================================================

def main():
    """
    Main function — build everything, train, save.

    FLOW:
        1. Build vocabulary from data
        2. Create train/test dataloaders
        3. Create seq2seq model
        4. Train for 30 epochs
        5. Save model, vocabulary, and history
    """

    print("=" * 60)
    print("STEP 08: TRAINING THE ALGEBRA SOLVER")
    print("=" * 60)
    print()

    # ---- Build vocabulary ----
    print("Building vocabulary...")
    vocab = build_vocab_from_data("algebra/data/equations.txt")
    print(f"  Vocabulary: {vocab.vocab_size} tokens")
    print()

    # ---- Create dataloaders ----
    print("Creating dataloaders...")
    train_loader, test_loader = create_dataloaders(
        vocab, batch_size=64, train_ratio=0.8
    )
    print()

    # ---- Create model ----
    print("Creating model...")
    model = create_model(vocab_size=vocab.vocab_size)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    print()

    # ---- Train ----
    print("Training...")
    print("-" * 60)
    history = train(
        model, train_loader, test_loader, vocab,
        num_epochs=50,
        learning_rate=0.002,
        grad_clip=1.0,
        print_every=10,
        eval_every=10,
    )
    print("-" * 60)
    print()

    # ---- Save ----
    print("Saving...")
    save_model(model)
    save_vocab(vocab)
    save_history(history)

    # ---- Quick demo from test set ----
    print()
    print("Quick demo — solving equations from test set:")
    print("-" * 60)

    model.eval()
    demo_correct = 0
    demo_total = 0
    for enc_input, dec_input, dec_target in test_loader:
        with torch.no_grad():
            preds, _ = model.solve(enc_input)
        for i in range(min(10, enc_input.shape[0])):
            eq_text = vocab.decode_until_eos(enc_input[i].tolist())
            pred_text = vocab.decode_until_eos(preds[i].tolist())
            target_text = vocab.decode_until_eos(dec_target[i].tolist())
            match = pred_text == target_text
            demo_correct += match
            demo_total += 1
            status = "✓" if match else "✗"
            print(f"  {eq_text:<30} → {pred_text:<12} "
                  f"(answer: {target_text:<12}) {status}")
        break
    print(f"\n  Demo: {demo_correct}/{demo_total} correct")

    print()
    print("=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
