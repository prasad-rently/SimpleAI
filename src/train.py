"""
train.py — Train the model on real data for one epoch.

PURPOSE:
    This is where ALL previous steps come together for the first time:
    - Step 03: Training data (data/input.txt)
    - Step 04: Vocabulary (text ↔ numbers)
    - Step 05: Dataset (training pairs)
    - Step 06: DataLoader (batched data)
    - Step 07: Model (neural network)
    - Step 08: Loss function + Optimizer

    We wire them all up and run one EPOCH — one complete pass through
    every batch of training data. After this, the model will have
    updated its weights 7 times (once per batch) and should show
    measurable improvement.

WHAT IS AN EPOCH?
    One epoch = processing EVERY training example exactly once.
    With our DataLoader:
      - 124 examples total
      - Batch size 16
      - 7 complete batches per epoch (112 examples, 12 dropped)
      - The model's weights update 7 times during one epoch

    Training typically runs for many epochs (e.g., 50-200).
    More epochs = more chances to learn patterns, but too many
    can lead to overfitting (memorizing instead of learning).

THE TRAINING FLOW:
    ┌──────────────────────────────────────────────────────────┐
    │  for each batch in DataLoader:                           │
    │    1. inputs, targets = batch          (get data)        │
    │    2. logits, _ = model(inputs)        (forward pass)    │
    │    3. loss = loss_fn(logits, targets)  (measure error)   │
    │    4. loss.backward()                  (compute grads)   │
    │    5. optimizer.step()                 (update weights)  │
    │    6. optimizer.zero_grad()            (clear grads)     │
    │    7. print loss                       (monitor)         │
    │                                                          │
    │  → After all batches: report average loss for the epoch  │
    └──────────────────────────────────────────────────────────┘

WHAT THIS FILE PROVIDES:
    1. train_one_epoch() — the core training function
    2. Full pipeline: load data → build vocab → create dataset →
       create dataloader → create model → train one epoch

INPUT:  data/input.txt
OUTPUT: Printed loss per batch and epoch summary

Usage:
    python src/train.py
"""

import torch
import torch.nn as nn

from vocabulary import Vocabulary
from dataset import TextDataset, create_dataloader
from model import TinyLanguageModel


def train_one_epoch(model, dataloader, loss_fn, optimizer, vocab_size, epoch_num=0):
    """
    Train the model for one complete epoch (one pass through all data).

    This function runs the 5-line training cycle (from Step 08) on
    every batch from the DataLoader. It tracks and reports the loss
    so you can see whether the model is improving.

    THE TRAINING LOOP EXPLAINED:
        The outer loop iterates over batches from the DataLoader:
          Batch 0: examples 0-15 (shuffled)
          Batch 1: examples 16-31 (shuffled)
          ...
          Batch 6: examples 96-111 (shuffled)

        For EACH batch, we run the 5-step learning cycle:
          1. Forward pass → get predictions
          2. Compute loss → measure how wrong
          3. Backward pass → compute gradients
          4. Optimizer step → adjust weights
          5. Zero gradients → clean up

        After all 7 batches, we return the average loss across all batches.
        This average is what gets plotted in the training curve (Step 11).

    WHAT "model.train()" DOES:
        PyTorch models have two modes:
        - model.train(): enables training-specific behaviors (dropout, etc.)
        - model.eval():  disables them for inference/generation
        Our simple RNN doesn't have dropout, but it's good practice to
        always call model.train() before training. It's a convention
        that prevents subtle bugs in more complex models.

    PARAMETERS:
        model (TinyLanguageModel): The model to train
        dataloader (DataLoader):   Batched training data
        loss_fn (CrossEntropyLoss): Measures prediction errors
        optimizer (Adam):           Updates weights using gradients
        vocab_size (int):           Number of characters (for reshaping)
        epoch_num (int):            Current epoch number (for display)

    RETURNS:
        float: Average loss across all batches in this epoch.
               Used for tracking progress and plotting the training curve.

    EXAMPLE:
        >>> avg_loss = train_one_epoch(model, dataloader, loss_fn, optimizer, 48, epoch_num=0)
        >>> print(f"Epoch 0 average loss: {avg_loss:.4f}")
        Epoch 0 average loss: 3.52
    """

    # ---- Set the model to training mode ----
    # This is a PyTorch convention. It tells the model we're about to
    # train, so it should enable any training-specific behaviors.
    # For our simple model this doesn't change anything, but always
    # do it — it prevents bugs when models have layers like Dropout
    # or BatchNorm that behave differently during training vs inference.
    model.train()

    # ---- Track loss across batches ----
    # We accumulate the total loss and count batches to compute the average.
    # The average loss is more meaningful than any single batch's loss
    # because individual batches can be noisy (some batches contain
    # harder examples than others).
    total_loss = 0.0
    num_batches = 0

    # ---- Loop through all batches ----
    # enumerate() gives us (batch_index, batch_data) pairs.
    # Each batch_data is a tuple of (input_tensor, target_tensor).
    #
    # With batch_size=16 and 124 examples:
    #   batch_idx 0: inputs shape (16, 50), targets shape (16, 50)
    #   batch_idx 1: inputs shape (16, 50), targets shape (16, 50)
    #   ...
    #   batch_idx 6: inputs shape (16, 50), targets shape (16, 50)
    #   (Total: 7 batches × 16 examples = 112 examples per epoch)

    for batch_idx, (inputs, targets) in enumerate(dataloader):

        # ============================================================
        # STEP 1: FORWARD PASS
        # ============================================================
        # Run the input through the model to get prediction scores.
        # model(inputs) calls model.forward(inputs) internally.
        #
        # INPUT:  inputs shape (16, 50) — batch of character indices
        # OUTPUT: logits shape (16, 50, 48) — prediction scores
        #         hidden — the RNN's final hidden state (we ignore it
        #                  during training with _ since we don't need
        #                  to carry context between batches)
        logits, _ = model(inputs)

        # ============================================================
        # STEP 2: COMPUTE LOSS
        # ============================================================
        # Reshape and compute how wrong the predictions are.
        #
        # CrossEntropyLoss expects:
        #   predictions: (N, C) where N = total predictions, C = classes
        #   targets:     (N,)   where each value is the correct class
        #
        # logits.view(-1, vocab_size): (16, 50, 48) → (800, 48)
        #   -1 means "figure out this dimension": 16 × 50 = 800
        #
        # targets.view(-1): (16, 50) → (800,)
        #   Flatten to a 1D list of 800 correct answers
        loss = loss_fn(logits.view(-1, vocab_size), targets.view(-1))

        # ============================================================
        # STEP 3: BACKWARD PASS (backpropagation)
        # ============================================================
        # Compute the gradient for every weight in the model.
        # PyTorch traces back from the loss through every operation
        # and computes: "how should each weight change to reduce the loss?"
        #
        # After this call, every parameter tensor in the model has a
        # .grad attribute filled with its gradient.
        loss.backward()

        # ============================================================
        # STEP 4: OPTIMIZER STEP
        # ============================================================
        # Use the gradients to adjust all 248,880 weights.
        # Each weight moves in the direction that reduces the loss.
        # The amount of movement is controlled by the learning rate.
        optimizer.step()

        # ============================================================
        # STEP 5: ZERO GRADIENTS
        # ============================================================
        # Clear all gradients so they don't accumulate across batches.
        # PyTorch adds gradients by default, so without this line,
        # batch 1's gradients would pile on top of batch 0's.
        optimizer.zero_grad()

        # ---- Track and report progress ----
        # .item() extracts the Python float from the loss tensor.
        # We track the total to compute the average at the end.
        batch_loss = loss.item()
        total_loss += batch_loss
        num_batches += 1

        # Print every batch so you can watch the loss change in real-time.
        # In larger projects you'd print less frequently to avoid spam.
        print(f"  Epoch {epoch_num:3d} | Batch {batch_idx}/{len(dataloader)-1} | "
              f"Loss: {batch_loss:.4f}")

    # ---- Compute average loss for the epoch ----
    # Average = total / count. This smooths out batch-to-batch noise
    # and gives a single number representing the epoch's performance.
    avg_loss = total_loss / num_batches

    return avg_loss


def main():
    """
    Main function — wires up the full pipeline and trains for one epoch.

    FLOW:
        1. Load data/input.txt
        2. Build Vocabulary (Step 04)
        3. Create TextDataset (Step 05)
        4. Create DataLoader (Step 06)
        5. Create TinyLanguageModel (Step 07)
        6. Create loss function + optimizer (Step 08)
        7. Train for one epoch (Step 09) ← NEW
        8. Report results
    """

    # ==================================================================
    # SETUP: Wire up the full pipeline
    # ==================================================================
    print("=" * 60)
    print("STEP 09: TRAINING — ONE EPOCH")
    print("=" * 60)
    print()

    # ---- Step A: Load data ----
    filepath = "data/input.txt"
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Data loaded: {len(text)} characters from {filepath}")

    # ---- Step B: Build vocabulary ----
    vocab = Vocabulary(text)
    print(f"Vocabulary: {vocab.vocab_size} unique characters")

    # ---- Step C: Create dataset ----
    seq_length = 50
    dataset = TextDataset(text, vocab, seq_length=seq_length)
    print(f"Dataset: {len(dataset)} training examples (seq_length={seq_length})")

    # ---- Step D: Create dataloader ----
    batch_size = 16
    dataloader = create_dataloader(dataset, batch_size=batch_size, shuffle=True)
    print(f"DataLoader: {len(dataloader)} batches of {batch_size}")

    # ---- Step E: Create model ----
    model = TinyLanguageModel(vocab_size=vocab.vocab_size)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: TinyLanguageModel with {total_params:,} parameters")

    # ---- Step F: Create loss function and optimizer ----
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print(f"Loss: CrossEntropyLoss | Optimizer: Adam (lr=0.001)")
    print()

    # ==================================================================
    # TRAINING: One epoch
    # ==================================================================
    # Expected: loss starts around 3.87 (random) and decreases by the
    # end of the epoch. Even with just 7 batches, you should see the
    # loss drop noticeably.

    import math
    random_baseline = -math.log(1.0 / vocab.vocab_size)
    print(f"Random baseline loss: {random_baseline:.4f}")
    print(f"(If the model is learning, loss should drop below this)")
    print()

    print(f"--- Training epoch 0 ({len(dataloader)} batches) ---")
    avg_loss = train_one_epoch(model, dataloader, loss_fn, optimizer,
                               vocab.vocab_size, epoch_num=0)

    print()
    print(f"  Epoch 0 complete!")
    print(f"    Average loss : {avg_loss:.4f}")
    print(f"    Random base  : {random_baseline:.4f}")
    print(f"    Improvement  : {random_baseline - avg_loss:+.4f}")
    print()

    if avg_loss < random_baseline:
        print(f"  The model is learning! Loss dropped below random baseline.")
    else:
        print(f"  Loss is still near random — one epoch isn't always enough.")
    print()

    # ==================================================================
    # WHAT COMES NEXT
    # ==================================================================
    print("=" * 60)
    print("WHAT COMES NEXT (Step 10)")
    print("=" * 60)
    print("""
We just trained for ONE epoch (7 batches, 7 weight updates).
The loss dropped a bit, but the model is far from good.

In Step 10, we'll train for MANY epochs (e.g., 50-100) and:
  - Track the loss at each epoch
  - Save the trained model to disk (outputs/model.pth)
  - Watch the loss curve descend over time

More epochs = more learning = better text generation.
""")


# ======================================================================
# ENTRY POINT
# ======================================================================
if __name__ == "__main__":
    main()
