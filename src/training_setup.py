"""
training_setup.py — The loss function and optimizer: how the model learns.

PURPOSE:
    The model from Step 07 exists but knows nothing — its weights are random.
    To TEACH it, we need two things:

    1. LOSS FUNCTION — measures how wrong the model is.
       "You predicted 'z' but the answer was 'e'. That's very wrong. Score: 3.87"
       Gives us a single number (the "loss"). Lower = better.

    2. OPTIMIZER — figures out how to adjust each weight to reduce the loss.
       "If you increase weight #3 by 0.01, the loss will decrease."
       Then it actually makes those adjustments.

    Together they form a feedback loop:

    ┌──────────────────────────────────────────────────────────────────┐
    │                    THE LEARNING LOOP                             │
    │                                                                  │
    │   ┌─────────┐  predictions  ┌──────────┐  loss   ┌───────────┐ │
    │   │  MODEL  │ ────────────▶ │   LOSS   │ ──────▶ │ OPTIMIZER │ │
    │   │(weights)│               │ FUNCTION │         │  (Adam)   │ │
    │   └────▲────┘               └──────────┘         └─────┬─────┘ │
    │        │                                               │       │
    │        └───────── adjust weights ──────────────────────┘       │
    │                                                                  │
    │   Repeat thousands of times → loss decreases → model improves   │
    └──────────────────────────────────────────────────────────────────┘

THE LEARNING PROCESS IN PLAIN ENGLISH:
    1. Show the model some text:     "The only way to do grea"
    2. Model predicts next chars:     "Zq!xmf..." (random garbage — untrained)
    3. Loss function scores it:       "That's 3.87 wrong" (high = bad)
    4. Optimizer adjusts weights:     tweaks all 248,880 numbers slightly
    5. Show the same text again:      model predicts "he only way to do great" (better!)
    6. Loss function scores it:       "That's 2.14 wrong" (lower = improving!)
    7. Repeat steps 1-6 thousands of times...
    8. Eventually loss reaches ~1.5:  model generates decent text

WHAT THIS FILE PROVIDES:
    1. Explanation and demo of CrossEntropyLoss
    2. Explanation and demo of Adam optimizer
    3. A complete ONE-STEP learning demo: forward → loss → backward → update
    4. Before/after comparison showing weights actually change

INPUT:  Model from Step 07, fake batch of data
OUTPUT: Demos showing loss computation, gradient flow, and weight updates

Usage:
    python src/training_setup.py
"""

import torch
import torch.nn as nn

from model import TinyLanguageModel


def demonstrate_loss_function(model, vocab_size):
    """
    Show how CrossEntropyLoss measures the model's mistakes.

    WHAT IS CROSS-ENTROPY LOSS?
        It measures how far the model's predictions are from the correct
        answers. Specifically, it:
        1. Takes the model's 48 raw scores (logits) for each position
        2. Converts them to probabilities using "softmax"
           (all values become 0-1 and sum to 1)
        3. Looks at the probability assigned to the CORRECT character
        4. Returns -log(that probability)

        If the model is confident and correct:
          probability of correct char = 0.95
          loss = -log(0.95) = 0.05  ← very low (good!)

        If the model is confident and WRONG:
          probability of correct char = 0.01
          loss = -log(0.01) = 4.6   ← very high (bad!)

        If the model is clueless (uniform random):
          probability of correct char = 1/48 = 0.021
          loss = -log(0.021) = 3.87  ← the "random baseline"

    WHY CROSS-ENTROPY (not something simpler)?
        Cross-entropy is mathematically ideal for classification tasks
        (predicting which of N classes is correct). It:
        - Penalizes confident wrong answers heavily
        - Rewards confident correct answers
        - Has smooth gradients that help the optimizer

    PARAMETERS:
        model (TinyLanguageModel): The model to test
        vocab_size (int):          Number of characters

    OUTPUT: Printed loss values with explanation.
    """

    print("=" * 60)
    print("LOSS FUNCTION: Measuring how wrong the model is")
    print("=" * 60)
    print()

    # ---- Create the loss function ----
    # nn.CrossEntropyLoss() creates a loss function object.
    # We call it like a function: loss_value = loss_fn(predictions, targets)
    #
    # IMPORTANT SHAPE REQUIREMENT:
    #   predictions shape: (N, C) where N = total predictions, C = num classes
    #   targets shape:     (N,)   where each value is the correct class index
    #
    #   For us: N = batch_size × seq_length, C = vocab_size (48)
    #   So we need to reshape our (16, 50, 48) logits to (800, 48)
    #   and our (16, 50) targets to (800,)
    loss_fn = nn.CrossEntropyLoss()

    print("  Loss function: CrossEntropyLoss")
    print("  - Takes model's 48 scores + correct answer")
    print("  - Returns a single number: how wrong the model is")
    print("  - Lower = better. Zero = perfect (never happens in practice)")
    print()

    # ---- Create fake data ----
    batch_size, seq_length = 4, 50
    fake_input = torch.randint(0, vocab_size, (batch_size, seq_length))
    fake_target = torch.randint(0, vocab_size, (batch_size, seq_length))

    # ---- Run forward pass ----
    logits, _ = model(fake_input)

    print(f"  Forward pass:")
    print(f"    Input shape:  {fake_input.shape}  (batch_size, seq_length)")
    print(f"    Logits shape: {logits.shape}  (batch_size, seq_length, vocab_size)")
    print(f"    Target shape: {fake_target.shape}  (batch_size, seq_length)")
    print()

    # ---- Reshape for loss computation ----
    # CrossEntropyLoss expects:
    #   predictions: (N, C) — N total predictions, C classes
    #   targets:     (N,)   — N correct class indices
    #
    # We have:
    #   logits: (4, 50, 48) → reshape to (200, 48)
    #   target: (4, 50)     → reshape to (200,)
    #
    # .view(-1, vocab_size) means "flatten to 2D, keeping last dim as vocab_size"
    # -1 tells PyTorch to figure out that dimension automatically:
    #   4 × 50 = 200, so shape becomes (200, 48)
    #
    # .view(-1) flattens completely: (4, 50) → (200,)
    logits_flat = logits.view(-1, vocab_size)
    targets_flat = fake_target.view(-1)

    print(f"  Reshape for loss computation:")
    print(f"    logits: {logits.shape} → {logits_flat.shape}  (flatten batch × seq)")
    print(f"    target: {fake_target.shape} → {targets_flat.shape}  (flatten batch × seq)")
    print()

    # ---- Compute the loss ----
    loss = loss_fn(logits_flat, targets_flat)

    print(f"  Loss value: {loss.item():.4f}")
    print()

    # ---- Explain the value ----
    # With vocab_size=48 and a random (untrained) model,
    # the expected loss is -log(1/48) ≈ 3.87
    import math
    random_baseline = -math.log(1.0 / vocab_size)
    print(f"  INTERPRETING THE LOSS:")
    print(f"    Random baseline : {random_baseline:.4f}  (expected for untrained model)")
    print(f"    Our model's loss: {loss.item():.4f}")
    print(f"    Difference      : {loss.item() - random_baseline:+.4f}")
    print()
    print(f"    If loss ≈ {random_baseline:.1f}  → model is guessing randomly (untrained)")
    print(f"    If loss ≈ 2.0   → model has learned some patterns")
    print(f"    If loss ≈ 1.0   → model has learned well")
    print(f"    If loss → 0.0   → model has memorized the data (overfitting!)")
    print()

    return loss_fn


def demonstrate_optimizer(model):
    """
    Show how the Adam optimizer adjusts weights to reduce the loss.

    WHAT IS AN OPTIMIZER?
        After computing the loss, we know HOW WRONG the model is.
        But we need to figure out HOW TO FIX IT — which of the 248,880
        weights should go up, which should go down, and by how much.

        The optimizer does this using "gradients":
        1. loss.backward() computes the gradient for each weight
           (gradient = "if I increase this weight slightly, how much
           does the loss change?")
        2. optimizer.step() uses these gradients to adjust weights
           in the direction that REDUCES the loss.

    WHY ADAM (not plain gradient descent)?
        Adam is an enhanced optimizer that:
        - Adapts the learning rate for each parameter individually
        - Uses momentum (remembers which direction it was going)
        - Handles noisy gradients well

        It's the most popular optimizer in deep learning — used by
        most research papers and industry models. "When in doubt, use Adam."

    WHAT IS LEARNING RATE?
        The learning rate (lr) controls how big each weight adjustment is:
        - lr=0.01:   large steps — learns fast but might overshoot
        - lr=0.001:  medium steps — good default for Adam
        - lr=0.0001: tiny steps — learns slowly but precisely

        Think of it like adjusting a radio dial:
        - Big turns (high lr): quickly get close to the right station
        - Tiny turns (low lr): fine-tune to get crystal clear reception

    PARAMETERS:
        model (TinyLanguageModel): The model whose weights to optimize

    OUTPUT: Printed optimizer configuration and explanation.
    """

    print("=" * 60)
    print("OPTIMIZER: How weights get adjusted")
    print("=" * 60)
    print()

    # ---- Create the optimizer ----
    # model.parameters() gives Adam access to all 248,880 weights.
    # lr=0.001 is the learning rate — how big each adjustment step is.
    #
    # Adam needs to track extra information per parameter:
    # - Running mean of gradients (momentum)
    # - Running mean of squared gradients (adaptive learning rate)
    # This is why Adam uses ~3x the memory of the model itself.
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"  Optimizer     : Adam")
    print(f"  Learning rate : {learning_rate}")
    print(f"  Parameters    : {sum(p.numel() for p in model.parameters()):,}")
    print()

    print("  WHAT ADAM DOES EACH STEP:")
    print("    1. Reads the gradient for each weight")
    print("       (computed by loss.backward())")
    print("    2. Updates its momentum estimates")
    print("       (which direction have gradients been going?)")
    print("    3. Computes an adaptive learning rate per weight")
    print("       (weights with large gradients get smaller steps)")
    print("    4. Adjusts each weight: weight -= lr * adjusted_gradient")
    print()

    return optimizer


def demonstrate_one_learning_step(model, loss_fn, optimizer, vocab_size):
    """
    Perform one complete learning step and show weights changing.

    This is the CORE of training — the atomic unit that gets repeated
    thousands of times. Every step:
        1. FORWARD  — run data through the model to get predictions
        2. LOSS     — measure how wrong the predictions are
        3. BACKWARD — compute gradients (which direction to adjust each weight)
        4. UPDATE   — optimizer adjusts weights using the gradients
        5. ZERO     — clear gradients for the next step

    PARAMETERS:
        model (TinyLanguageModel): The model to train
        loss_fn (nn.CrossEntropyLoss): The loss function
        optimizer (torch.optim.Adam): The optimizer
        vocab_size (int):              Number of characters

    OUTPUT: Printed before/after weights showing the model learned something.
    """

    print("=" * 60)
    print("ONE LEARNING STEP: The complete forward → backward → update cycle")
    print("=" * 60)
    print()

    # ---- Snapshot weights BEFORE the step ----
    # We'll compare these to weights AFTER to prove they changed.
    # .clone() makes a copy so the original values are preserved.
    weight_before = model.output_layer.weight.data[0, :5].clone()

    print(f"  Output layer weights BEFORE (first 5 of row 0):")
    print(f"    {weight_before.tolist()}")
    print()

    # ---- Create fake training data ----
    batch_size, seq_length = 4, 50
    fake_input = torch.randint(0, vocab_size, (batch_size, seq_length))
    fake_target = torch.randint(0, vocab_size, (batch_size, seq_length))

    # ============================================================
    # STEP 1: FORWARD PASS
    # ============================================================
    # Run input through the model to get predictions.
    # This is the same as what we did in Step 07.
    logits, _ = model(fake_input)

    print("  Step 1 — FORWARD PASS:")
    print(f"    Input:  {fake_input.shape} → Model → Logits: {logits.shape}")

    # ============================================================
    # STEP 2: COMPUTE LOSS
    # ============================================================
    # Measure how wrong the predictions are.
    # Reshape logits (4, 50, 48) → (200, 48) and targets (4, 50) → (200,)
    loss = loss_fn(logits.view(-1, vocab_size), fake_target.view(-1))

    print(f"  Step 2 — COMPUTE LOSS:")
    print(f"    Loss = {loss.item():.4f}  (how wrong the model is)")

    # ============================================================
    # STEP 3: BACKWARD PASS (compute gradients)
    # ============================================================
    # loss.backward() is where the MAGIC happens.
    # PyTorch traces back through EVERY operation that produced the loss
    # and computes a "gradient" for every weight in the model.
    #
    # A gradient tells us: "if I increase this weight by a tiny amount,
    # how much does the loss change?"
    #   - Positive gradient: increasing the weight INCREASES the loss (bad)
    #     → the optimizer should DECREASE this weight
    #   - Negative gradient: increasing the weight DECREASES the loss (good)
    #     → the optimizer should INCREASE this weight
    #   - Large gradient: this weight has a big effect on the loss
    #     → make a bigger adjustment
    #   - Small gradient: this weight barely matters right now
    #     → make a tiny adjustment
    #
    # This is called "backpropagation" — the key algorithm in all of deep learning.
    # Invented in the 1980s, it's what makes training neural networks possible.
    loss.backward()

    print(f"  Step 3 — BACKWARD PASS (backpropagation):")
    print(f"    Computed gradients for all {sum(p.numel() for p in model.parameters()):,} weights")

    # Show a sample gradient
    grad_sample = model.output_layer.weight.grad[0, :5]
    print(f"    Sample gradients (output layer, first 5 of row 0):")
    print(f"    {[f'{g:.6f}' for g in grad_sample.tolist()]}")

    # ============================================================
    # STEP 4: OPTIMIZER STEP (update weights)
    # ============================================================
    # The optimizer reads all the gradients and adjusts every weight
    # in the direction that reduces the loss.
    #
    # For each weight:
    #   new_weight = old_weight - learning_rate * gradient
    #   (simplified — Adam actually uses momentum and adaptive rates)
    optimizer.step()

    print(f"  Step 4 — OPTIMIZER STEP:")
    print(f"    Adjusted all 248,880 weights using their gradients")

    # ============================================================
    # STEP 5: ZERO GRADIENTS (cleanup for next step)
    # ============================================================
    # PyTorch ACCUMULATES gradients by default (adds new gradients
    # to existing ones). We must zero them out before the next step,
    # otherwise gradients from different batches would pile up.
    #
    # This is a common source of bugs in PyTorch:
    # Forgetting optimizer.zero_grad() → gradients accumulate →
    # model makes increasingly wild weight adjustments → training explodes.
    optimizer.zero_grad()

    print(f"  Step 5 — ZERO GRADIENTS:")
    print(f"    Cleared all gradients for the next step")
    print()

    # ---- Compare weights BEFORE and AFTER ----
    weight_after = model.output_layer.weight.data[0, :5]

    print(f"  PROOF THAT THE MODEL LEARNED SOMETHING:")
    print(f"    Weights BEFORE: {weight_before.tolist()}")
    print(f"    Weights AFTER:  {weight_after.tolist()}")
    print(f"    Changed?        {not torch.equal(weight_before, weight_after)} ← weights moved!")
    print()

    # Show the differences
    diffs = weight_after - weight_before
    print(f"    Weight changes: {[f'{d:+.6f}' for d in diffs.tolist()]}")
    print(f"    (tiny changes — each step nudges weights slightly)")
    print()

    # ---- Run ANOTHER forward pass to see if loss improved ----
    logits2, _ = model(fake_input)
    loss2 = loss_fn(logits2.view(-1, vocab_size), fake_target.view(-1))

    print(f"  DID THE LOSS IMPROVE?")
    print(f"    Loss before step: {loss.item():.4f}")
    print(f"    Loss after step:  {loss2.item():.4f}")
    print(f"    Change: {loss2.item() - loss.item():+.4f}")
    if loss2.item() < loss.item():
        print(f"    The model improved in just ONE step!")
    else:
        print(f"    (One step isn't always enough — it takes many steps to converge)")
    print()

    # ---- The full picture ----
    print("  THE COMPLETE TRAINING STEP:")
    print("  ┌─────────────────────────────────────────────────────────┐")
    print("  │  1. logits, _ = model(input)          # forward pass   │")
    print("  │  2. loss = loss_fn(logits, target)     # compute loss   │")
    print("  │  3. loss.backward()                    # compute grads  │")
    print("  │  4. optimizer.step()                   # update weights │")
    print("  │  5. optimizer.zero_grad()              # clear grads    │")
    print("  │                                                         │")
    print("  │  Repeat this for every batch, for many epochs.          │")
    print("  │  That's ALL training is. These 5 lines.                 │")
    print("  └─────────────────────────────────────────────────────────┘")
    print()


def main():
    """
    Main function — sets up loss + optimizer and demos one learning step.

    FLOW:
        1. Create the model (from Step 07)
        2. Create and explain the loss function (CrossEntropyLoss)
        3. Create and explain the optimizer (Adam)
        4. Run one complete learning step: forward → loss → backward → update
        5. Show that weights actually changed
    """

    vocab_size = 48

    print("=" * 60)
    print("STEP 08: LOSS FUNCTION + OPTIMIZER")
    print("=" * 60)
    print()
    print(f"Setting up the training tools for a {vocab_size}-character model...")
    print()

    # ---- Create the model ----
    model = TinyLanguageModel(vocab_size=vocab_size)

    # ---- Demo the loss function ----
    loss_fn = demonstrate_loss_function(model, vocab_size)

    # ---- Demo the optimizer ----
    optimizer = demonstrate_optimizer(model)

    # ---- Demo one complete learning step ----
    demonstrate_one_learning_step(model, loss_fn, optimizer, vocab_size)

    # ---- What comes next ----
    print("=" * 60)
    print("WHAT COMES NEXT (Step 09)")
    print("=" * 60)
    print("""
We just did ONE learning step on ONE fake batch. In Step 09,
we'll run a full training EPOCH — looping through ALL 7 batches
from the DataLoader and performing this 5-line cycle on each one.

After one epoch, the model will have updated its weights 7 times.
After 50 epochs: 350 updates. That's when things get interesting.
""")


# ======================================================================
# ENTRY POINT
# ======================================================================
if __name__ == "__main__":
    main()
