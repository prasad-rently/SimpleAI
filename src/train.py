"""
train.py — Train the model for multiple epochs and save the result.

PURPOSE:
    This file handles the COMPLETE training process:
    - Step 09: train_one_epoch() — runs one pass through all batches
    - Step 10: train() — runs MANY epochs, tracks loss history, saves model

    After training, we save two things:
    1. The trained model weights → outputs/model.pth
    2. The vocabulary            → outputs/vocab.pth
    Both are needed later for text generation (Step 12).

WHAT HAPPENS OVER MANY EPOCHS:
    Epoch  0: loss ≈ 3.90 (random — model knows nothing)
    Epoch 10: loss ≈ 2.50 (learning common characters like 'e', ' ')
    Epoch 30: loss ≈ 1.80 (learning common words like "the", "is")
    Epoch 50: loss ≈ 1.50 (learning phrase patterns)
    Epoch 80: loss ≈ 1.20 (generating recognizable text)

    The loss CURVE shows this improvement visually (Step 11).

WHY SAVE THE MODEL?
    Training takes time (minutes for us, days for GPT-4).
    Once trained, we save the weights to disk so we can:
    - Generate text without retraining (Step 12)
    - Share the model with others
    - Resume training later if needed

    We save with torch.save(), which serializes (converts to bytes)
    the model's state_dict — a dictionary of all 248,880 learned weights.

WHAT THIS FILE PROVIDES:
    1. train_one_epoch()  — train for one epoch (from Step 09)
    2. train()            — train for many epochs, return loss history
    3. save_model()       — save trained model to disk
    4. save_vocabulary()  — save vocabulary for generation
    5. main()             — full pipeline: setup → train → save

INPUT:  data/input.txt
OUTPUT: outputs/model.pth, outputs/vocab.pth, printed loss per epoch

Usage:
    PYTHONPATH=src python src/train.py
"""

import os
import math
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

    # ---- Compute average loss for the epoch ----
    # Average = total / count. This smooths out batch-to-batch noise
    # and gives a single number representing the epoch's performance.
    avg_loss = total_loss / num_batches

    return avg_loss


def train(model, dataloader, loss_fn, optimizer, vocab_size, num_epochs=50,
          print_every=5):
    """
    Train the model for multiple epochs, tracking loss over time.

    This is the OUTER training loop. It calls train_one_epoch() repeatedly,
    building up a list of loss values — the "loss history". This history
    is what gets plotted as the training curve (Step 11).

    HOW MULTI-EPOCH TRAINING WORKS:
        Each epoch is a FULL pass through the data (7 batches).
        With 50 epochs, the model sees every example 50 times.
        That's 50 × 7 = 350 total weight updates.

        Why see the same data multiple times?
        - First pass:  model learns the most obvious patterns
        - Later passes: model refines and picks up subtler patterns
        - Like re-reading a textbook — each pass reveals more detail

        The DataLoader SHUFFLES examples each epoch, so the model
        sees them in a different order every time. This prevents it
        from memorizing the order of examples.

    WHAT IS "LOSS HISTORY"?
        A list of average loss values, one per epoch:
          [3.33, 2.91, 2.65, 2.48, ...]

        This list has two important uses:
        1. We can SEE if the model is improving (loss going down)
        2. We can PLOT it as a chart (Step 11) to visualize training

        If loss stops decreasing, the model has learned what it can
        (or we need to change hyperparameters).

    WHAT IS print_every?
        Controls how often we print progress. With 50 epochs:
        - print_every=1:  50 lines of output (noisy)
        - print_every=5:  10 lines of output (cleaner)
        - print_every=10: 5 lines of output (compact)

        We always print the first epoch (0) and the last epoch
        so you can see the start and end.

    PARAMETERS:
        model (TinyLanguageModel):   The model to train
        dataloader (DataLoader):     Batched training data
        loss_fn (CrossEntropyLoss):  Measures prediction errors
        optimizer (Adam):            Updates weights using gradients
        vocab_size (int):            Number of characters (for reshaping)
        num_epochs (int):            How many times to pass through all data
        print_every (int):           Print progress every N epochs

    RETURNS:
        list[float]: Loss history — one average loss value per epoch.
                     Used for plotting the training curve (Step 11).

    EXAMPLE:
        >>> loss_history = train(model, dataloader, loss_fn, optimizer, 48,
        ...                      num_epochs=50, print_every=10)
        Epoch   0/50 | Loss: 3.3251
        Epoch  10/50 | Loss: 2.1534
        Epoch  20/50 | Loss: 1.7821
        ...
        >>> len(loss_history)
        50
    """

    # ---- Store loss for every epoch ----
    # This list will have num_epochs entries when training finishes.
    # Each entry is the average loss for that epoch.
    # Example after 50 epochs: [3.33, 2.91, 2.65, 2.48, 2.35, ...]
    loss_history = []

    # ---- Train for each epoch ----
    # range(num_epochs) gives [0, 1, 2, ..., num_epochs-1]
    # Each iteration = one complete pass through the training data.
    for epoch in range(num_epochs):

        # Run one epoch and get its average loss
        # This calls train_one_epoch() which loops through all 7 batches
        avg_loss = train_one_epoch(
            model, dataloader, loss_fn, optimizer,
            vocab_size, epoch_num=epoch
        )

        # ---- Record the loss ----
        # Append to our history list so we can plot it later.
        # loss_history[0] = epoch 0's average loss
        # loss_history[1] = epoch 1's average loss
        # etc.
        loss_history.append(avg_loss)

        # ---- Print progress at intervals ----
        # We print on:
        #   - The first epoch (epoch == 0) → see starting loss
        #   - Every print_every epochs     → see progress
        #   - The last epoch               → see final loss
        #
        # The % operator (modulo) gives the remainder after division.
        # epoch % 5 == 0 is True for epochs 0, 5, 10, 15, 20, ...
        is_first = (epoch == 0)
        is_interval = (epoch % print_every == 0)
        is_last = (epoch == num_epochs - 1)

        if is_first or is_interval or is_last:
            print(f"  Epoch {epoch:3d}/{num_epochs} | Avg Loss: {avg_loss:.4f}")

    return loss_history


def save_model(model, filepath="outputs/model.pth"):
    """
    Save the trained model's weights to a file on disk.

    WHY SAVE THE MODEL?
        Training takes time. Once the model has learned useful patterns,
        we save those patterns (the weights) to a file. Later, we can
        load the weights into a fresh model and skip all the training.

        This is how AI models are distributed:
        - OpenAI trains GPT for months → saves weights
        - You download the weights → load into the same architecture
        - Now you have a trained model without training it yourself

    WHAT IS state_dict()?
        A Python dictionary containing ALL learnable parameters:
        {
            'embedding.weight':    tensor of shape (48, 128),
            'rnn.weight_ih_l0':    tensor of shape (256, 128),
            'rnn.weight_hh_l0':    tensor of shape (256, 256),
            'rnn.bias_ih_l0':      tensor of shape (256,),
            'rnn.bias_hh_l0':      tensor of shape (256,),
            'rnn.weight_ih_l1':    tensor of shape (256, 256),
            'rnn.weight_hh_l1':    tensor of shape (256, 256),
            'rnn.bias_ih_l1':      tensor of shape (256,),
            'rnn.bias_hh_l1':      tensor of shape (256,),
            'output_layer.weight': tensor of shape (48, 256),
            'output_layer.bias':   tensor of shape (48,),
        }

        That's 248,880 numbers total — the model's entire "knowledge"
        compressed into a single file (usually ~1-2 MB).

    WHAT IS torch.save()?
        Python's pickle + some PyTorch optimizations.
        Converts the dictionary of tensors into bytes and writes to disk.
        The .pth extension is the convention for PyTorch model files
        (short for "PyTorch").

    PARAMETERS:
        model (TinyLanguageModel): The trained model to save
        filepath (str):            Where to save (default: outputs/model.pth)

    RETURNS:
        None (writes file to disk)
    """

    # ---- Ensure the output directory exists ----
    # os.path.dirname("outputs/model.pth") → "outputs"
    # os.makedirs creates it if it doesn't exist.
    # exist_ok=True prevents errors if the folder already exists.
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # ---- Save the model's learned weights ----
    # model.state_dict() returns the dictionary of all parameters.
    # torch.save() serializes it (converts to bytes) and writes to disk.
    #
    # We save ONLY the state_dict, not the entire model object.
    # This is the recommended approach because:
    # 1. It's smaller (just the numbers, not the code)
    # 2. It's more portable (works even if you change the code slightly)
    # 3. It's the PyTorch community convention
    torch.save(model.state_dict(), filepath)

    # ---- Report file size ----
    # os.path.getsize() returns file size in bytes.
    # Divide by 1024 to get kilobytes (KB).
    file_size_kb = os.path.getsize(filepath) / 1024
    print(f"  Model saved to {filepath} ({file_size_kb:.1f} KB)")


def save_vocabulary(vocab, filepath="outputs/vocab.pth"):
    """
    Save the vocabulary mappings to disk.

    WHY SAVE THE VOCABULARY?
        The model outputs NUMBERS (indices 0-47). To convert those
        numbers back to text, we need the SAME idx_to_char mapping
        that was used during training.

        If we rebuilt the vocabulary from different text, the mappings
        could be different (e.g., 'a' might map to 23 instead of 22),
        and the generated text would be gibberish.

        So we save the vocabulary alongside the model to ensure
        encode/decode work correctly during generation (Step 12).

    WHAT WE SAVE:
        A dictionary with three essential pieces:
        {
            'chars':       ['\n', ' ', ',', '.', ...],  # the 48 characters
            'char_to_idx': {'\n': 0, ' ': 1, ...},     # encoding map
            'idx_to_char': {0: '\n', 1: ' ', ...},     # decoding map
        }

    PARAMETERS:
        vocab (Vocabulary): The vocabulary to save
        filepath (str):     Where to save (default: outputs/vocab.pth)

    RETURNS:
        None (writes file to disk)
    """

    # ---- Ensure the output directory exists ----
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # ---- Save vocabulary as a dictionary ----
    # We extract the three key attributes from the Vocabulary object
    # and save them as a plain dictionary. This makes it easy to
    # reconstruct the vocabulary later without needing the original text.
    vocab_data = {
        'chars': vocab.chars,
        'char_to_idx': vocab.char_to_idx,
        'idx_to_char': vocab.idx_to_char,
    }

    torch.save(vocab_data, filepath)

    file_size_kb = os.path.getsize(filepath) / 1024
    print(f"  Vocabulary saved to {filepath} ({file_size_kb:.1f} KB)")


def main():
    """
    Main function — full training pipeline with multi-epoch training.

    FLOW:
        1. Load data/input.txt
        2. Build Vocabulary (Step 04)
        3. Create TextDataset (Step 05)
        4. Create DataLoader (Step 06)
        5. Create TinyLanguageModel (Step 07)
        6. Create loss function + optimizer (Step 08)
        7. Train for MANY epochs (Step 10) ← NEW
        8. Save trained model + vocabulary to disk ← NEW
        9. Report results

    TRAINING CONFIGURATION:
        - num_epochs = 100  → the model sees all data 100 times
        - batch_size = 16   → 7 batches per epoch
        - learning_rate = 0.003 → slightly faster than default 0.001
        - Total weight updates: 100 × 7 = 700

        Why 100 epochs? For our small dataset (6201 chars), 100 epochs
        is enough to learn most patterns without severe overfitting.
        Larger datasets need fewer epochs; smaller ones need more.

        Why lr=0.003? Our model is small and our dataset is small.
        A slightly higher learning rate helps it converge faster.
        For larger models, lower learning rates are safer.
    """

    # ==================================================================
    # SETUP: Wire up the full pipeline
    # ==================================================================
    print("=" * 60)
    print("STEP 10: FULL TRAINING — MULTIPLE EPOCHS")
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
    # Using lr=0.003 instead of default 0.001 because our model and
    # dataset are small. A slightly higher learning rate helps the
    # model converge (reach low loss) faster.
    learning_rate = 0.003
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print(f"Loss: CrossEntropyLoss | Optimizer: Adam (lr={learning_rate})")
    print()

    # ==================================================================
    # TRAINING: Multiple epochs
    # ==================================================================
    # With 100 epochs and 7 batches per epoch, the model's weights
    # will be updated 700 times. Each update nudges all 248,880
    # parameters in the direction that reduces the loss.

    num_epochs = 100
    random_baseline = -math.log(1.0 / vocab.vocab_size)

    print(f"Random baseline loss: {random_baseline:.4f}")
    print(f"Training for {num_epochs} epochs ({num_epochs * len(dataloader)} "
          f"total weight updates)...")
    print()

    # ---- Run the training loop ----
    # train() calls train_one_epoch() for each epoch and collects
    # the average loss into a list: [loss_0, loss_1, ..., loss_99]
    loss_history = train(
        model, dataloader, loss_fn, optimizer,
        vocab.vocab_size, num_epochs=num_epochs, print_every=10
    )

    # ==================================================================
    # RESULTS: Summarize what the model learned
    # ==================================================================
    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    # ---- Report the improvement ----
    # loss_history[0] = first epoch's loss (near random)
    # loss_history[-1] = last epoch's loss (much lower)
    first_loss = loss_history[0]
    final_loss = loss_history[-1]
    best_loss = min(loss_history)
    best_epoch = loss_history.index(best_loss)

    print(f"""
  Training summary:
    Epochs trained  : {num_epochs}
    Weight updates  : {num_epochs * len(dataloader)}
    Random baseline : {random_baseline:.4f}
    First epoch loss: {first_loss:.4f}
    Final epoch loss: {final_loss:.4f}
    Best epoch loss : {best_loss:.4f} (epoch {best_epoch})
    Total improvement: {first_loss - final_loss:.4f} ({((first_loss - final_loss) / first_loss * 100):.1f}% reduction)
""")

    # ==================================================================
    # SAVE: Write model and vocabulary to disk
    # ==================================================================
    # After all that training, we save the results so we don't have
    # to retrain every time we want to generate text.

    print("Saving trained model and vocabulary...")
    save_model(model, filepath="outputs/model.pth")
    save_vocabulary(vocab, filepath="outputs/vocab.pth")
    print()

    # ---- Save loss history for plotting (Step 11) ----
    # We also save the loss history so the plotting step can
    # create a chart without retraining.
    torch.save(loss_history, "outputs/loss_history.pth")
    print(f"  Loss history saved to outputs/loss_history.pth ({len(loss_history)} epochs)")
    print()

    # ==================================================================
    # WHAT COMES NEXT
    # ==================================================================
    print("=" * 60)
    print("WHAT COMES NEXT (Step 11)")
    print("=" * 60)
    print("""
The model is trained and saved! Next steps:

  Step 11: Plot the loss curve (loss_history → chart)
           Visualize how loss decreased over 100 epochs.

  Step 12: Generate text using the saved model
           Load model.pth + vocab.pth → produce new text!

Files saved:
  outputs/model.pth        — trained model weights (248,880 parameters)
  outputs/vocab.pth        — vocabulary mappings (48 characters)
  outputs/loss_history.pth — loss values for plotting
""")


# ======================================================================
# ENTRY POINT
# ======================================================================
if __name__ == "__main__":
    main()
