"""
generate.py — Generate new text using the trained model.

PURPOSE:
    This is the PAYOFF — where the trained model finally produces text.
    We load the saved model weights and vocabulary from disk, then
    generate new text one character at a time.

    The process is called INFERENCE — using a trained model to produce
    output. No learning happens here; we're just reading the model's
    predictions.

HOW TEXT GENERATION WORKS:
    The model was trained to predict "given these characters, what comes
    next?" During generation, we exploit this ability:

    1. Start with a seed (e.g., "The")
    2. Feed it to the model → get 48 scores (one per character)
    3. Pick the highest-scoring character (e.g., " ")
    4. Append it to our text: "The "
    5. Feed "The " to the model → get 48 new scores
    6. Pick the highest → "o"
    7. Now we have "The o"
    8. Repeat for as many characters as we want

    This is called AUTOREGRESSIVE generation — each new character
    becomes input for predicting the next one.

    ┌──────────────────────────────────────────────────────────┐
    │  "The" → model → scores → pick 'e' → "The "            │
    │  "The " → model → scores → pick 'o' → "The o"          │
    │  "The o" → model → scores → pick 'n' → "The on"        │
    │  "The on" → model → scores → pick 'l' → "The onl"      │
    │  "The onl" → model → scores → pick 'y' → "The only"    │
    │  ...and so on for hundreds of characters                 │
    └──────────────────────────────────────────────────────────┘

GREEDY vs SAMPLING:
    There are two ways to pick the next character from the scores:

    GREEDY (argmax):
      Always pick the highest-scoring character.
      → Deterministic: same seed always produces same output.
      → Can be repetitive: gets stuck in loops.

    SAMPLING (probability-based):
      Convert scores to probabilities, then randomly sample.
      → More creative and varied output.
      → Can produce surprising or nonsensical text.
      → Controlled by "temperature" (Step 13).

    This step uses GREEDY. Step 13 adds temperature-controlled sampling.

WHAT THIS FILE PROVIDES:
    1. load_model()     — load trained weights from disk
    2. load_vocabulary() — load vocabulary mappings from disk
    3. generate_text()  — generate text character by character
    4. main()           — demo: load model + generate several examples

INPUT:  outputs/model.pth, outputs/vocab.pth
OUTPUT: Printed generated text

Usage:
    PYTHONPATH=src python src/generate.py
"""

import torch
import torch.nn.functional as F

from model import TinyLanguageModel


def load_model(model_path="outputs/model.pth", vocab_size=48):
    """
    Load a trained model from disk.

    This reverses the save_model() process from Step 10:
    1. Create a fresh model (random weights)
    2. Load the trained weights from the .pth file
    3. Overwrite the random weights with the trained ones

    WHY TWO STEPS (create then load)?
        torch.save() only saved the WEIGHTS (state_dict), not the
        model architecture (the code). So we need to:
        - Create the model structure first (TinyLanguageModel)
        - Then fill in the learned weights (load_state_dict)

        This is like building an empty house (architecture) and then
        moving in the furniture (weights). The house plan is in model.py;
        the furniture is in model.pth.

    WHAT IS model.eval()?
        Sets the model to EVALUATION mode (opposite of model.train()).
        In eval mode:
        - Dropout layers are disabled (all neurons active)
        - BatchNorm uses stored statistics instead of batch stats
        Our simple model doesn't use these, but it's important practice.
        ALWAYS call model.eval() before inference/generation.

    WHAT IS weights_only=True?
        A security parameter for torch.load(). The .pth file format uses
        Python's pickle, which can execute arbitrary code when loaded.
        weights_only=True restricts loading to just tensor data, blocking
        potential code execution from malicious files.
        Since state_dict contains only tensors, this is safe and recommended.

    PARAMETERS:
        model_path (str):  Path to the saved model file
        vocab_size (int):  Number of characters (must match training)

    RETURNS:
        TinyLanguageModel: The model with trained weights, in eval mode
    """

    # ---- Create a fresh model with random weights ----
    # This builds the same architecture as during training:
    # Embedding(48, 128) → RNN(128, 256, 2) → Linear(256, 48)
    model = TinyLanguageModel(vocab_size=vocab_size)

    # ---- Load the trained weights ----
    # torch.load() reads the .pth file and returns a state_dict.
    # model.load_state_dict() overwrites the random weights with
    # the trained weights from the file.
    #
    # After this line, the model contains the same 248,880 numbers
    # that were saved after 100 epochs of training.
    model.load_state_dict(torch.load(model_path, weights_only=True))

    # ---- Set to evaluation mode ----
    # No training will happen — we're only generating text.
    model.eval()

    return model


def load_vocabulary(vocab_path="outputs/vocab.pth"):
    """
    Load the vocabulary mappings from disk.

    The vocabulary was saved as a dictionary with three keys:
    - 'chars': list of all characters
    - 'char_to_idx': character → number mapping
    - 'idx_to_char': number → character mapping

    We need these to:
    1. ENCODE the seed text → numbers (so the model can read it)
    2. DECODE the model's number outputs → text (so we can read it)

    Without the SAME vocabulary used during training, the model's
    outputs would map to wrong characters and produce gibberish.

    PARAMETERS:
        vocab_path (str): Path to the saved vocabulary file

    RETURNS:
        dict: Dictionary with 'chars', 'char_to_idx', 'idx_to_char'
    """

    # ---- Load the vocabulary dictionary ----
    # weights_only=False because this is a plain Python dict (not tensors).
    # We saved this file ourselves in train.py, so it's safe.
    vocab_data = torch.load(vocab_path, weights_only=False)

    return vocab_data


def generate_text(model, vocab_data, seed_text="The", length=200):
    """
    Generate text character by character using the trained model.

    This is AUTOREGRESSIVE generation:
    - Start with a seed text
    - Predict the next character
    - Append it to the text
    - Use the extended text to predict the next character
    - Repeat

    THE GENERATION LOOP IN DETAIL:

    Step 1: Encode the seed
      "The" → [19, 29, 26]

    Step 2: Feed to model
      model([19, 29, 26]) → logits shape (1, 3, 48)
      We only care about the LAST position's predictions:
      logits[0, -1, :] → 48 scores

    Step 3: Pick the next character (GREEDY)
      torch.argmax(scores) → index of highest score
      e.g., index 1 → ' ' (space)

    Step 4: Append and repeat
      text becomes "The " → feed again → predict next → "The o" → ...

    WHY DO WE ONLY LOOK AT THE LAST POSITION?
        The model outputs predictions for EVERY position in the input.
        But we only need the prediction after the LAST character —
        that's the "what comes next?" prediction we want.

        logits shape: (1, seq_len, 48)
                            ↑
                       logits[0, -1, :] ← last position's 48 scores

    WHAT IS torch.no_grad()?
        Tells PyTorch NOT to track operations for gradient computation.
        During training, PyTorch builds a computation graph to compute
        gradients. During generation, we don't need gradients (no
        learning happening), so disabling them:
        - Uses less memory (no graph stored)
        - Runs faster (no gradient bookkeeping)

    WHAT IS unsqueeze(0)?
        Adds a batch dimension. The model expects input shape
        (batch_size, seq_len), but we have just (seq_len,).
        unsqueeze(0) adds a dimension at position 0:
        tensor([19, 29, 26]) shape (3,)
        → tensor([[19, 29, 26]]) shape (1, 3)

    PARAMETERS:
        model (TinyLanguageModel): Trained model in eval mode
        vocab_data (dict):         Vocabulary with char_to_idx, idx_to_char
        seed_text (str):           Starting text to continue from
        length (int):              Number of characters to generate

    RETURNS:
        str: The seed text + generated characters

    EXAMPLE:
        >>> text = generate_text(model, vocab_data, seed_text="The", length=100)
        >>> print(text)
        The only way to do great work is to love what you do...
    """

    # ---- Extract vocabulary mappings ----
    char_to_idx = vocab_data['char_to_idx']
    idx_to_char = vocab_data['idx_to_char']

    # ---- Encode the seed text ----
    # Convert each character to its number using the vocabulary.
    # "The" → [19, 29, 26]
    #
    # We use a list comprehension: for each character in the seed,
    # look up its index in char_to_idx.
    input_indices = [char_to_idx[ch] for ch in seed_text]

    # ---- Start building the generated text ----
    # We'll append each new character to this list and join at the end.
    generated_chars = list(seed_text)

    # ---- Initialize the hidden state ----
    # The RNN's hidden state carries context from previous characters.
    # We start with None (the model will initialize it to zeros).
    # As we generate, we pass the hidden state forward so each new
    # character has context from ALL previous characters.
    hidden = None

    # ---- Disable gradient tracking ----
    # torch.no_grad() is a context manager (used with 'with').
    # Everything inside this block runs without building the
    # computation graph, saving memory and time.
    with torch.no_grad():

        # ---- Generate one character at a time ----
        for i in range(length):

            # ---- Prepare the input tensor ----
            # Convert our list of indices to a PyTorch tensor.
            # torch.tensor([19, 29, 26]) → shape (3,)
            #
            # .unsqueeze(0) adds batch dimension:
            # shape (3,) → shape (1, 3) — a "batch" of 1 sequence
            #
            # The model expects (batch_size, seq_len).
            input_tensor = torch.tensor(input_indices).unsqueeze(0)

            # ---- Forward pass ----
            # Run the input through the model to get predictions.
            # logits shape: (1, seq_len, 48) — 48 scores per position
            # hidden: the RNN's updated hidden state
            #
            # We pass the hidden state from the previous iteration
            # so the RNN has context from all previous characters.
            logits, hidden = model(input_tensor, hidden)

            # ---- Get the last position's predictions ----
            # We only care about the prediction AFTER the last character.
            # logits[0, -1, :] → shape (48,) — 48 scores
            #
            # logits[0]     → remove batch dimension
            # logits[0, -1] → last position (the "next character" prediction)
            next_logits = logits[0, -1, :]

            # ---- GREEDY selection: pick the highest score ----
            # torch.argmax() returns the INDEX of the maximum value.
            # This index IS the predicted character's number.
            #
            # Example: if scores are [..., 8.5, 2.1, ...] and position
            # 26 has the highest score (8.5), argmax returns 26.
            # idx_to_char[26] = 'e', so the model predicts 'e'.
            #
            # .item() converts the tensor to a plain Python int.
            next_idx = torch.argmax(next_logits).item()

            # ---- Convert number back to character ----
            # Use idx_to_char to decode: 26 → 'e'
            next_char = idx_to_char[next_idx]

            # ---- Append to generated text ----
            generated_chars.append(next_char)

            # ---- Prepare input for next iteration ----
            # For the next prediction, we only need the LAST character
            # as input (since the hidden state already carries context
            # from all previous characters).
            #
            # This is more efficient than feeding the entire text each time.
            # The hidden state is the RNN's "memory" — it already knows
            # about "The only way to..." from previous iterations.
            input_indices = [next_idx]

    # ---- Join all characters into a single string ----
    # generated_chars is a list like ['T', 'h', 'e', ' ', 'o', 'n', ...]
    # ''.join() concatenates them: "The on..."
    return ''.join(generated_chars)


def main():
    """
    Main function — load trained model and generate text examples.

    FLOW:
        1. Load model from outputs/model.pth
        2. Load vocabulary from outputs/vocab.pth
        3. Generate text with several different seeds
        4. Show the results
    """

    import os

    # ==================================================================
    # SETUP
    # ==================================================================
    print("=" * 60)
    print("STEP 12: TEXT GENERATION")
    print("=" * 60)
    print()

    # ---- Check that required files exist ----
    model_path = "outputs/model.pth"
    vocab_path = "outputs/vocab.pth"

    if not os.path.exists(model_path):
        print(f"  ERROR: {model_path} not found!")
        print(f"  Run train.py first (Step 10) to train and save the model.")
        return

    if not os.path.exists(vocab_path):
        print(f"  ERROR: {vocab_path} not found!")
        print(f"  Run train.py first (Step 10) to save the vocabulary.")
        return

    # ---- Load the trained model ----
    print("Loading trained model...")
    vocab_data = load_vocabulary(vocab_path)
    vocab_size = len(vocab_data['chars'])
    model = load_model(model_path, vocab_size=vocab_size)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {total_params:,} parameters")
    print(f"  Vocabulary: {vocab_size} characters")
    print()

    # ==================================================================
    # GENERATE TEXT: Multiple examples with different seeds
    # ==================================================================
    # We try different seed texts to see how the model continues them.
    # The seed gives the model a starting context — different seeds
    # lead to different generated text.

    seeds = [
        "The",
        "Life",
        "In the",
        "Success",
        "Be",
    ]

    print("=" * 60)
    print("GENERATED TEXT (Greedy — always picks highest probability)")
    print("=" * 60)

    for seed in seeds:
        print(f'\n--- Seed: "{seed}" ---')

        # Generate 200 characters starting from this seed
        generated = generate_text(model, vocab_data,
                                   seed_text=seed, length=200)

        print(generated)
        print()

    # ==================================================================
    # EXPLAIN THE OUTPUT
    # ==================================================================
    print("=" * 60)
    print("UNDERSTANDING THE OUTPUT")
    print("=" * 60)
    print("""
What you're seeing:
  - The model generates text character by character
  - It learned patterns from 99 inspirational quotes
  - Greedy decoding always picks the most likely next character
  - This can lead to repetitive output (same phrases over and over)

Why it might look repetitive:
  - Greedy decoding always makes the same choice in the same context
  - The model memorized common phrases from the training data
  - With only 6,201 characters of training data, variety is limited

In Step 13, we'll add TEMPERATURE to control randomness:
  - Low temperature (0.5):  safe, predictable, closer to training data
  - Medium temperature (1.0): balanced creativity
  - High temperature (1.5):  wild, creative, sometimes nonsensical
""")


# ======================================================================
# ENTRY POINT
# ======================================================================
if __name__ == "__main__":
    main()
