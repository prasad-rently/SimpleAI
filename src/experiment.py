"""
experiment.py — Run experiments to see how settings affect AI output.

PURPOSE:
    This is the "science" step — where we change ONE variable at a time
    and observe how it affects the model's behavior. This is how real
    AI researchers work: change a setting, retrain, compare results.

    We run three experiments:
    1. Data size:  What happens with less training data?
    2. Model size: What happens with fewer/more neurons?
    3. Epochs:     What happens with less/more training?

    Each experiment trains a fresh model and generates sample text,
    so you can see the differences side by side.

WHY EXPERIMENT?
    The settings we used (embed_size=128, hidden_size=256, 100 epochs)
    were chosen as good defaults. But HOW do you know they're good?
    By trying alternatives and comparing.

    This is the core of machine learning engineering:
    - Form a hypothesis ("more epochs = better text")
    - Run the experiment
    - Compare results
    - Draw conclusions

WHAT THIS FILE PROVIDES:
    1. run_experiment()           — train + generate with given config
    2. experiment_data_size()     — compare 25%, 50%, 100% of data
    3. experiment_model_size()    — compare small, medium, large models
    4. experiment_num_epochs()    — compare 10, 50, 100 epochs
    5. main()                     — run all experiments

INPUT:  data/input.txt
OUTPUT: Printed comparison tables with loss and generated text

Usage:
    PYTHONPATH=src python src/experiment.py
"""

import torch
import torch.nn as nn
import math

from vocabulary import Vocabulary
from dataset import TextDataset, create_dataloader
from model import TinyLanguageModel
from generate import generate_text_with_temperature


def run_experiment(text, label, num_epochs=50, embed_size=128,
                   hidden_size=256, num_layers=2, seed="The",
                   temperature=0.8, gen_length=150):
    """
    Train a model with the given config and generate sample text.

    This is a self-contained experiment: it builds everything from
    scratch (vocab, dataset, model, optimizer), trains for the specified
    number of epochs, and generates text.

    PARAMETERS:
        text (str):        Training text to learn from
        label (str):       Name for this experiment (for display)
        num_epochs (int):  How many passes through the data
        embed_size (int):  Embedding dimension (characters → vectors)
        hidden_size (int): RNN hidden state size (model's "memory")
        num_layers (int):  Number of stacked RNN layers
        seed (str):        Seed text for generation
        temperature (float): Generation temperature
        gen_length (int):  Characters to generate

    RETURNS:
        dict: Results with keys: label, final_loss, generated_text, params
    """

    # ---- Build pipeline ----
    vocab = Vocabulary(text)
    dataset = TextDataset(text, vocab, seq_length=50)
    dataloader = create_dataloader(dataset, batch_size=16, shuffle=True)

    # ---- Create model with specified size ----
    model = TinyLanguageModel(
        vocab_size=vocab.vocab_size,
        embed_size=embed_size,
        hidden_size=hidden_size,
        num_layers=num_layers
    )
    total_params = sum(p.numel() for p in model.parameters())

    # ---- Train ----
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    model.train()
    final_loss = 0.0

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0

        for inputs, targets in dataloader:
            logits, _ = model(inputs)
            loss = loss_fn(logits.view(-1, vocab.vocab_size),
                          targets.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            num_batches += 1

        final_loss = total_loss / num_batches

    # ---- Generate text ----
    model.eval()
    vocab_data = {
        'chars': vocab.chars,
        'char_to_idx': vocab.char_to_idx,
        'idx_to_char': vocab.idx_to_char,
    }

    generated = generate_text_with_temperature(
        model, vocab_data,
        seed_text=seed, length=gen_length, temperature=temperature
    )

    return {
        'label': label,
        'final_loss': final_loss,
        'generated_text': generated,
        'params': total_params,
        'num_epochs': num_epochs,
    }


def experiment_data_size(full_text):
    """
    Experiment 1: How does the amount of training data affect quality?

    We train on 25%, 50%, and 100% of the data and compare.

    HYPOTHESIS:
        More data = lower loss = better text.
        But even 25% might produce recognizable patterns since
        our data has many repeated structures (quotes).

    WHY THIS MATTERS:
        In real AI, data is often the bottleneck. GPT-4 was trained
        on trillions of words. Our model has only 6,201 characters.
        This experiment shows why data quantity matters.
    """

    print("=" * 60)
    print("EXPERIMENT 1: DATA SIZE")
    print("=" * 60)
    print("How does the amount of training data affect output quality?")
    print()

    # ---- Define data splits ----
    # We take the first N% of the text for each experiment.
    splits = [
        (0.25, "25% of data"),
        (0.50, "50% of data"),
        (1.00, "100% of data"),
    ]

    results = []

    for fraction, label in splits:
        # Slice the text to the desired fraction
        end_idx = int(len(full_text) * fraction)
        text_slice = full_text[:end_idx]

        print(f"  Training on {label} ({len(text_slice)} chars)...")
        result = run_experiment(text_slice, label, num_epochs=50)
        results.append(result)
        print(f"    Final loss: {result['final_loss']:.4f}")

    # ---- Print comparison ----
    print()
    print("  Results comparison:")
    print("  " + "-" * 56)
    print(f"  {'Config':<16} {'Chars':>6} {'Loss':>8} {'Params':>10}")
    print("  " + "-" * 56)

    for i, (fraction, label) in enumerate(splits):
        r = results[i]
        chars = int(len(full_text) * fraction)
        print(f"  {r['label']:<16} {chars:>6} {r['final_loss']:>8.4f} "
              f"{r['params']:>10,}")

    print()
    for r in results:
        print(f"  --- {r['label']} ---")
        print(f"  {r['generated_text']}")
        print()

    return results


def experiment_model_size(text):
    """
    Experiment 2: How does model size affect quality?

    We compare three model sizes:
    - Tiny:   embed=32,  hidden=64,   1 layer  (~5K params)
    - Medium: embed=128, hidden=256,  2 layers (~249K params, our default)
    - Large:  embed=256, hidden=512,  2 layers (~1.3M params)

    HYPOTHESIS:
        Larger models can learn more complex patterns but may also
        memorize (overfit) more easily on our small dataset.
        The tiny model might struggle with even basic patterns.

    WHY THIS MATTERS:
        Model size is one of the biggest decisions in AI engineering.
        GPT-4 has ~1.7 trillion parameters. Our "large" has 1.3 million.
        This experiment shows the tradeoff: bigger = more capable,
        but also slower and more expensive.
    """

    print("=" * 60)
    print("EXPERIMENT 2: MODEL SIZE")
    print("=" * 60)
    print("How does the number of parameters affect output quality?")
    print()

    # ---- Define model sizes ----
    configs = [
        {"label": "Tiny (5K)",    "embed_size": 32,  "hidden_size": 64,
         "num_layers": 1},
        {"label": "Medium (249K)", "embed_size": 128, "hidden_size": 256,
         "num_layers": 2},
        {"label": "Large (1.3M)",  "embed_size": 256, "hidden_size": 512,
         "num_layers": 2},
    ]

    results = []

    for config in configs:
        print(f"  Training {config['label']}...")
        result = run_experiment(
            text, config['label'], num_epochs=50,
            embed_size=config['embed_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers']
        )
        results.append(result)
        print(f"    {result['params']:,} parameters, "
              f"final loss: {result['final_loss']:.4f}")

    # ---- Print comparison ----
    print()
    print("  Results comparison:")
    print("  " + "-" * 56)
    print(f"  {'Config':<16} {'Params':>10} {'Loss':>8}")
    print("  " + "-" * 56)

    for r in results:
        print(f"  {r['label']:<16} {r['params']:>10,} "
              f"{r['final_loss']:>8.4f}")

    print()
    for r in results:
        print(f"  --- {r['label']} ---")
        print(f"  {r['generated_text']}")
        print()

    return results


def experiment_num_epochs(text):
    """
    Experiment 3: How does training duration affect quality?

    We compare 10, 50, and 100 epochs of training.

    HYPOTHESIS:
        More epochs = lower loss = better text, up to a point.
        After that, the model starts memorizing instead of learning
        (overfitting). With our tiny dataset, overfitting happens fast.

    WHY THIS MATTERS:
        Training time is a major cost in AI. GPT-4 training cost
        ~$100 million. Knowing WHEN to stop training saves money
        and can produce better models (less overfitting).
    """

    print("=" * 60)
    print("EXPERIMENT 3: NUMBER OF EPOCHS")
    print("=" * 60)
    print("How does training duration affect output quality?")
    print()

    # ---- Define epoch counts ----
    epoch_counts = [10, 50, 100]

    results = []

    for epochs in epoch_counts:
        label = f"{epochs} epochs"
        print(f"  Training for {label}...")
        result = run_experiment(text, label, num_epochs=epochs)
        results.append(result)
        print(f"    Final loss: {result['final_loss']:.4f}")

    # ---- Print comparison ----
    print()
    print("  Results comparison:")
    print("  " + "-" * 56)
    print(f"  {'Config':<16} {'Epochs':>8} {'Loss':>8} {'Updates':>10}")
    print("  " + "-" * 56)

    for r in results:
        updates = r['num_epochs'] * 7
        print(f"  {r['label']:<16} {r['num_epochs']:>8} "
              f"{r['final_loss']:>8.4f} {updates:>10}")

    print()
    for r in results:
        print(f"  --- {r['label']} ---")
        print(f"  {r['generated_text']}")
        print()

    return results


def print_conclusions():
    """Print key takeaways from the experiments."""

    print("=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
  1. DATA SIZE MATTERS MOST
     More data = better patterns = better text.
     In real AI, companies spend millions collecting data.
     Our 6,201 chars is tiny — real LLMs use trillions of characters.

  2. MODEL SIZE HAS DIMINISHING RETURNS
     Bigger isn't always better, especially with little data.
     A large model on small data memorizes instead of generalizing.
     The model should be sized appropriately for the data.

  3. MORE EPOCHS HELP, THEN PLATEAU
     Early epochs give the biggest improvement.
     After convergence, more epochs give minimal gains and
     risk overfitting (memorizing the training data).

  4. THE REAL-WORLD PARALLELS
     These same tradeoffs apply to ChatGPT, Claude, etc:
     - GPT-4: ~1.7T parameters, trillions of training tokens
     - Claude: similar scale, different architecture
     - Our model: 249K parameters, 6K characters
     The PRINCIPLES are identical — only the scale differs.

  ┌─────────────────────────────────────────────────────────┐
  │  SimpleAI is complete!                                   │
  │                                                          │
  │  You built a working AI model from scratch:              │
  │    - Data pipeline (text → numbers → batches)           │
  │    - Neural network (embedding → RNN → output)          │
  │    - Training loop (forward → loss → backward → update) │
  │    - Text generation (autoregressive with temperature)  │
  │    - Interactive interface (chat-like prompt)            │
  │    - Scientific experiments (varying hyperparameters)    │
  │                                                          │
  │  These are the same building blocks used in every        │
  │  large language model in the world.                      │
  └─────────────────────────────────────────────────────────┘
""")


def main():
    """
    Main function — run all three experiments.

    FLOW:
        1. Load the full training text
        2. Run Experiment 1: Data size comparison
        3. Run Experiment 2: Model size comparison
        4. Run Experiment 3: Epoch count comparison
        5. Print conclusions
    """

    print("=" * 60)
    print("STEP 15: EXPERIMENTS — HOW SETTINGS AFFECT AI OUTPUT")
    print("=" * 60)
    print()
    print("We'll train multiple models with different settings and")
    print("compare the results. This takes a minute or two...")
    print()

    # ---- Load the full training text ----
    filepath = "data/input.txt"
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Data loaded: {len(text)} characters from {filepath}")
    print()

    # ---- Run experiments ----
    experiment_data_size(text)
    experiment_model_size(text)
    experiment_num_epochs(text)

    # ---- Conclusions ----
    print_conclusions()


# ======================================================================
# ENTRY POINT
# ======================================================================
if __name__ == "__main__":
    main()
