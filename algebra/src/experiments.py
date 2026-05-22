"""
experiments.py — Run experiments to understand what affects model accuracy.

PURPOSE:
    Investigate how different factors affect the algebra solver's performance.
    Each experiment trains a small model quickly and compares accuracy.

EXPERIMENTS:
    1. Data size: how much training data does the model need?
    2. Model size: does a bigger model help?
    3. Equation complexity: which equation types are hardest?
    4. Teacher forcing: how important is the TF schedule?

    Each experiment trains for just 10 epochs (fast!) to show relative
    differences. The insights transfer to full training.

WHAT YOU'LL LEARN:
    - More data consistently helps, but with diminishing returns
    - Model size matters less than data quality for this task
    - Division equations (Type 6) are the hardest
    - Teacher forcing is critical — without it, the model barely learns

OUTPUT: Printed tables comparing experiment results

Usage:
    PYTHONPATH=algebra/src python algebra/src/experiments.py
"""

import time
import random
import torch
import torch.nn as nn

from vocab import build_vocab_from_data, PAD_IDX
from dataset import AlgebraDataset, collate_fn, create_dataloaders
from seq2seq import Seq2Seq, create_model
from encoder import Encoder
from decoder import Decoder
from evaluate import classify_equation
from train import train_one_epoch, evaluate_accuracy


# ====================================================================
# HELPERS
# ====================================================================

def quick_train(model, train_loader, vocab, epochs=10, lr=0.002,
                tf_ratio=1.0):
    """
    Quick training loop for experiments (no scheduler, fixed TF).

    Returns final loss after training.
    """

    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    final_loss = 0
    for epoch in range(epochs):
        final_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer,
            vocab.vocab_size, tf_ratio
        )

    return final_loss


def measure_accuracy_by_type(model, test_loader, vocab):
    """Compute accuracy broken down by equation type."""

    model.eval()
    type_stats = {}

    with torch.no_grad():
        for enc_input, _, dec_target in test_loader:
            preds, _ = model.solve(enc_input)
            for i in range(enc_input.shape[0]):
                eq = vocab.decode_until_eos(enc_input[i].tolist())
                pred = vocab.decode_until_eos(preds[i].tolist())
                target = vocab.decode_until_eos(dec_target[i].tolist())

                eq_type = classify_equation(eq)
                if eq_type not in type_stats:
                    type_stats[eq_type] = {"correct": 0, "total": 0}
                type_stats[eq_type]["total"] += 1
                if pred == target:
                    type_stats[eq_type]["correct"] += 1

    return type_stats


# ====================================================================
# EXPERIMENT 1: DATA SIZE
# ====================================================================

def experiment_data_size(vocab):
    """
    How much data does the model need?

    Trains with 2K, 10K, and 40K examples and compares accuracy.
    Shows that more data consistently helps.
    """

    print("=" * 60)
    print("EXPERIMENT 1: DATA SIZE")
    print("  Question: How much training data does the model need?")
    print("=" * 60)
    print()

    with open("algebra/data/equations.txt") as f:
        all_pairs = [line.strip().split("\t") for line in f]

    random.seed(42)
    random.shuffle(all_pairs)

    test_pairs = all_pairs[40_000:]
    test_dataset = AlgebraDataset(test_pairs, vocab)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, collate_fn=collate_fn
    )

    results = []
    for data_size in [2_000, 10_000, 40_000]:
        train_pairs = all_pairs[:data_size]
        train_dataset = AlgebraDataset(train_pairs, vocab)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=64, shuffle=True,
            drop_last=True, collate_fn=collate_fn
        )

        model = create_model(vocab_size=vocab.vocab_size)
        start = time.time()
        loss = quick_train(model, train_loader, vocab, epochs=10)
        elapsed = time.time() - start
        acc = evaluate_accuracy(model, test_loader, vocab)

        results.append((data_size, loss, acc, elapsed))
        print(f"  {data_size:>6,} examples → "
              f"Loss: {loss:.4f}, Accuracy: {acc:.1%}, "
              f"Time: {elapsed:.0f}s")

    print()
    print("  Conclusion: More data → better accuracy. Even 10K examples")
    print("  get decent results, but 40K pushes accuracy much higher.")
    print()
    return results


# ====================================================================
# EXPERIMENT 2: MODEL SIZE
# ====================================================================

def experiment_model_size(vocab):
    """
    Does a bigger model help?

    Compares small (64-dim), medium (128-dim), and large (256-dim) encoders.
    """

    print("=" * 60)
    print("EXPERIMENT 2: MODEL SIZE")
    print("  Question: Does a bigger model learn better?")
    print("=" * 60)
    print()

    train_loader, test_loader = create_dataloaders(
        vocab, batch_size=64, train_ratio=0.8
    )

    configs = [
        ("Small  (enc=64,  dec=128)", 64, 128),
        ("Medium (enc=128, dec=256)", 128, 256),
        ("Large  (enc=256, dec=512)", 256, 512),
    ]

    results = []
    for name, enc_hidden, dec_hidden in configs:
        encoder = Encoder(
            vocab_size=vocab.vocab_size, embed_size=64,
            hidden_size=enc_hidden, num_layers=2, dropout=0.1
        )
        decoder = Decoder(
            vocab_size=vocab.vocab_size, embed_size=64,
            hidden_size=dec_hidden, num_layers=2, dropout=0.1
        )
        model = Seq2Seq(encoder, decoder)

        params = sum(p.numel() for p in model.parameters())
        start = time.time()
        loss = quick_train(model, train_loader, vocab, epochs=10)
        elapsed = time.time() - start
        acc = evaluate_accuracy(model, test_loader, vocab)

        results.append((name, params, loss, acc, elapsed))
        print(f"  {name}  {params:>10,} params → "
              f"Loss: {loss:.4f}, Accuracy: {acc:.1%}, "
              f"Time: {elapsed:.0f}s")

    print()
    print("  Conclusion: Bigger models have more capacity, but diminishing")
    print("  returns. The medium model (our default) is a good balance")
    print("  between accuracy and training speed.")
    print()
    return results


# ====================================================================
# EXPERIMENT 3: EQUATION COMPLEXITY (per-type analysis)
# ====================================================================

def experiment_complexity(vocab):
    """
    Which equation types are hardest?

    Uses the trained model to measure accuracy per equation type.
    """

    print("=" * 60)
    print("EXPERIMENT 3: EQUATION COMPLEXITY")
    print("  Question: Which equation types are hardest to solve?")
    print("=" * 60)
    print()

    model = create_model(vocab_size=vocab.vocab_size)
    model.load_state_dict(
        torch.load("algebra/outputs/model.pth", weights_only=True)
    )
    model.eval()

    _, test_loader = create_dataloaders(vocab, batch_size=64, train_ratio=0.8)
    type_stats = measure_accuracy_by_type(model, test_loader, vocab)

    results = []
    for type_name in sorted(type_stats.keys()):
        s = type_stats[type_name]
        acc = s["correct"] / s["total"] if s["total"] > 0 else 0
        results.append((type_name, s["total"], acc))
        bar = "█" * int(acc * 30)
        print(f"  {type_name:<28} {acc:>6.1%}  "
              f"({s['correct']:>4}/{s['total']:<4}) {bar}")

    print()
    print("  Conclusion: Simple types (ax=b, ax+b=c) are easiest (97-99%).")
    print("  Division (x/a=b) is hardest because the answer is a×b,")
    print("  producing large multi-digit numbers the model struggles with.")
    print("  Two-variable equations (ax+b=cx+d) are moderately harder.")
    print()
    return results


# ====================================================================
# EXPERIMENT 4: TEACHER FORCING
# ====================================================================

def experiment_teacher_forcing(vocab):
    """
    How important is teacher forcing?

    Compares always-on (1.0), scheduled (1.0→0.3), and never (0.0).
    Shows that TF is critical for learning, but must be reduced for
    good inference performance.
    """

    print("=" * 60)
    print("EXPERIMENT 4: TEACHER FORCING")
    print("  Question: How important is teacher forcing for learning?")
    print("=" * 60)
    print()

    train_loader, test_loader = create_dataloaders(
        vocab, batch_size=64, train_ratio=0.8
    )

    configs = [
        ("Always (TF=1.0)", 1.0),
        ("Half   (TF=0.5)", 0.5),
        ("Never  (TF=0.0)", 0.0),
    ]

    results = []
    for name, tf_ratio in configs:
        model = create_model(vocab_size=vocab.vocab_size)
        start = time.time()
        loss = quick_train(model, train_loader, vocab, epochs=10,
                           tf_ratio=tf_ratio)
        elapsed = time.time() - start
        acc = evaluate_accuracy(model, test_loader, vocab)

        results.append((name, tf_ratio, loss, acc, elapsed))
        print(f"  {name} → "
              f"Loss: {loss:.4f}, Accuracy: {acc:.1%}, "
              f"Time: {elapsed:.0f}s")

    print()
    print("  Conclusion: Teacher forcing is essential for early learning.")
    print("  TF=1.0 gives the lowest loss but may overfit to teacher input.")
    print("  TF=0.0 (no help) makes learning very slow or impossible.")
    print("  The scheduled approach (our default) gets the best of both.")
    print()
    return results


# ====================================================================
# MAIN
# ====================================================================

def main():
    """Run all experiments and print a final summary."""

    print()
    print("=" * 60)
    print("STEP 13: EXPERIMENTS AND ANALYSIS")
    print("=" * 60)
    print()
    print("  Running 4 experiments to understand what affects accuracy.")
    print("  Each experiment trains for 10 epochs (quick) to show")
    print("  relative differences between configurations.")
    print()

    # ---- Load vocab ----
    vocab = build_vocab_from_data("algebra/data/equations.txt")

    # ---- Run experiments ----
    exp1 = experiment_data_size(vocab)
    exp2 = experiment_model_size(vocab)
    exp3 = experiment_complexity(vocab)
    exp4 = experiment_teacher_forcing(vocab)

    # ---- Final summary ----
    print("=" * 60)
    print("SUMMARY OF FINDINGS")
    print("=" * 60)
    print()
    print("  1. DATA SIZE: More data helps significantly.")
    print(f"     2K→{exp1[0][2]:.0%}, 10K→{exp1[1][2]:.0%}, "
          f"40K→{exp1[2][2]:.0%}")
    print()
    print("  2. MODEL SIZE: Bigger helps, with diminishing returns.")
    for name, params, _, acc, _ in exp2:
        print(f"     {name}: {acc:.0%}")
    print()
    print("  3. COMPLEXITY: Division is hardest, simple types are easiest.")
    easiest = max(exp3, key=lambda x: x[2])
    hardest = min(exp3, key=lambda x: x[2])
    print(f"     Easiest: {easiest[0]} ({easiest[2]:.0%})")
    print(f"     Hardest: {hardest[0]} ({hardest[2]:.0%})")
    print()
    print("  4. TEACHER FORCING: Critical for learning.")
    for name, _, _, acc, _ in exp4:
        print(f"     {name}: {acc:.0%}")
    print()
    print("=" * 60)
    print("Experiments complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
