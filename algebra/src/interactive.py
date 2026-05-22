"""
interactive.py — Interactive equation solver prompt.

PURPOSE:
    Provides a REPL (Read-Eval-Print Loop) where you type equations
    and the trained model solves them in real time. This is the
    "front end" of the algebra solver — the user-facing interface.

COMMANDS:
    :help     — show available commands
    :verify   — toggle verification mode (substitute answer back)
    :batch N  — solve N random equations from the test set
    :quit     — exit the solver

HOW IT WORKS:
    1. Load the trained model and vocabulary from disk
    2. Wait for user to type an equation
    3. Encode the equation using the vocabulary
    4. Run it through the model (encoder → decoder)
    5. Decode the predicted tokens back to text
    6. Optionally verify by substituting back into the equation
    7. Print the result

INPUT:  algebra/outputs/model.pth, vocab.pth
OUTPUT: Interactive terminal session

Usage:
    PYTHONPATH=algebra/src python algebra/src/interactive.py
"""

import re
import random
import torch

from vocab import build_vocab_from_data, PAD_IDX
from seq2seq import create_model

# The model was trained with padded batches. Single equations must be
# padded to a similar length for the bidirectional encoder to produce
# the same hidden states it learned during training.
PAD_TO_LENGTH = 25


# ====================================================================
# VERIFICATION
# ====================================================================

def verify_answer(equation, answer_str):
    """
    Verify an answer by substituting x back into the equation.

    PARAMETERS:
        equation (str):   The equation (e.g., "2x + 3 = 7")
        answer_str (str): The answer (e.g., "x = 2")

    RETURNS:
        tuple: (is_correct: bool, details: str)
    """

    match = re.match(r'x\s*=\s*(-?\d+)', answer_str.strip())
    if not match:
        return False, "Could not parse answer"

    val = int(match.group(1))

    try:
        left, right = equation.split("=", 1)

        def substitute(expr, v):
            expr = expr.strip()
            expr = re.sub(r'(\d)x', rf'\1*{v}', expr)
            expr = re.sub(r'(?<!\d)x', str(v), expr)
            expr = expr.replace("- -", "+ ")
            expr = expr.replace("+ -", "- ")
            return expr

        left_val = eval(substitute(left, val))
        right_val = eval(substitute(right, val))

        if abs(left_val - right_val) < 1e-9:
            return True, f"  Check: {left.strip()} → {left_val},  {right.strip()} → {right_val}  ✓"
        else:
            return False, f"  Check: {left.strip()} → {left_val},  {right.strip()} → {right_val}  ✗"
    except Exception as e:
        return False, f"  Verification error: {e}"


# ====================================================================
# SOLVER
# ====================================================================

def solve_equation(model, vocab, equation):
    """
    Solve a single equation using the trained model.

    PARAMETERS:
        model (Seq2Seq):      The trained model
        vocab (AlgebraVocab): Vocabulary
        equation (str):       The equation to solve

    RETURNS:
        str: The predicted solution (e.g., "x = 2")
    """

    encoded = vocab.encode_with_eos(equation.strip())
    pad_len = max(PAD_TO_LENGTH, len(encoded))
    padded = encoded + [PAD_IDX] * (pad_len - len(encoded))
    src = torch.tensor([padded])

    model.eval()
    with torch.no_grad():
        preds, _ = model.solve(src)

    return vocab.decode_until_eos(preds[0].tolist())


# ====================================================================
# BATCH TESTING
# ====================================================================

def run_batch(model, vocab, data_path, count=10):
    """
    Solve random equations from the data file and show results.

    PARAMETERS:
        model (Seq2Seq):      The trained model
        vocab (AlgebraVocab): Vocabulary
        data_path (str):      Path to equations.txt
        count (int):          Number of equations to test
    """

    with open(data_path) as f:
        lines = f.readlines()

    samples = random.sample(lines, min(count, len(lines)))
    correct = 0
    total = 0

    print()
    for line in samples:
        parts = line.strip().split("\t")
        if len(parts) != 2:
            continue

        eq, expected = parts
        predicted = solve_equation(model, vocab, eq)
        match = predicted == expected
        if match:
            correct += 1
        total += 1

        status = "✓" if match else "✗"
        print(f"  {eq:<30} → {predicted:<12} "
              f"(expected: {expected:<12}) {status}")

    print()
    print(f"  Batch result: {correct}/{total} correct ({correct/total:.0%})")
    print()


# ====================================================================
# HELP TEXT
# ====================================================================

HELP_TEXT = """
  Commands:
    :help       Show this help message
    :verify     Toggle verification mode (substitution check)
    :batch N    Solve N random equations from the dataset
    :quit       Exit the solver

  Examples:
    >> 2x + 3 = 7
    >> 5x - 10 = 15
    >> 3x + 2 = x + 8
    >> x / 4 = 3
    >> 10 - 2x = 4
"""


# ====================================================================
# MAIN LOOP
# ====================================================================

def main():
    """
    Interactive equation solver REPL.

    FLOW:
        1. Load model and vocabulary
        2. Print welcome banner
        3. Loop: read input → solve → print result
        4. Handle commands (:help, :verify, :batch, :quit)
    """

    # ---- Load model ----
    print()
    print("Loading model...")

    vocab = build_vocab_from_data("algebra/data/equations.txt")

    model = create_model(vocab_size=vocab.vocab_size)
    model.load_state_dict(
        torch.load("algebra/outputs/model.pth", weights_only=True)
    )
    model.eval()

    print("Model loaded!")
    print()

    # ---- Welcome banner ----
    print("=" * 58)
    print("           AlgebrAI — Equation Solver")
    print("=" * 58)
    print("  Type any linear equation and the model will solve it.")
    print("  Type :help for commands, :quit to exit.")
    print("=" * 58)
    print()

    verify_mode = False
    data_path = "algebra/data/equations.txt"

    # ---- REPL loop ----
    while True:
        try:
            user_input = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        if not user_input:
            continue

        # ---- Commands ----
        if user_input.lower() == ":quit":
            print("  Goodbye!")
            break

        if user_input.lower() == ":help":
            print(HELP_TEXT)
            continue

        if user_input.lower() == ":verify":
            verify_mode = not verify_mode
            state = "ON" if verify_mode else "OFF"
            print(f"  Verification mode: {state}")
            print()
            continue

        if user_input.lower().startswith(":batch"):
            parts = user_input.split()
            count = int(parts[1]) if len(parts) > 1 else 10
            run_batch(model, vocab, data_path, count)
            continue

        # ---- Solve equation ----
        if "=" not in user_input:
            print("  That doesn't look like an equation (missing '=')")
            print("  Try something like: 2x + 3 = 7")
            print()
            continue

        if "x" not in user_input.lower():
            print("  No variable 'x' found in equation.")
            print("  Try something like: 2x + 3 = 7")
            print()
            continue

        prediction = solve_equation(model, vocab, user_input)
        print(f"   {prediction}")

        if verify_mode:
            is_correct, details = verify_answer(user_input, prediction)
            print(details)

        print()


if __name__ == "__main__":
    main()
