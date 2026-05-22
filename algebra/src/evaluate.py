"""
evaluate.py — Evaluate the trained algebra solver on the test set.

PURPOSE:
    Run the trained model on the 10,000 held-out test equations and
    measure how well it performs. This is the "final exam" — the model
    has never seen these equations during training.

WHAT IT MEASURES:
    1. Overall exact-match accuracy (target: ≥ 90%)
    2. Per-type accuracy (target: all types ≥ 80%)
    3. Accuracy on negative vs positive vs zero answers
    4. Error analysis — examples of wrong predictions
    5. Substitution verification — plug answers back into equations

EQUATION TYPES:
    Type 1: ax = b              (e.g., "3x = 12")
    Type 2: ax + b = c          (e.g., "2x + 3 = 7")
    Type 3: ax - b = c          (e.g., "5x - 10 = 15")
    Type 4: b + ax = c          (e.g., "4 + 3x = 13")
    Type 5: b - ax = c          (e.g., "10 - 2x = 4")
    Type 6: x / a = b           (e.g., "x / 4 = 3")
    Type 7: ax + b = cx + d     (e.g., "3x + 2 = x + 8")

WHAT THIS FILE PROVIDES:
    1. classify_equation()       — identify equation type from string
    2. parse_answer()            — extract numeric value from "x = N"
    3. verify_by_substitution()  — plug answer back into equation
    4. evaluate_full()           — comprehensive evaluation
    5. main()                    — run evaluation and print report

INPUT:  algebra/outputs/model.pth, vocab.pth
OUTPUT: Printed evaluation report

Usage:
    PYTHONPATH=algebra/src python algebra/src/evaluate.py
"""

import re
import torch

from vocab import build_vocab_from_data
from dataset import create_dataloaders
from seq2seq import create_model


# ====================================================================
# EQUATION CLASSIFICATION
# ====================================================================

def classify_equation(equation):
    """
    Classify an equation into one of 7 types based on its structure.

    We use pattern matching on the equation string to determine the type.
    The patterns are matched in order — Type 7 is checked first because
    it has 'x' on both sides, which would also match other patterns.

    PARAMETERS:
        equation (str): The equation string (e.g., "2x + 3 = 7")

    RETURNS:
        str: Type label (e.g., "Type 2: ax + b = c")
    """

    eq = equation.strip()
    left, right = eq.split("=", 1)
    left = left.strip()
    right = right.strip()

    # Type 7: x appears on both sides (ax + b = cx + d)
    if "x" in left and "x" in right:
        return "Type 7: ax + b = cx + d"

    # Type 6: x / a = b
    if "/" in left:
        return "Type 6: x / a = b"

    # Type 1: ax = b (just ax on the left, no + or -)
    # Match patterns like "3x", "-x", "-3x" with no operator after
    if re.match(r'^-?\d*x$', left):
        return "Type 1: ax = b"

    # Types 2-5: have an operator in the left side
    # Type 4: b + ax = c (number before x term, with +)
    # Type 5: b - ax = c (number before x term, with -)
    # We check if the equation starts with a digit (not x term)
    if re.match(r'^-?\d+\s*[\+\-]', left) and "x" in left:
        if re.search(r'\+\s*-?\d*x', left):
            return "Type 4: b + ax = c"
        if re.search(r'-\s*-?\d*x', left):
            return "Type 5: b - ax = c"

    # Type 2: ax + b = c
    if re.search(r'x.*\+', left):
        return "Type 2: ax + b = c"

    # Type 3: ax - b = c
    if re.search(r'x.*-', left):
        return "Type 3: ax - b = c"

    return "Other"


# ====================================================================
# ANSWER PARSING
# ====================================================================

def parse_answer(answer_str):
    """
    Extract the numeric value from an answer string like "x = 2".

    PARAMETERS:
        answer_str (str): Answer string (e.g., "x = 2", "x = -13")

    RETURNS:
        int or None: The numeric value, or None if parsing fails
    """

    match = re.match(r'x\s*=\s*(-?\d+)', answer_str.strip())
    if match:
        return int(match.group(1))
    return None


# ====================================================================
# SUBSTITUTION VERIFICATION
# ====================================================================

def verify_by_substitution(equation, answer_value):
    """
    Verify an answer by substituting x back into the equation.

    Replaces 'x' with the answer value and evaluates both sides.
    Returns True if both sides are equal.

    PARAMETERS:
        equation (str):    The equation (e.g., "2x + 3 = 7")
        answer_value (int): The answer to verify (e.g., 2)

    RETURNS:
        bool: True if the answer is correct by substitution
    """

    try:
        left, right = equation.split("=", 1)

        # Replace 'x' with the value, handling implicit multiplication
        # "2x" → "2*5", "-x" → "-1*5", "x" → "5"
        def substitute(expr, val):
            expr = expr.strip()
            # Handle "x / a" → "val / a"
            expr = re.sub(r'(\d)x', rf'\1*{val}', expr)
            expr = re.sub(r'(?<!\d)x', str(val), expr)
            # Handle double negatives: "- -5" → "+ 5"
            expr = expr.replace("- -", "+ ")
            expr = expr.replace("+ -", "- ")
            return expr

        left_val = eval(substitute(left, answer_value))
        right_val = eval(substitute(right, answer_value))

        return abs(left_val - right_val) < 1e-9
    except Exception:
        return False


# ====================================================================
# FULL EVALUATION
# ====================================================================

def evaluate_full(model, test_loader, vocab):
    """
    Run comprehensive evaluation on the test set.

    Computes:
    - Overall exact-match accuracy
    - Per-type accuracy breakdown
    - Accuracy by answer sign (positive/negative/zero)
    - Collects error examples for analysis
    - Verifies correct predictions by substitution

    PARAMETERS:
        model (Seq2Seq):       The trained model
        test_loader (DataLoader): Test data batches
        vocab (AlgebraVocab):  Vocabulary for decoding

    RETURNS:
        dict: Full evaluation results
    """

    model.eval()

    # Tracking
    total = 0
    correct = 0
    type_stats = {}       # {type_name: {"correct": N, "total": N}}
    sign_stats = {        # accuracy by answer sign
        "positive": {"correct": 0, "total": 0},
        "negative": {"correct": 0, "total": 0},
        "zero": {"correct": 0, "total": 0},
    }
    errors = []           # wrong predictions for analysis
    substitution_verified = 0
    substitution_total = 0

    with torch.no_grad():
        for enc_input, dec_input, dec_target in test_loader:
            predictions, _ = model.solve(enc_input)

            for i in range(enc_input.shape[0]):
                # Decode strings
                eq_text = vocab.decode_until_eos(enc_input[i].tolist())
                pred_text = vocab.decode_until_eos(predictions[i].tolist())
                target_text = vocab.decode_until_eos(dec_target[i].tolist())

                # Classify equation type
                eq_type = classify_equation(eq_text)
                if eq_type not in type_stats:
                    type_stats[eq_type] = {"correct": 0, "total": 0}
                type_stats[eq_type]["total"] += 1

                # Classify answer sign
                target_val = parse_answer(target_text)
                if target_val is not None:
                    if target_val > 0:
                        sign_key = "positive"
                    elif target_val < 0:
                        sign_key = "negative"
                    else:
                        sign_key = "zero"
                    sign_stats[sign_key]["total"] += 1

                # Check match
                is_correct = pred_text == target_text
                total += 1

                if is_correct:
                    correct += 1
                    type_stats[eq_type]["correct"] += 1
                    if target_val is not None:
                        sign_stats[sign_key]["correct"] += 1

                    # Verify by substitution
                    pred_val = parse_answer(pred_text)
                    if pred_val is not None:
                        substitution_total += 1
                        if verify_by_substitution(eq_text, pred_val):
                            substitution_verified += 1
                else:
                    if target_val is not None:
                        sign_stats[sign_key]["total"] += 0  # already counted

                    # Collect error examples (up to 20)
                    if len(errors) < 20:
                        errors.append({
                            "equation": eq_text,
                            "predicted": pred_text,
                            "expected": target_text,
                            "type": eq_type,
                        })

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0,
        "type_stats": type_stats,
        "sign_stats": sign_stats,
        "errors": errors,
        "substitution_verified": substitution_verified,
        "substitution_total": substitution_total,
    }


# ====================================================================
# REPORT PRINTER
# ====================================================================

def print_report(results):
    """
    Print a formatted evaluation report.

    PARAMETERS:
        results (dict): Results from evaluate_full()
    """

    print("=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print()

    # ---- Overall accuracy ----
    acc = results["accuracy"]
    total = results["total"]
    correct = results["correct"]
    target_met = "PASS" if acc >= 0.90 else "FAIL"

    print(f"  Overall accuracy: {acc:.1%} ({correct:,} / {total:,})")
    print(f"  Target (≥ 90%):   {target_met}")
    print()

    # ---- Per-type breakdown ----
    print("  Per-type accuracy:")
    print("  " + "-" * 50)

    type_pass_count = 0
    type_total_count = 0
    for type_name in sorted(results["type_stats"].keys()):
        stats = results["type_stats"][type_name]
        type_acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        type_target = "✓" if type_acc >= 0.80 else "✗"
        type_total_count += 1
        if type_acc >= 0.80:
            type_pass_count += 1
        print(f"    {type_name:<28} {type_acc:>6.1%}  "
              f"({stats['correct']:>4}/{stats['total']:<4}) {type_target}")

    print(f"\n  All types ≥ 80%: {type_pass_count}/{type_total_count} passed")
    print()

    # ---- By answer sign ----
    print("  Accuracy by answer sign:")
    print("  " + "-" * 50)

    for sign in ["positive", "negative", "zero"]:
        stats = results["sign_stats"][sign]
        if stats["total"] > 0:
            sign_acc = stats["correct"] / stats["total"]
            print(f"    {sign:<12} {sign_acc:>6.1%}  "
                  f"({stats['correct']:>4}/{stats['total']:<4})")
        else:
            print(f"    {sign:<12}   N/A   (0 examples)")
    print()

    # ---- Substitution verification ----
    sub_v = results["substitution_verified"]
    sub_t = results["substitution_total"]
    if sub_t > 0:
        print(f"  Substitution verification: {sub_v}/{sub_t} correct "
              f"predictions verified ({sub_v/sub_t:.1%})")
    print()

    # ---- Error analysis ----
    errors = results["errors"]
    if errors:
        print(f"  Error analysis ({len(errors)} examples):")
        print("  " + "-" * 50)

        for err in errors[:15]:
            print(f"    {err['equation']:<30} "
                  f"predicted: {err['predicted']:<12} "
                  f"expected: {err['expected']:<12}")

        # Summarize error types
        print()
        print("  Error patterns:")
        print("  " + "-" * 50)

        off_by_one = 0
        wrong_sign = 0
        garbled = 0

        for err in errors:
            pred_val = parse_answer(err["predicted"])
            exp_val = parse_answer(err["expected"])

            if pred_val is not None and exp_val is not None:
                diff = abs(pred_val - exp_val)
                if diff == 1:
                    off_by_one += 1
                elif (pred_val > 0 and exp_val < 0) or \
                     (pred_val < 0 and exp_val > 0):
                    wrong_sign += 1
            elif pred_val is None:
                garbled += 1

        if off_by_one:
            print(f"    Off-by-one errors:  {off_by_one}")
        if wrong_sign:
            print(f"    Wrong sign errors:  {wrong_sign}")
        if garbled:
            print(f"    Garbled output:     {garbled}")
        other = len(errors) - off_by_one - wrong_sign - garbled
        if other:
            print(f"    Other errors:       {other}")
    else:
        print("  No errors found — perfect accuracy!")

    print()
    print("=" * 60)


# ====================================================================
# MAIN
# ====================================================================

def main():
    """
    Load trained model, run evaluation, print report.

    FLOW:
        1. Load vocabulary and create dataloaders
        2. Create model and load trained weights
        3. Run comprehensive evaluation on test set
        4. Print formatted report
    """

    print("=" * 60)
    print("STEP 09: EVALUATION AND ACCURACY")
    print("=" * 60)
    print()

    # ---- Load vocabulary ----
    print("Loading vocabulary...")
    vocab = build_vocab_from_data("algebra/data/equations.txt")
    print(f"  Vocabulary: {vocab.vocab_size} tokens")

    # ---- Create dataloaders ----
    print("Creating dataloaders...")
    train_loader, test_loader = create_dataloaders(
        vocab, batch_size=64, train_ratio=0.8
    )

    # ---- Load trained model ----
    print("Loading trained model...")
    model = create_model(vocab_size=vocab.vocab_size)
    model.load_state_dict(
        torch.load("algebra/outputs/model.pth", weights_only=True)
    )
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    print()

    # ---- Run evaluation ----
    print("Evaluating on test set...")
    print("-" * 60)
    results = evaluate_full(model, test_loader, vocab)
    print()

    # ---- Print report ----
    print_report(results)


if __name__ == "__main__":
    main()
