"""
generate_data.py — Create synthetic linear equations with solutions.

PURPOSE:
    Instead of collecting real math problems (expensive, limited),
    we GENERATE unlimited training data with code. This is called
    "synthetic data" — and it's a powerful technique used throughout AI.

    We create 50,000 equations like:
        "2x + 3 = 7"    →  "x = 2"
        "5x - 10 = 15"  →  "x = 5"
        "x / 4 = 3"     →  "x = 12"

    Every equation is verified by substitution before being saved:
    we plug the answer back in and check that both sides are equal.

WHY SYNTHETIC DATA?
    1. Unlimited quantity — we can make as much as we want
    2. Perfect labels — every answer is guaranteed correct
    3. Controllable difficulty — we choose the number ranges
    4. No collection cost — no humans needed to solve equations

    Real-world AI often can't do this (you can't synthetically generate
    photos of cats), but for math it works perfectly.

EQUATION TYPES GENERATED:
    Type 1:  ax = b              e.g., "3x = 12"       → "x = 4"
    Type 2:  ax + b = c          e.g., "2x + 3 = 7"    → "x = 2"
    Type 3:  ax - b = c          e.g., "5x - 10 = 15"  → "x = 5"
    Type 4:  b + ax = c          e.g., "3 + 4x = 11"   → "x = 2"
    Type 5:  b - ax = c          e.g., "10 - 2x = 4"   → "x = 3"
    Type 6:  x / a = b           e.g., "x / 4 = 3"     → "x = 12"
    Type 7:  ax + b = cx + d     e.g., "3x + 2 = x + 8" → "x = 3"

WHAT THIS FILE PROVIDES:
    1. generate_type_1() through generate_type_7() — one per equation type
    2. format_number()   — clean number formatting (no trailing .0)
    3. verify_solution() — check answer by substitution
    4. generate_dataset() — create the full dataset
    5. save_dataset()    — write to file
    6. print_statistics() — summarize what was generated
    7. main()            — run everything

INPUT:  Random number generator (no external files needed)
OUTPUT: algebra/data/equations.txt (50,000 equation-solution pairs)

Usage:
    PYTHONPATH=algebra/src python algebra/src/generate_data.py
"""

import random
import os
import re


# ====================================================================
# NUMBER FORMATTING
# ====================================================================

def format_number(n):
    """
    Format a number for clean display in equations and solutions.

    We want integers to look like integers (no ".0" suffix) and
    decimals to show only necessary digits.

    EXAMPLES:
        format_number(4)    → "4"
        format_number(4.0)  → "4"
        format_number(3.5)  → "3.5"
        format_number(-3)   → "-3"
        format_number(0)    → "0"

    WHY THIS MATTERS:
        Without this, Python would format 4.0 as "4.0", making
        equations look like "x = 4.0" instead of the cleaner "x = 4".
        The model would have to learn that "4" and "4.0" mean the same
        thing — unnecessary complexity.

    PARAMETERS:
        n (int or float): The number to format

    RETURNS:
        str: Clean string representation
    """

    # ---- If it's a whole number, show it as an integer ----
    # int(n) == n checks if 4.0 is really just 4
    if isinstance(n, float) and n == int(n):
        return str(int(n))
    elif isinstance(n, int):
        return str(n)
    else:
        # ---- For decimals, round to avoid floating point noise ----
        # Python can produce things like 3.5000000000001 due to
        # floating-point arithmetic. We round to 4 decimal places
        # and strip trailing zeros.
        formatted = f"{n:.4f}".rstrip('0').rstrip('.')
        return formatted


# ====================================================================
# EQUATION GENERATORS — one function per equation type
# ====================================================================
# Each generator returns a tuple: (equation_string, solution_string)
#
# STRATEGY: Instead of generating random equations and solving them
# (which might produce ugly numbers), we generate the ANSWER first
# and work backwards to create an equation. This guarantees clean
# integer or simple decimal answers.

def generate_type_1():
    """
    Type 1: ax = b

    Generate an equation of the form "coefficient * x = result".

    STRATEGY:
        1. Pick a random coefficient 'a' (nonzero integer, -20 to 20)
        2. Pick a random answer 'x_val' (integer, -50 to 50)
        3. Compute b = a * x_val
        4. Format as "ax = b" with solution "x = x_val"

    WHY GENERATE ANSWER FIRST?
        If we generated a=3, b=10, the answer would be x = 3.333...
        which is messy. By starting with x_val=4, we get a=3, b=12,
        giving the clean equation "3x = 12" → "x = 4".

    EXAMPLES:
        a=3, x_val=4   → "3x = 12"     → "x = 4"
        a=-2, x_val=5  → "-2x = -10"   → "x = 5"
        a=1, x_val=7   → "x = 7"       → "x = 7"

    RETURNS:
        tuple: (equation_string, solution_string)
    """

    # ---- Pick coefficient and answer ----
    a = random.choice([i for i in range(-20, 21) if i != 0])
    x_val = random.randint(-50, 50)

    # ---- Compute the right side ----
    b = a * x_val

    # ---- Format the equation ----
    # Special case: if a is 1, write "x" not "1x"
    # If a is -1, write "-x" not "-1x"
    if a == 1:
        left = "x"
    elif a == -1:
        left = "-x"
    else:
        left = f"{a}x"

    equation = f"{left} = {format_number(b)}"
    solution = f"x = {format_number(x_val)}"

    return equation, solution


def generate_type_2():
    """
    Type 2: ax + b = c

    The classic linear equation form.

    STRATEGY:
        1. Pick a, x_val, b randomly
        2. Compute c = a * x_val + b
        3. Format as "ax + b = c"

    EXAMPLES:
        a=2, x_val=2, b=3  → "2x + 3 = 7"    → "x = 2"
        a=4, x_val=3, b=1  → "4x + 1 = 13"   → "x = 3"

    RETURNS:
        tuple: (equation_string, solution_string)
    """

    a = random.choice([i for i in range(-20, 21) if i != 0])
    x_val = random.randint(-50, 50)
    b = random.randint(1, 20)

    c = a * x_val + b

    if a == 1:
        ax = "x"
    elif a == -1:
        ax = "-x"
    else:
        ax = f"{a}x"

    equation = f"{ax} + {b} = {format_number(c)}"
    solution = f"x = {format_number(x_val)}"

    return equation, solution


def generate_type_3():
    """
    Type 3: ax - b = c

    Same as Type 2 but with subtraction.

    STRATEGY:
        1. Pick a, x_val, b randomly
        2. Compute c = a * x_val - b
        3. Format as "ax - b = c"

    EXAMPLES:
        a=5, x_val=5, b=10  → "5x - 10 = 15"  → "x = 5"
        a=3, x_val=2, b=4   → "3x - 4 = 2"    → "x = 2"

    RETURNS:
        tuple: (equation_string, solution_string)
    """

    a = random.choice([i for i in range(-20, 21) if i != 0])
    x_val = random.randint(-50, 50)
    b = random.randint(1, 20)

    c = a * x_val - b

    if a == 1:
        ax = "x"
    elif a == -1:
        ax = "-x"
    else:
        ax = f"{a}x"

    equation = f"{ax} - {b} = {format_number(c)}"
    solution = f"x = {format_number(x_val)}"

    return equation, solution


def generate_type_4():
    """
    Type 4: b + ax = c

    Same math as Type 2, but the constant comes first.
    This tests whether the model understands commutativity
    (that order doesn't matter in addition).

    STRATEGY:
        1. Pick a, x_val, b randomly
        2. Compute c = b + a * x_val
        3. Format as "b + ax = c"

    EXAMPLES:
        a=4, x_val=2, b=3   → "3 + 4x = 11"   → "x = 2"
        a=2, x_val=5, b=7   → "7 + 2x = 17"   → "x = 5"

    RETURNS:
        tuple: (equation_string, solution_string)
    """

    a = random.choice([i for i in range(-20, 21) if i != 0])
    x_val = random.randint(-50, 50)
    b = random.randint(1, 20)

    c = b + a * x_val

    if a == 1:
        ax = "x"
    elif a == -1:
        ax = "-x"
    else:
        ax = f"{a}x"

    equation = f"{b} + {ax} = {format_number(c)}"
    solution = f"x = {format_number(x_val)}"

    return equation, solution


def generate_type_5():
    """
    Type 5: b - ax = c

    Subtraction with the constant first. This is trickier because
    the sign of the x term is flipped.

    STRATEGY:
        1. Pick a, x_val, b randomly
        2. Compute c = b - a * x_val
        3. Format as "b - ax = c"

    MATH TO SOLVE:
        b - ax = c
        -ax = c - b
        x = (b - c) / a

    EXAMPLES:
        a=2, x_val=3, b=10  → "10 - 2x = 4"   → "x = 3"
        a=3, x_val=1, b=8   → "8 - 3x = 5"    → "x = 1"

    RETURNS:
        tuple: (equation_string, solution_string)
    """

    a = random.choice([i for i in range(-20, 21) if i != 0])
    x_val = random.randint(-50, 50)
    b = random.randint(1, 20)

    c = b - a * x_val

    if a == 1:
        ax = "x"
    elif a == -1:
        ax = "-x"
    else:
        ax = f"{a}x"

    equation = f"{b} - {ax} = {format_number(c)}"
    solution = f"x = {format_number(x_val)}"

    return equation, solution


def generate_type_6():
    """
    Type 6: x / a = b

    Division equation. The answer is x = a * b.

    STRATEGY:
        1. Pick a (nonzero) and b randomly
        2. Compute x_val = a * b
        3. Format as "x / a = b"

    WHY NOT "ax / b = c"?
        Keeping it simple: x is always in the numerator alone.
        This is enough to teach the model about division.

    EXAMPLES:
        a=4, b=3   → "x / 4 = 3"    → "x = 12"
        a=5, b=-2  → "x / 5 = -2"   → "x = -10"

    RETURNS:
        tuple: (equation_string, solution_string)
    """

    a = random.choice([i for i in range(-20, 21) if i != 0])
    b = random.randint(-20, 20)

    x_val = a * b

    equation = f"x / {a} = {format_number(b)}"
    solution = f"x = {format_number(x_val)}"

    return equation, solution


def generate_type_7():
    """
    Type 7: ax + b = cx + d (x on both sides)

    The hardest type — requires collecting x terms on one side.

    STRATEGY:
        1. Pick a, c (different, both nonzero) and x_val
        2. Pick b randomly
        3. Compute d = a * x_val + b - c * x_val
           (which simplifies to (a-c) * x_val + b)
        4. Format as "ax + b = cx + d"

    MATH TO SOLVE:
        ax + b = cx + d
        ax - cx = d - b
        (a - c)x = d - b
        x = (d - b) / (a - c)

    EXAMPLES:
        a=3, c=1, x_val=3, b=2  → "3x + 2 = x + 8"    → "x = 3"
        a=5, c=2, x_val=4, b=1  → "5x + 1 = 2x + 13"  → "x = 4"

    RETURNS:
        tuple: (equation_string, solution_string)
    """

    # ---- Pick two DIFFERENT nonzero coefficients ----
    a = random.choice([i for i in range(-15, 16) if i != 0])
    c = random.choice([i for i in range(-15, 16) if i != 0 and i != a])
    x_val = random.randint(-20, 20)
    b = random.randint(1, 20)

    d = a * x_val + b - c * x_val

    # ---- Format coefficients ----
    if a == 1:
        ax = "x"
    elif a == -1:
        ax = "-x"
    else:
        ax = f"{a}x"

    if c == 1:
        cx = "x"
    elif c == -1:
        cx = "-x"
    else:
        cx = f"{c}x"

    equation = f"{ax} + {b} = {cx} + {format_number(d)}"
    solution = f"x = {format_number(x_val)}"

    return equation, solution


# ====================================================================
# VERIFICATION — check every answer by substitution
# ====================================================================

def verify_solution(equation, solution):
    """
    Verify a solution by substituting x back into the equation.

    This is the mathematical gold standard: if plugging the answer
    back in makes both sides equal, the answer is correct.

    HOW IT WORKS:
        1. Parse "x = 5" to get x_value = 5
        2. Split equation on "=" to get left side and right side
        3. Replace "x" with the value in both sides
        4. Evaluate both sides as math expressions
        5. Check if they're equal (within floating point tolerance)

    WHY eval()?
        We use Python's eval() to calculate expressions like "2*5 + 3".
        This is safe here because we control the input (we generated it).
        In production code, you'd use a proper math parser.

    PARAMETERS:
        equation (str): The equation, e.g., "2x + 3 = 7"
        solution (str): The solution, e.g., "x = 2"

    RETURNS:
        bool: True if the solution is correct
    """

    # ---- Parse the x value from the solution ----
    # "x = 2" → 2, "x = -3.5" → -3.5
    x_str = solution.split("=")[1].strip()
    x_value = float(x_str)

    # ---- Split equation into left and right sides ----
    sides = equation.split("=")
    left_str = sides[0].strip()
    right_str = sides[1].strip()

    # ---- Prepare expressions for evaluation ----
    # Replace "x" with the actual value
    # "2x" needs to become "2*5" (add explicit multiplication)
    # We handle cases like "-x", "3x", just "x"
    def prepare_expr(expr, x_val):
        # Add explicit multiplication: "2x" → "2*x", "-3x" → "-3*x"
        # But don't touch things like "x /" (already separated by space)
        result = re.sub(r'(\d)x', r'\1*x', expr)
        # Handle standalone "-x" (no digit before x)
        result = result.replace('-x', '-1*x')
        # Handle standalone "x" at the start
        if result.startswith('x'):
            result = '1*x' + result[1:]
        # Replace remaining standalone "x" (after operators)
        result = re.sub(r'(?<=[\s+\-*/=(])x', '1*x', result)
        # Finally replace x with value
        result = result.replace('x', str(x_val))
        return result

    left_expr = prepare_expr(left_str, x_value)
    right_expr = prepare_expr(right_str, x_value)

    # ---- Evaluate both sides ----
    try:
        left_val = eval(left_expr)
        right_val = eval(right_expr)
    except Exception:
        return False

    # ---- Compare with tolerance for floating-point errors ----
    return abs(left_val - right_val) < 1e-6


# ====================================================================
# DATASET GENERATION
# ====================================================================

# ---- Map of type number to generator function ----
# This makes it easy to randomly pick a type and call the right function.
GENERATORS = {
    1: generate_type_1,
    2: generate_type_2,
    3: generate_type_3,
    4: generate_type_4,
    5: generate_type_5,
    6: generate_type_6,
    7: generate_type_7,
}

# ---- Weights for each type ----
# Types 2 and 3 are the most common linear equation forms,
# so we generate more of them. Type 7 (x on both sides) is
# harder, so we include a decent amount for the model to learn.
TYPE_WEIGHTS = {
    1: 0.12,    # ax = b          (simple)
    2: 0.18,    # ax + b = c      (classic)
    3: 0.18,    # ax - b = c      (classic with subtraction)
    4: 0.12,    # b + ax = c      (reversed order)
    5: 0.12,    # b - ax = c      (reversed with subtraction)
    6: 0.10,    # x / a = b       (division)
    7: 0.18,    # ax + b = cx + d (x on both sides — hardest)
}


def generate_dataset(num_equations=50000, seed=42):
    """
    Generate the full dataset of equation-solution pairs.

    PROCESS:
        1. For each equation, randomly pick a type (weighted)
        2. Call the corresponding generator function
        3. Verify the solution by substitution
        4. Check for duplicates (skip if already seen)
        5. Repeat until we have the desired number

    WHY USE A SEED?
        random.seed(42) makes the random number generator produce
        the same sequence every time. This means:
        - Running the script twice gives identical data
        - Results are reproducible (crucial for science)
        - You can share the script and others get the same dataset

    PARAMETERS:
        num_equations (int): How many equations to generate (default 50,000)
        seed (int):          Random seed for reproducibility

    RETURNS:
        list: List of (equation, solution, type_number) tuples
    """

    random.seed(seed)

    # ---- Build weighted type list for random.choices() ----
    types = list(TYPE_WEIGHTS.keys())
    weights = list(TYPE_WEIGHTS.values())

    dataset = []
    seen = set()
    failed_verification = 0
    duplicates_skipped = 0

    print(f"  Generating {num_equations:,} equations...")

    while len(dataset) < num_equations:
        # ---- Pick a random equation type ----
        eq_type = random.choices(types, weights=weights, k=1)[0]

        # ---- Generate the equation ----
        equation, solution = GENERATORS[eq_type]()

        # ---- Skip duplicates ----
        if equation in seen:
            duplicates_skipped += 1
            continue
        seen.add(equation)

        # ---- Verify by substitution ----
        if not verify_solution(equation, solution):
            failed_verification += 1
            continue

        dataset.append((equation, solution, eq_type))

        # ---- Progress update every 10,000 ----
        if len(dataset) % 10000 == 0:
            print(f"    Generated {len(dataset):,} / {num_equations:,}")

    # ---- Shuffle the dataset ----
    # We don't want all Type 1 equations first, then all Type 2, etc.
    # Shuffling ensures the model sees a mix during training.
    random.shuffle(dataset)

    print(f"    Done! Generated {len(dataset):,} equations")
    if duplicates_skipped > 0:
        print(f"    Skipped {duplicates_skipped:,} duplicates")
    if failed_verification > 0:
        print(f"    WARNING: {failed_verification:,} failed verification")

    return dataset


# ====================================================================
# SAVE AND STATISTICS
# ====================================================================

def save_dataset(dataset, filepath="algebra/data/equations.txt"):
    """
    Save the dataset to a tab-separated file.

    FORMAT:
        Each line is: equation<TAB>solution
        Example: "2x + 3 = 7\tx = 2"

    WHY TAB-SEPARATED?
        Equations contain spaces ("2x + 3 = 7"), so we can't use
        space as a delimiter. Tab is safe because it never appears
        in equations.

    PARAMETERS:
        dataset (list): List of (equation, solution, type) tuples
        filepath (str):  Output file path
    """

    # ---- Create directory if needed ----
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        for equation, solution, _ in dataset:
            f.write(f"{equation}\t{solution}\n")

    size_kb = os.path.getsize(filepath) / 1024
    print(f"  Saved to {filepath} ({size_kb:.1f} KB)")


def print_statistics(dataset):
    """
    Print detailed statistics about the generated dataset.

    Shows:
        - Total count
        - Count per equation type
        - Answer distribution (positive, negative, zero, decimal)
        - Sample equations from each type
        - Character set used

    PARAMETERS:
        dataset (list): List of (equation, solution, type) tuples
    """

    print()
    print("  " + "=" * 56)
    print("  DATASET STATISTICS")
    print("  " + "=" * 56)
    print(f"  Total equations: {len(dataset):,}")
    print()

    # ---- Count per type ----
    type_counts = {}
    for _, _, eq_type in dataset:
        type_counts[eq_type] = type_counts.get(eq_type, 0) + 1

    type_names = {
        1: "ax = b",
        2: "ax + b = c",
        3: "ax - b = c",
        4: "b + ax = c",
        5: "b - ax = c",
        6: "x / a = b",
        7: "ax + b = cx + d",
    }

    print("  Equations per type:")
    print("  " + "-" * 56)
    for t in sorted(type_counts.keys()):
        count = type_counts[t]
        pct = count / len(dataset) * 100
        print(f"    Type {t} ({type_names[t]:<18}): {count:>6,} ({pct:.1f}%)")

    # ---- Answer distribution ----
    positive = 0
    negative = 0
    zero = 0
    decimal = 0

    for _, solution, _ in dataset:
        x_str = solution.split("=")[1].strip()
        x_val = float(x_str)
        if x_val > 0:
            positive += 1
        elif x_val < 0:
            negative += 1
        else:
            zero += 1
        if '.' in x_str:
            decimal += 1

    print()
    print("  Answer distribution:")
    print("  " + "-" * 56)
    print(f"    Positive answers : {positive:>6,} ({positive/len(dataset)*100:.1f}%)")
    print(f"    Negative answers : {negative:>6,} ({negative/len(dataset)*100:.1f}%)")
    print(f"    Zero answers     : {zero:>6,} ({zero/len(dataset)*100:.1f}%)")
    print(f"    Decimal answers  : {decimal:>6,} ({decimal/len(dataset)*100:.1f}%)")

    # ---- Sample equations ----
    print()
    print("  Sample equations (2 per type):")
    print("  " + "-" * 56)

    type_samples = {}
    for eq, sol, eq_type in dataset:
        if eq_type not in type_samples:
            type_samples[eq_type] = []
        if len(type_samples[eq_type]) < 2:
            type_samples[eq_type].append((eq, sol))

    for t in sorted(type_samples.keys()):
        print(f"    Type {t} ({type_names[t]}):")
        for eq, sol in type_samples[t]:
            print(f"      {eq:<24} → {sol}")

    # ---- Character set ----
    all_chars = set()
    for eq, sol, _ in dataset:
        all_chars.update(eq)
        all_chars.update(sol)

    print()
    print(f"  Unique characters used: {len(all_chars)}")
    print(f"  Characters: {sorted(all_chars)}")

    # ---- Equation length stats ----
    eq_lengths = [len(eq) for eq, _, _ in dataset]
    sol_lengths = [len(sol) for _, sol, _ in dataset]

    print()
    print("  Sequence lengths:")
    print("  " + "-" * 56)
    print(f"    Equation:  min={min(eq_lengths)}, max={max(eq_lengths)}, "
          f"avg={sum(eq_lengths)/len(eq_lengths):.1f}")
    print(f"    Solution:  min={min(sol_lengths)}, max={max(sol_lengths)}, "
          f"avg={sum(sol_lengths)/len(sol_lengths):.1f}")


# ====================================================================
# MAIN
# ====================================================================

def main():
    """
    Main function — generate, verify, save, and report.

    FLOW:
        1. Generate 50,000 equations with solutions
        2. Save to algebra/data/equations.txt
        3. Print statistics about the dataset
    """

    print("=" * 60)
    print("STEP 02: SYNTHETIC DATA GENERATION")
    print("=" * 60)
    print()
    print("Generating linear equations with verified solutions...")
    print()

    # ---- Generate ----
    dataset = generate_dataset(num_equations=50000, seed=42)

    # ---- Save ----
    print()
    save_dataset(dataset, filepath="algebra/data/equations.txt")

    # ---- Statistics ----
    print_statistics(dataset)

    print()
    print("  " + "=" * 56)
    print("  Data generation complete!")
    print("  " + "=" * 56)


# ======================================================================
# ENTRY POINT
# ======================================================================
if __name__ == "__main__":
    main()
