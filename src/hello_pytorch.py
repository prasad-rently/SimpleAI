"""
hello_pytorch.py — Your first PyTorch script!

Run this to verify PyTorch is installed correctly and to get a feel
for "tensors", the fundamental building block of all AI models.

Usage:
    python src/hello_pytorch.py
"""

import torch


def main():
    # ------------------------------------------------------------------
    # 1. CHECK INSTALLATION
    # ------------------------------------------------------------------
    # If this prints without error, PyTorch is installed and working.
    print("=" * 50)
    print("PYTORCH INSTALLATION CHECK")
    print("=" * 50)
    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    # CUDA = GPU support. We don't need it — our model is tiny enough
    # for CPU. But it's good to know whether your machine has it.
    print()

    # ------------------------------------------------------------------
    # 2. WHAT IS A TENSOR?
    # ------------------------------------------------------------------
    # A tensor is just a container for numbers. Think of it like:
    #   - A single number        → 0-dimensional tensor (scalar)
    #   - A list of numbers      → 1-dimensional tensor (vector)
    #   - A grid of numbers      → 2-dimensional tensor (matrix)
    #   - A cube of numbers      → 3-dimensional tensor
    #
    # AI models store ALL their data as tensors — input text, internal
    # weights, output predictions — everything is just numbers in tensors.

    print("=" * 50)
    print("TENSOR BASICS")
    print("=" * 50)

    # Scalar — a single number
    scalar = torch.tensor(42)
    print(f"Scalar          : {scalar}")
    print(f"  Shape         : {scalar.shape}")
    # shape = () means zero dimensions — it's just one number.
    print()

    # Vector — a list of numbers (1D)
    vector = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    print(f"Vector          : {vector}")
    print(f"  Shape         : {vector.shape}")
    # shape = (5,) means 1 dimension with 5 elements.
    print()

    # Matrix — a grid of numbers (2D)
    matrix = torch.tensor([
        [1, 2, 3],
        [4, 5, 6],
    ])
    print(f"Matrix          :\n{matrix}")
    print(f"  Shape         : {matrix.shape}")
    # shape = (2, 3) means 2 rows and 3 columns.
    print()

    # ------------------------------------------------------------------
    # 3. MATH WITH TENSORS
    # ------------------------------------------------------------------
    # PyTorch can do math on entire tensors at once — no loops needed.
    # This is called "vectorized operations" and it's MUCH faster than
    # doing math one number at a time.

    print("=" * 50)
    print("TENSOR MATH")
    print("=" * 50)

    a = torch.tensor([10.0, 20.0, 30.0])
    b = torch.tensor([1.0, 2.0, 3.0])

    print(f"a               : {a}")
    print(f"b               : {b}")
    print(f"a + b           : {a + b}")        # Element-wise addition
    print(f"a * b           : {a * b}")        # Element-wise multiplication
    print(f"a.mean()        : {a.mean()}")     # Average of all elements
    print(f"a.sum()         : {a.sum()}")      # Sum of all elements
    print()

    # ------------------------------------------------------------------
    # 4. RANDOM TENSORS
    # ------------------------------------------------------------------
    # Neural networks START with random numbers for their internal
    # weights. Training adjusts these random numbers until the model
    # produces useful output. This is a key insight:
    #
    #   AI models begin as random noise → training shapes them into
    #   something that understands patterns.

    print("=" * 50)
    print("RANDOM TENSORS (how models start)")
    print("=" * 50)

    random_weights = torch.randn(3, 4)
    # randn = random numbers from a "normal distribution" (bell curve).
    # Shape (3, 4) = 3 rows, 4 columns = 12 random numbers.
    print(f"Random 'weights' (3x4 matrix):\n{random_weights}")
    print(f"  Shape         : {random_weights.shape}")
    print()
    print("These random numbers are like a newborn brain — no knowledge")
    print("yet. Training will adjust them until they encode useful patterns.")
    print()

    # ------------------------------------------------------------------
    # 5. WHY THIS MATTERS
    # ------------------------------------------------------------------
    print("=" * 50)
    print("WHY THIS MATTERS FOR AI")
    print("=" * 50)
    print("""
In the next steps, we'll use tensors to:

  1. Convert text characters into numbers     (tensors of integers)
  2. Store the model's learned knowledge       (tensors of weights)
  3. Calculate how wrong the model's guesses   (a single tensor: the loss)
     are
  4. Generate new text from predictions        (tensors of probabilities)

Everything in AI is just math on tensors. That's the big secret.
""")

    print("PyTorch is working! You're ready for Step 03.")


if __name__ == "__main__":
    main()
