"""
hello_pytorch.py — Your first PyTorch script!

PURPOSE:
    Verify that PyTorch is installed correctly and introduce you to
    "tensors" — the fundamental building block of ALL AI models.

WHAT IS PYTORCH?
    PyTorch is an open-source library created by Meta (Facebook) that
    makes it easy to build and train neural networks. It's used by
    researchers and companies worldwide (including OpenAI, Tesla, etc.).

    Think of PyTorch as a super-powered calculator that can:
    - Work with massive arrays of numbers (tensors) very fast
    - Automatically figure out how to improve a model (autograd)
    - Run on GPUs for even faster computation

WHAT THIS SCRIPT DOES:
    1. Checks that PyTorch is installed → prints version info
    2. Creates different types of tensors → shows shapes
    3. Performs math on tensors → demonstrates vectorized operations
    4. Creates random tensors → shows how models start
    5. Summarizes why this matters → connects to future steps

INPUT:  None (no arguments needed)
OUTPUT: Printed text showing tensor examples and math operations

Usage:
    python src/hello_pytorch.py
"""

# ======================================================================
# IMPORTS
# ======================================================================
# 'torch' is the main PyTorch library. By convention, everyone imports
# it as 'torch' (not 'pytorch'). This single import gives us access to:
#   - torch.tensor()   → create tensors
#   - torch.randn()    → create random tensors
#   - torch.nn         → neural network layers (used in later steps)
#   - torch.optim      → optimizers for training (used in later steps)
import torch


def main():
    """
    Main function that runs all demonstrations.

    WHY USE A main() FUNCTION?
    Instead of putting code at the top level of the file, we wrap it
    in a function. This is a Python best practice because:
    - It keeps code organized
    - It prevents code from running when this file is imported by
      another file (only runs when executed directly)
    - Variables inside the function don't "leak" into global scope
    """

    # ==================================================================
    # SECTION 1: CHECK INSTALLATION
    # ==================================================================
    # The simplest test — if we can import torch and print its version,
    # the installation is working. If this fails, you need to run:
    #   pip install -r requirements.txt
    #
    # EXPECTED OUTPUT:
    #   PyTorch version : 2.x.x  (some version number)
    #   CUDA available  : False   (True if you have an NVIDIA GPU)

    print("=" * 50)
    print("PYTORCH INSTALLATION CHECK")
    print("=" * 50)

    # torch.__version__ is a string like "2.12.0" that tells us which
    # version of PyTorch is installed.
    print(f"PyTorch version : {torch.__version__}")

    # CUDA is NVIDIA's technology for running code on GPUs (graphics
    # cards). GPUs can do math ~100x faster than CPUs for AI workloads.
    # We don't need it for this project — our model is tiny enough for
    # CPU. But professional AI models (like ChatGPT) NEED GPUs.
    #
    # torch.cuda.is_available() returns:
    #   True  → you have an NVIDIA GPU with CUDA drivers installed
    #   False → you're using CPU only (totally fine for us)
    print(f"CUDA available  : {torch.cuda.is_available()}")
    print()

    # ==================================================================
    # SECTION 2: WHAT IS A TENSOR?
    # ==================================================================
    # A tensor is just a container for numbers. The name sounds fancy,
    # but it's really just a generalization of familiar concepts:
    #
    #   MATH NAME       DIMENSIONS    PYTHON EQUIVALENT    EXAMPLE
    #   ─────────       ──────────    ─────────────────    ───────
    #   Scalar          0D            a single number      42
    #   Vector          1D            a list               [1, 2, 3]
    #   Matrix          2D            list of lists        [[1,2], [3,4]]
    #   3D Tensor       3D            list of matrices     (think: a cube)
    #
    # WHY TENSORS INSTEAD OF REGULAR PYTHON LISTS?
    # 1. Speed: PyTorch tensors use optimized C code under the hood
    # 2. GPU:   Tensors can be moved to GPU for faster math
    # 3. Autograd: PyTorch can track operations on tensors to
    #    automatically compute gradients (crucial for training)
    #
    # AI models store ALL their data as tensors — input text, internal
    # weights, output predictions — everything is just numbers in tensors.

    print("=" * 50)
    print("TENSOR BASICS")
    print("=" * 50)

    # --- Scalar (0D tensor) ---
    # A scalar is a single number wrapped in a tensor.
    # torch.tensor(42) creates a 0-dimensional tensor containing 42.
    #
    # INPUT:  the number 42
    # OUTPUT: tensor(42) with shape torch.Size([])
    #
    # The empty shape [] means "zero dimensions" — just a lone number.
    scalar = torch.tensor(42)
    print(f"Scalar          : {scalar}")
    print(f"  Shape         : {scalar.shape}")
    print(f"  Value         : {scalar.item()}")
    # .item() extracts the plain Python number from a scalar tensor.
    # scalar = tensor(42), scalar.item() = 42 (plain int)
    print()

    # --- Vector (1D tensor) ---
    # A vector is a list of numbers. In AI, vectors are everywhere:
    # - A word might be represented as a vector of 100 numbers
    # - The model's prediction might be a vector of probabilities
    #
    # INPUT:  a Python list [1.0, 2.0, 3.0, 4.0, 5.0]
    # OUTPUT: tensor([1., 2., 3., 4., 5.]) with shape torch.Size([5])
    #
    # Shape (5,) means: 1 dimension containing 5 elements.
    # Note: we use 1.0 (floats) instead of 1 (ints) because neural
    # networks work with floating-point numbers for precision.
    vector = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    print(f"Vector          : {vector}")
    print(f"  Shape         : {vector.shape}")
    print(f"  Num elements  : {vector.numel()}")
    # .numel() = "number of elements" — counts all numbers in the tensor
    print()

    # --- Matrix (2D tensor) ---
    # A matrix is a grid (table) of numbers with rows and columns.
    # In AI, the model's "weights" (learned knowledge) are stored as
    # matrices. When we say "a model has 1 billion parameters", we mean
    # it has 1 billion numbers spread across many matrices.
    #
    # INPUT:  a nested Python list [[1,2,3], [4,5,6]]
    # OUTPUT: tensor([[1, 2, 3],
    #                 [4, 5, 6]]) with shape torch.Size([2, 3])
    #
    # Shape (2, 3) means: 2 rows and 3 columns = 6 numbers total.
    matrix = torch.tensor([
        [1, 2, 3],
        [4, 5, 6],
    ])
    print(f"Matrix          :\n{matrix}")
    print(f"  Shape         : {matrix.shape}")
    print(f"  Num elements  : {matrix.numel()}")
    # Even though it's 2D, numel() counts ALL numbers: 2 * 3 = 6
    print(f"  Row 0         : {matrix[0]}")
    # matrix[0] gives the first row: tensor([1, 2, 3])
    # matrix[1] gives the second row: tensor([4, 5, 6])
    # matrix[0][2] gives row 0, column 2: tensor(3)
    print(f"  Element [0][2]: {matrix[0][2]}")
    print()

    # ==================================================================
    # SECTION 3: MATH WITH TENSORS
    # ==================================================================
    # PyTorch can do math on entire tensors at once — no loops needed.
    # This is called "vectorized operations" and it's MUCH faster than
    # doing math one number at a time with a for-loop.
    #
    # EXAMPLE — adding two lists WITHOUT PyTorch (slow way):
    #   result = []
    #   for i in range(len(a)):
    #       result.append(a[i] + b[i])
    #
    # WITH PyTorch (fast way):
    #   result = a + b    ← does the same thing, but ~100x faster
    #
    # Why? PyTorch uses optimized C/C++ code and can even use your GPU.
    # For our tiny project this doesn't matter much, but for real AI
    # models processing millions of numbers, it's essential.

    print("=" * 50)
    print("TENSOR MATH")
    print("=" * 50)

    a = torch.tensor([10.0, 20.0, 30.0])
    b = torch.tensor([1.0, 2.0, 3.0])

    print(f"a               : {a}")
    print(f"b               : {b}")
    print()

    # --- Element-wise addition ---
    # Each element in 'a' is added to the corresponding element in 'b':
    #   [10+1, 20+2, 30+3] = [11, 22, 33]
    print(f"a + b           : {a + b}")

    # --- Element-wise multiplication ---
    # Each element in 'a' is multiplied by the corresponding element in 'b':
    #   [10*1, 20*2, 30*3] = [10, 40, 90]
    print(f"a * b           : {a * b}")

    # --- Aggregation operations ---
    # These reduce an entire tensor down to a single number:
    #   mean = average of all elements = (10+20+30)/3 = 20.0
    #   sum  = total of all elements   = 10+20+30     = 60.0
    print(f"a.mean()        : {a.mean()}")
    print(f"a.sum()         : {a.sum()}")
    print()

    # --- Why this matters for AI ---
    # During training, the model does millions of these operations:
    # - Multiply inputs by weights (matrix multiplication)
    # - Add biases (element-wise addition)
    # - Compute average loss (mean)
    # All of this happens on tensors, using these same operations.

    # ==================================================================
    # SECTION 4: RANDOM TENSORS
    # ==================================================================
    # Neural networks START with random numbers for their internal
    # weights. Training adjusts these random numbers until the model
    # produces useful output. This is a key insight:
    #
    #   AI models begin as random noise → training shapes them into
    #   something that understands patterns.
    #
    # It's like giving a baby random scribbles and slowly teaching them
    # to write letters through practice and correction.

    print("=" * 50)
    print("RANDOM TENSORS (how models start)")
    print("=" * 50)

    # torch.randn(rows, cols) creates a matrix filled with random
    # numbers drawn from a "normal distribution" (bell curve).
    #
    # Normal distribution means:
    # - Most numbers will be close to 0 (between -1 and 1)
    # - A few will be larger (up to ~3 or -3)
    # - The average of many random numbers ≈ 0
    #
    # INPUT:  shape (3, 4) — we want 3 rows and 4 columns
    # OUTPUT: a 3x4 matrix where each number is random
    #
    # Every time you run this, you get DIFFERENT numbers.
    random_weights = torch.randn(3, 4)
    print(f"Random 'weights' (3x4 matrix):\n{random_weights}")
    print(f"  Shape         : {random_weights.shape}")
    print(f"  Total params  : {random_weights.numel()}")
    # "params" = parameters = the individual numbers the model learns.
    # Our real model will have thousands of these. GPT-4 has ~1.8 TRILLION.
    print()
    print("These random numbers are like a newborn brain — no knowledge")
    print("yet. Training will adjust them until they encode useful patterns.")
    print()

    # ==================================================================
    # SECTION 5: DATA TYPES (dtype)
    # ==================================================================
    # Tensors have a "data type" that controls precision and memory.
    # The most common types in AI:
    #
    #   dtype              Size     Use case
    #   ─────              ────     ────────
    #   torch.float32      4 bytes  Default for training (good precision)
    #   torch.int64        8 bytes  Default for integers (indices, labels)
    #   torch.float16      2 bytes  Faster training on GPUs (less precise)
    #
    # We mostly use float32 (for weights/calculations) and int64 (for
    # character indices when we convert text to numbers in Step 04).

    print("=" * 50)
    print("DATA TYPES")
    print("=" * 50)

    float_tensor = torch.tensor([1.0, 2.0, 3.0])
    int_tensor = torch.tensor([1, 2, 3])

    print(f"Float tensor    : {float_tensor}  dtype={float_tensor.dtype}")
    print(f"Int tensor      : {int_tensor}    dtype={int_tensor.dtype}")
    print()
    # In Step 04, we'll convert characters to integers (int64),
    # then the model will process them as floats (float32) internally.

    # ==================================================================
    # SECTION 6: WHY THIS MATTERS
    # ==================================================================
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


# ======================================================================
# ENTRY POINT
# ======================================================================
# This is a Python convention. When you run:
#   python src/hello_pytorch.py
#
# Python sets __name__ to "__main__" for the file being executed.
# So this block ONLY runs when you execute this file directly.
#
# If another file does 'import hello_pytorch', __name__ would be
# "hello_pytorch" (not "__main__"), so main() would NOT run.
# This prevents side effects when importing.
if __name__ == "__main__":
    main()
