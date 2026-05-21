# SimpleAI — Technical Documentation

> A semi-technical guide describing every file, method, and concept in this project.
> Updated with each step so you always have a complete picture of the system.

---

## Table of Contents

- [Project Overview](#project-overview)
- [How It All Connects (The Big Picture)](#how-it-all-connects-the-big-picture)
- [File Reference](#file-reference)
  - [Configuration Files](#configuration-files)
  - [Source Code Files](#source-code-files)
  - [Data Files](#data-files)
- [Step-by-Step Flow](#step-by-step-flow)
  - [Step 01 — Project Skeleton](#step-01--project-skeleton)
  - [Step 02 — Verify PyTorch](#step-02--verify-pytorch)
  - [Step 03 — Training Data](#step-03--training-data)
- [Glossary](#glossary)

---

## Project Overview

SimpleAI builds a **character-level text generator** — a tiny neural network that reads text, learns its patterns, and generates new text one character at a time. It's the same core idea behind ChatGPT and Claude, scaled down to run on any laptop.

### The Core Idea in One Sentence

> Given a sequence of characters, predict the next character.

If you show the model thousands of examples where "t-h-" is followed by "e", it learns that "e" is a likely next character after "th". Repeat this for every pattern in the text, and the model learns to write coherent text.

### Architecture at 10,000 Feet

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SimpleAI PIPELINE                            │
│                                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐  │
│  │  Raw      │    │  Numbers │    │  Neural  │    │  Generated   │  │
│  │  Text     │───▶│  (tokens)│───▶│  Network │───▶│  Text        │  │
│  │  "The..." │    │  [7,2,1] │    │  (model) │    │  "The fu..." │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────────┘  │
│                                                                     │
│  Steps 03-04       Steps 04-06      Steps 07-11     Steps 12-14    │
│  (data prep)       (tokenize)       (train)         (generate)     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## How It All Connects (The Big Picture)

This section shows which files feed into which, and what data flows between them.

```
                    ┌─────────────────┐
                    │  data/input.txt │  ← Raw text (quotes, stories, etc.)
                    └────────┬────────┘
                             │
                             │  (read by)
                             ▼
                    ┌─────────────────┐
                    │ explore_data.py │  ← Step 03: Analyze the data
                    │                 │     Outputs: stats, vocab preview
                    └────────┬────────┘
                             │
                             │  (same file read by)
                             ▼
                    ┌─────────────────┐
                    │  vocabulary.py  │  ← Step 04: Build char↔number maps
                    │                 │     Outputs: char_to_idx, idx_to_char
                    └────────┬────────┘
                             │
                             │  (maps used by)
                             ▼
                    ┌─────────────────┐
                    │   dataset.py    │  ← Steps 05-06: Create training pairs
                    │                 │     Outputs: DataLoader with batches
                    └────────┬────────┘
                             │
                             │  (batches fed to)
                             ▼
                    ┌─────────────────┐
                    │    model.py     │  ← Step 07: The neural network itself
                    │                 │     Contains: embedding → RNN → output
                    └────────┬────────┘
                             │
                             │  (trained by)
                             ▼
                    ┌─────────────────┐
                    │    train.py     │  ← Steps 08-11: Training loop
                    │                 │     Outputs: model.pth, loss_plot.png
                    └────────┬────────┘
                             │
                             │  (loaded by)
                             ▼
                    ┌─────────────────┐
                    │   generate.py   │  ← Steps 12-13: Text generation
                    │                 │     Outputs: generated text
                    └────────┬────────┘
                             │
                             │  (wrapped by)
                             ▼
                    ┌─────────────────┐
                    │ interactive.py  │  ← Step 14: Chat-like prompt loop
                    └─────────────────┘
```

---

## File Reference

### Configuration Files

#### `requirements.txt`

| Property | Value |
|----------|-------|
| **Purpose** | Declares Python package dependencies |
| **Used by** | `pip install -r requirements.txt` |
| **Created in** | Step 01 |

**What it contains:**

| Package | Why we need it |
|---------|---------------|
| `torch` | PyTorch — the AI/ML framework. Provides tensors, neural network layers, optimizers, and automatic differentiation. It's the engine that does all the math. |
| `matplotlib` | Charting library. We use it to plot the training loss curve — a visual way to see if the model is learning. |

**How `pip install` works:**
```
pip install -r requirements.txt
        │
        ├── Reads each line of the file
        ├── Downloads the package from PyPI (Python Package Index)
        ├── Installs it into your virtual environment
        └── Also installs any sub-dependencies automatically
```

---

#### `.gitignore`

| Property | Value |
|----------|-------|
| **Purpose** | Tells git which files/folders to ignore |
| **Created in** | Step 01 |

**What it ignores and why:**

| Pattern | What it matches | Why ignore it |
|---------|----------------|---------------|
| `__pycache__/` | Compiled Python bytecode folders | Machine-specific, auto-generated when you run `.py` files |
| `*.pyc` | Individual compiled Python files | Same reason — auto-generated cache files |
| `venv/`, `.venv/` | Virtual environment folders | Can be 100s of MBs, each developer creates their own |
| `outputs/model.pth` | Saved model weights | Large binary file, can be recreated by running `train.py` |
| `.DS_Store` | macOS Finder metadata | OS-specific, not relevant to the project |

---

#### `README.md`

| Property | Value |
|----------|-------|
| **Purpose** | Project overview and quick-start guide |
| **Created in** | Step 01 |

The front door of the project. Contains a one-paragraph description, the folder structure, and three commands to get started (`venv`, `activate`, `pip install`).

---

### Source Code Files

---

#### `src/__init__.py`

| Property | Value |
|----------|-------|
| **Purpose** | Makes `src/` a Python package |
| **Content** | Empty file (0 bytes) |
| **Created in** | Step 01 |

**Why it exists:**
In Python, a folder with an `__init__.py` file becomes a "package" — meaning other files can import from it using `from src.module import something`. Without this file, Python wouldn't recognize `src/` as a place to look for imports.

Even though it's empty, its mere existence has meaning. It's like a sign on a door that says "this room has things you can use."

---

#### `src/hello_pytorch.py`

| Property | Value |
|----------|-------|
| **Purpose** | Verify PyTorch installation and introduce tensors |
| **Created in** | Step 02 |
| **Run with** | `python src/hello_pytorch.py` |
| **Input** | None |
| **Output** | Printed text — tensor examples, math results, random weights |

**Methods:**

| Method | Description |
|--------|-------------|
| `main()` | Runs all 6 demonstration sections sequentially |

**Detailed Flow:**

```
main()
  │
  ├── Section 1: Installation Check
  │   ├── Print torch.__version__  →  e.g., "2.12.0"
  │   └── Print torch.cuda.is_available()  →  True/False
  │
  ├── Section 2: Tensor Basics
  │   ├── Create scalar tensor(42)           →  shape: ()
  │   │   INPUT:  Python int 42
  │   │   OUTPUT: tensor(42)
  │   │
  │   ├── Create vector tensor([1,2,3,4,5])  →  shape: (5,)
  │   │   INPUT:  Python list [1.0, 2.0, 3.0, 4.0, 5.0]
  │   │   OUTPUT: tensor([1., 2., 3., 4., 5.])
  │   │
  │   └── Create matrix tensor([[1,2,3],[4,5,6]])  →  shape: (2,3)
  │       INPUT:  Nested Python list
  │       OUTPUT: tensor([[1, 2, 3],
  │                       [4, 5, 6]])
  │
  ├── Section 3: Tensor Math
  │   ├── a + b   →  [10+1, 20+2, 30+3] = [11, 22, 33]
  │   ├── a * b   →  [10*1, 20*2, 30*3] = [10, 40, 90]
  │   ├── a.mean()  →  (10+20+30) / 3   = 20.0
  │   └── a.sum()   →  10+20+30         = 60.0
  │
  ├── Section 4: Random Tensors
  │   └── torch.randn(3, 4)  →  3×4 matrix of random floats
  │       INPUT:  shape dimensions (3, 4)
  │       OUTPUT: tensor([[ 0.31, -2.13,  0.60, -0.35],
  │                       [-1.94, -0.58, -1.67,  0.09],
  │                       [-1.01,  0.07, -0.81, -1.89]])
  │       (different every run — these are random!)
  │
  ├── Section 5: Data Types
  │   ├── Float tensor  →  dtype=torch.float32  (for weights)
  │   └── Int tensor    →  dtype=torch.int64    (for indices)
  │
  └── Section 6: Summary
      └── Print explanation of how tensors connect to AI
```

**Key concepts introduced:**

| Concept | Explanation | Example |
|---------|-------------|---------|
| **Tensor** | A container for numbers — can be 0D (scalar), 1D (vector), 2D (matrix), or higher | `torch.tensor([1, 2, 3])` → 1D tensor |
| **Shape** | The dimensions of a tensor — tells you how the numbers are organized | `(2, 3)` = 2 rows, 3 columns |
| **Vectorized ops** | Math on entire tensors at once, without Python loops — much faster | `a + b` adds each pair of elements |
| **Random init** | Models start as random numbers; training adjusts them into useful patterns | `torch.randn(3, 4)` → random 3×4 matrix |
| **dtype** | Data type of a tensor — float32 for decimals, int64 for whole numbers | `tensor([1.0]).dtype` → `torch.float32` |

---

#### `src/explore_data.py`

| Property | Value |
|----------|-------|
| **Purpose** | Analyze the training dataset before feeding it to the model |
| **Created in** | Step 03 |
| **Run with** | `python src/explore_data.py` |
| **Input** | Reads `data/input.txt` |
| **Output** | Printed analysis — character counts, vocabulary, frequency chart |

**Methods:**

| Method | Description |
|--------|-------------|
| `main()` | Loads the text file and runs all 6 analysis steps |

**Detailed Flow:**

```
main()
  │
  ├── Step 1: Load text file
  │   ├── open("data/input.txt", "r", encoding="utf-8")
  │   └── text = f.read()  →  one big string with entire file contents
  │
  │   INPUT:  file path "data/input.txt"
  │   OUTPUT: text = "The only way to do great work is to love...\n..."
  │
  ├── Step 2: Basic statistics
  │   ├── total_chars = len(text)             →  6201
  │   ├── lines = text.strip().split("\n")    →  list of 99 lines
  │   ├── total_lines = len(lines)            →  99
  │   └── total_words = len(text.split())     →  1200
  │
  │   INPUT:  text string
  │   OUTPUT: character count, word count, line count
  │
  ├── Step 3: Character vocabulary
  │   ├── unique_chars = sorted(set(text))    →  ['\\n', ' ', ',', '.', ...]
  │   └── vocab_size = len(unique_chars)      →  48
  │
  │   INPUT:  text string
  │   OUTPUT: sorted list of 48 unique characters
  │
  │   HOW set() WORKS:
  │     "hello" → set("hello") → {'h', 'e', 'l', 'o'}  (removes duplicates)
  │     sorted({'h', 'e', 'l', 'o'}) → ['e', 'h', 'l', 'o']  (alphabetical)
  │
  ├── Step 4: Preview data
  │   ├── Print lines[:5]   →  first 5 quotes
  │   └── Print lines[-3:]  →  last 3 quotes
  │
  │   PURPOSE: Visual sanity check — spot encoding issues, weird chars, etc.
  │
  ├── Step 5: Character frequency
  │   ├── Count occurrences of each character using a dictionary
  │   ├── Sort by count (descending)
  │   └── Print top 15 with percentage and bar chart
  │
  │   INPUT:  text string
  │   OUTPUT: sorted list of (character, count) pairs
  │
  │   EXAMPLE:
  │     ' ' (space) : 1101 (17.8%) ###################################
  │     'e'         :  654 (10.5%) #####################
  │     't'         :  444 ( 7.2%) ##############
  │
  │   HOW THE COUNTING WORKS:
  │     char_counts = {}
  │     for char in "hello":
  │         char_counts[char] = char_counts.get(char, 0) + 1
  │     Result: {'h': 1, 'e': 1, 'l': 2, 'o': 1}
  │
  └── Step 6: Summary
      └── Print what the stats mean for model training
```

**Key concepts introduced:**

| Concept | Explanation | Example |
|---------|-------------|---------|
| **Dataset** | The text the model will learn from — just a plain text file | `data/input.txt` with 99 quotes |
| **Vocabulary** | The set of all unique characters in the data — the model's "alphabet" | 48 chars: letters, space, comma, period, etc. |
| **Character frequency** | How often each character appears — common chars are easier to learn | Space: 17.8%, 'e': 10.5%, 't': 7.2% |
| **Data inspection** | Always look at your data before training — "garbage in, garbage out" | Check for weird chars, encoding issues |

---

### Data Files

#### `data/input.txt`

| Property | Value |
|----------|-------|
| **Purpose** | Training dataset — the text the model learns from |
| **Created in** | Step 03 |
| **Size** | 6,201 characters, 1,200 words, 99 lines |
| **Content** | Inspirational and philosophical quotes |
| **Vocabulary** | 48 unique characters |

**Character breakdown:**

| Category | Characters | Count |
|----------|-----------|-------|
| Lowercase letters | a-z | 26 |
| Uppercase letters | A, B, C, D, E, G, H, I, K, L, M, N, R, S, T, W, Y | 17 |
| Punctuation | `. , :` | 3 |
| Whitespace | space, newline | 2 |
| **Total** | | **48** |

**Why quotes?**
- Short, self-contained sentences (easy for a small model)
- Rich vocabulary but consistent style
- Repetitive patterns ("The", "is", "to", "you") give the model clear signals
- Small enough to train in minutes, large enough to see patterns

---

## Step-by-Step Flow

### Step 01 — Project Skeleton

**What was built:** The folder structure and configuration files.

**Why it matters:** Every software project needs an organized structure. This step establishes where code lives (`src/`), where data goes (`data/`), where outputs are saved (`outputs/`), and what dependencies are needed (`requirements.txt`).

```
What changed:
  + .gitignore           ← tells git what to ignore
  + README.md            ← project overview
  + requirements.txt     ← Python dependencies (torch, matplotlib)
  + src/__init__.py      ← makes src/ a Python package
  + data/.gitkeep        ← placeholder so git tracks empty folder
  + outputs/.gitkeep     ← placeholder so git tracks empty folder
```

**Nothing runs yet** — this is purely organizational.

---

### Step 02 — Verify PyTorch

**What was built:** A script that tests PyTorch and teaches you about tensors.

**Why it matters:** Before building anything complex, verify your tools work. This script also introduces the single most important concept in AI: **tensors** (containers for numbers that all AI math operates on).

```
What changed:
  + src/hello_pytorch.py   ← tensor demo script

Run:
  python src/hello_pytorch.py

Expected output:
  PyTorch version, tensor examples, math operations, random weights
```

**Key takeaway:** Everything in AI is just math on tensors.

---

### Step 03 — Training Data

**What was built:** The training dataset and a script to analyze it.

**Why it matters:** AI models learn from data. Before training, you need to understand what you're feeding the model — how much text, what characters, how they're distributed. This is the most important habit in ML engineering.

```
What changed:
  + data/input.txt        ← 99 quotes (training data)
  + src/explore_data.py   ← data analysis script

Run:
  python src/explore_data.py

Expected output:
  6201 chars, 48 unique characters, frequency chart, data preview
```

**Key takeaway:** The model's vocabulary is defined by the data. 48 unique characters = the model's entire "alphabet".

**The flow so far:**

```
data/input.txt ──(read by)──▶ explore_data.py ──(prints)──▶ statistics
                                                              vocab size
                                                              frequency chart
```

---

## Glossary

Terms are listed in the order you'll encounter them, not alphabetically.

| Term | Simple Definition | First Seen |
|------|-------------------|------------|
| **Tensor** | A container for numbers — can be a single number, a list, a grid, or higher dimensions. All AI data lives in tensors. | Step 02 |
| **Shape** | The dimensions of a tensor. `(2, 3)` means 2 rows, 3 columns. `(5,)` means a list of 5 numbers. | Step 02 |
| **Scalar** | A single number. A 0-dimensional tensor. | Step 02 |
| **Vector** | A list of numbers. A 1-dimensional tensor. | Step 02 |
| **Matrix** | A grid (table) of numbers. A 2-dimensional tensor. | Step 02 |
| **CUDA** | NVIDIA's technology for running code on GPUs. Not needed for this project. | Step 02 |
| **Vectorized operation** | Math on entire tensors at once (no loops). Much faster than element-by-element. | Step 02 |
| **dtype** | The data type of a tensor — float32 for decimals, int64 for whole numbers. | Step 02 |
| **Dataset** | The collection of text (or other data) that a model learns from. | Step 03 |
| **Vocabulary** | The set of all unique tokens (characters, in our case) that the model knows about. | Step 03 |
| **Character frequency** | How often each character appears in the data. Affects how well the model learns each character. | Step 03 |
| **Tokenization** | Converting text into numbers. We'll map each character to a unique integer. | Step 04 (upcoming) |
| **Embedding** | Converting a token number into a rich vector of floats that captures meaning. | Step 07 (upcoming) |
| **Loss** | A number that measures how wrong the model's predictions are. Lower = better. | Step 08 (upcoming) |
| **Epoch** | One complete pass through the entire training dataset. | Step 09 (upcoming) |
| **Batch** | A small group of training examples processed together. More efficient than one at a time. | Step 06 (upcoming) |
| **Gradient** | The direction and amount to adjust each weight to reduce the loss. | Step 08 (upcoming) |
| **Optimizer** | The algorithm that uses gradients to update the model's weights. | Step 08 (upcoming) |
| **Inference** | Using a trained model to produce output (generate text). No learning happens. | Step 12 (upcoming) |
| **Temperature** | Controls randomness in generation. Low = predictable, high = creative. | Step 13 (upcoming) |
| **Overfitting** | When a model memorizes training data instead of learning general patterns. | Step 15 (upcoming) |

---

> *This document is updated with each new step. Last updated: Step 03.*
