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
  - [Step 04 — Character Vocabulary](#step-04--character-vocabulary)
  - [Step 05 — Training Sequences](#step-05--training-sequences)
  - [Step 06 — DataLoader Batching](#step-06--dataloader-batching)
  - [Step 07 — Neural Network Model](#step-07--neural-network-model)
  - [Step 08 — Loss Function and Optimizer](#step-08--loss-function-and-optimizer)
  - [Step 09 — Single Epoch Training](#step-09--single-epoch-training)
  - [Step 10 — Full Training and Model Saving](#step-10--full-training-and-model-saving)
  - [Step 11 — Loss Curve Visualization](#step-11--loss-curve-visualization)
  - [Step 12 — Text Generation](#step-12--text-generation)
  - [Step 13 — Temperature Control](#step-13--temperature-control)
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

#### `src/vocabulary.py`

| Property | Value |
|----------|-------|
| **Purpose** | Convert text to numbers and back (tokenization) |
| **Created in** | Step 04 |
| **Run with** | `python src/vocabulary.py` |
| **Input** | Reads `data/input.txt` to build vocabulary |
| **Output** | Printed demos — encoding, decoding, vocabulary table, round-trip proof |
| **Imported by** | `dataset.py` (Step 05), `train.py` (Step 09), `generate.py` (Step 12) |

**Classes:**

| Class | Description |
|-------|-------------|
| `Vocabulary` | Builds and stores two-way character↔number mappings from training text |

**Vocabulary class — Methods:**

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `__init__(text)` | `text` (str): full training text | None (sets up instance) | Builds char↔number mappings from all unique characters in the text |
| `encode(text)` | `text` (str): any text to encode | `list[int]`: number per character | Converts text → numbers using `char_to_idx` dictionary |
| `decode(indices)` | `indices` (list[int]): numbers to decode | `str`: the decoded text | Converts numbers → text using `idx_to_char` dictionary |

**Vocabulary class — Attributes:**

| Attribute | Type | Description | Example value |
|-----------|------|-------------|---------------|
| `chars` | `list[str]` | Sorted list of all unique characters | `['\n', ' ', ',', '.', ':', 'A', ...]` |
| `vocab_size` | `int` | Total number of unique characters | `48` |
| `char_to_idx` | `dict[str, int]` | Character → number mapping | `{'a': 22, 'b': 23, ...}` |
| `idx_to_char` | `dict[int, str]` | Number → character mapping | `{22: 'a', 23: 'b', ...}` |

**Standalone functions:**

| Function | Parameters | Description |
|----------|-----------|-------------|
| `demonstrate_encoding(vocab)` | `vocab` (Vocabulary) | Prints step-by-step encoding examples with character-by-character breakdown |
| `demonstrate_decoding(vocab)` | `vocab` (Vocabulary) | Prints step-by-step decoding examples and a full round-trip proof |
| `demonstrate_vocabulary_details(vocab)` | `vocab` (Vocabulary) | Prints the complete 48-character mapping table and observations |
| `main()` | None | Loads training text, builds Vocabulary, runs all three demos |

**Detailed Flow:**

```
main()
  │
  ├── Load data/input.txt → text string (6201 chars)
  │
  ├── Build Vocabulary(text)
  │   │
  │   ├── set(text) → find 48 unique characters
  │   │   INPUT:  "The only way..." (6201 chars)
  │   │   OUTPUT: {'T', 'h', 'e', ' ', 'o', 'n', ...} (48 unique)
  │   │
  │   ├── sorted() → alphabetical order
  │   │   OUTPUT: ['\n', ' ', ',', '.', ':', 'A', 'B', ...]
  │   │
  │   ├── Build char_to_idx
  │   │   OUTPUT: {'\n': 0, ' ': 1, ',': 2, '.': 3, ..., 'z': 47}
  │   │
  │   └── Build idx_to_char
  │       OUTPUT: {0: '\n', 1: ' ', 2: ',', 3: '.', ..., 47: 'z'}
  │
  ├── demonstrate_encoding(vocab)
  │   │
  │   ├── encode("The")
  │   │   'T' → char_to_idx['T'] → 19
  │   │   'h' → char_to_idx['h'] → 29
  │   │   'e' → char_to_idx['e'] → 26
  │   │   OUTPUT: [19, 29, 26]
  │   │
  │   ├── encode("Life is short.")
  │   │   OUTPUT: [14, 30, 27, 26, 1, 30, 40, 1, 40, 29, 36, 39, 41, 3]
  │   │   (spaces → 1, period → 3, each letter → its number)
  │   │
  │   └── Determinism proof: encode("the") twice → same result both times
  │
  ├── demonstrate_decoding(vocab)
  │   │
  │   ├── decode([19, 29, 26])
  │   │   19 → idx_to_char[19] → 'T'
  │   │   29 → idx_to_char[29] → 'h'
  │   │   26 → idx_to_char[26] → 'e'
  │   │   OUTPUT: "The"
  │   │
  │   └── Round-trip proof:
  │       "Dream big." → encode → [8,39,26,22,34,1,23,30,28,3] → decode → "Dream big."
  │       Original == Decoded? True ← no information lost
  │
  └── demonstrate_vocabulary_details(vocab)
      ├── Print all 48 mappings in table format
      └── Key insight: model output layer will have 48 neurons
```

**Key concepts introduced:**

| Concept | Explanation | Example |
|---------|-------------|---------|
| **Tokenization** | Converting text into numbers so a neural network can process it | `"The"` → `[19, 29, 26]` |
| **Encoding** | Text → numbers direction. Used before feeding text to the model | `vocab.encode("hi")` → `[29, 30]` |
| **Decoding** | Numbers → text direction. Used after model produces output | `vocab.decode([29, 30])` → `"hi"` |
| **char_to_idx** | Dictionary mapping each character to its unique number | `{'a': 22, 'b': 23, 'e': 26, ...}` |
| **idx_to_char** | Reverse dictionary mapping each number back to its character | `{22: 'a', 23: 'b', 26: 'e', ...}` |
| **Deterministic** | Same input always produces same output — crucial for reproducibility | `encode("the")` always returns `[41, 29, 26]` |

---

#### `src/dataset.py`

| Property | Value |
|----------|-------|
| **Purpose** | Turn text into training pairs (input, target) the model can learn from |
| **Created in** | Step 05 |
| **Run with** | `PYTHONPATH=src python src/dataset.py` |
| **Input** | Reads `data/input.txt`, uses `Vocabulary` from `vocabulary.py` |
| **Output** | Printed training pair examples, shape info, shift verification |
| **Imported by** | `train.py` (Step 09) |

**Classes:**

| Class | Parent | Description |
|-------|--------|-------------|
| `TextDataset` | `torch.utils.data.Dataset` | Creates (input, target) pairs from encoded text |

**TextDataset class — Methods:**

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `__init__(text, vocab, seq_length)` | `text` (str), `vocab` (Vocabulary), `seq_length` (int, default=50) | None | Encodes full text into tensor, calculates number of examples |
| `__len__()` | None | `int` | Returns total number of training examples |
| `__getitem__(idx)` | `idx` (int): example index | `tuple[Tensor, Tensor]` | Returns (input, target) pair, each of shape `(seq_length,)` |

**TextDataset class — Attributes:**

| Attribute | Type | Description | Example value |
|-----------|------|-------------|---------------|
| `vocab` | `Vocabulary` | The vocabulary used for encoding/decoding | (from Step 04) |
| `seq_length` | `int` | Number of characters per training example | `50` |
| `data` | `Tensor` | The entire text encoded as a 1D tensor of integers | `tensor([19, 29, 26, 1, ...])` shape `(6201,)` |
| `num_examples` | `int` | How many complete training pairs fit | `124` |

**Standalone functions:**

| Function | Parameters | Description |
|----------|-----------|-------------|
| `create_dataloader(dataset, batch_size, shuffle)` | dataset (TextDataset), batch_size (int, default=16), shuffle (bool, default=True) | Wraps dataset in a PyTorch DataLoader for batched, shuffled iteration. Returns `DataLoader`. |
| `demonstrate_training_pairs(dataset, vocab, num_examples)` | dataset, vocab, count | Shows training pairs as both text and numbers with position-by-position breakdown |
| `demonstrate_data_shapes(dataset)` | dataset | Shows tensor shapes, dtypes, and verifies the input→target shift |
| `demonstrate_batching(dataloader, vocab, dataset)` | dataloader, vocab, dataset | Shows batch shapes, shape transformation, and how to index into batches |
| `demonstrate_shuffling(dataset, vocab)` | dataset, vocab | Compares shuffled vs unshuffled DataLoaders to show randomization |
| `demonstrate_full_epoch(dataloader)` | dataloader | Iterates through ALL batches, shows what one epoch looks like |
| `main()` | None | Loads text, builds vocab, creates dataset + dataloader, runs all demos |

**Detailed Flow:**

```
main()
  │
  ├── Load data/input.txt → text (6201 chars)
  │
  ├── Build Vocabulary(text) → vocab (48 chars, from Step 04)
  │
  ├── Create TextDataset(text, vocab, seq_length=50)
  │   │
  │   ├── vocab.encode(text) → [19, 29, 26, 1, 36, ...] (6201 ints)
  │   ├── torch.tensor(encoded) → tensor of shape (6201,)
  │   └── num_examples = (6201 - 1) ÷ 50 = 124
  │
  ├── demonstrate_training_pairs()
  │   │
  │   ├── dataset[0]:
  │   │   input  = data[0:50]   "The only way to do great work is to love what you "
  │   │   target = data[1:51]   "he only way to do great work is to love what you d"
  │   │
  │   │   Position 0: input='T' → target='h'  (after T, predict h)
  │   │   Position 1: input='h' → target='e'  (after h, predict e)
  │   │   Position 2: input='e' → target=' '  (after e, predict space)
  │   │   ...
  │   │
  │   ├── dataset[1]:
  │   │   input  = data[50:100]  "do.\nIn the middle of difficulty..."
  │   │   target = data[51:101]  "o.\nIn the middle of difficulty l..."
  │   │
  │   └── dataset[2]:
  │       input  = data[100:150] "Life is what happens when you..."
  │       target = data[101:151] "ife is what happens when you ..."
  │
  ├── demonstrate_data_shapes()
  │   │
  │   ├── Full data:   shape (6201,), dtype int64
  │   ├── One input:   shape (50,),   dtype int64
  │   ├── One target:  shape (50,),   dtype int64
  │   └── Shift proof: target[0]=data[1], target[1]=data[2], ...
  │
  └── Preview Step 06: batching with DataLoader
```

**The input→target shift visualized:**

```
data:    [19, 29, 26,  1, 36, 35, 33, 46, ...]
          T    h   e   _   o    n   l    y

input:   [19, 29, 26,  1, 36]     ← positions 0-4
target:  [29, 26,  1, 36, 35]     ← positions 1-5 (shifted by 1)
          ↑
          "After T(19), the next char should be h(29)"
```

**Key concepts introduced:**

| Concept | Explanation | Example |
|---------|-------------|---------|
| **Training pair** | An (input, target) pair. The model sees the input and tries to predict the target. | input="The o", target="he on" |
| **Sequence length** | How many characters per training example. Controls context window size. | 50 chars (real LLMs: 4096+) |
| **Input→target shift** | Target is the input shifted right by 1 character. Each position: predict the next char. | input[0]='T' → target[0]='h' |
| **PyTorch Dataset** | A class with `__len__` and `__getitem__` methods. Standard way to serve training data. | `dataset[0]` returns one (input, target) pair |

---

#### `src/model.py`

| Property | Value |
|----------|-------|
| **Purpose** | Define the neural network that learns to predict the next character |
| **Created in** | Step 07 |
| **Run with** | `PYTHONPATH=src python src/model.py` |
| **Input** | Batched character indices, shape `(batch_size, seq_length)` |
| **Output** | Prediction scores, shape `(batch_size, seq_length, vocab_size)` |
| **Imported by** | `train.py` (Step 09), `generate.py` (Step 12) |

**Classes:**

| Class | Parent | Description |
|-------|--------|-------------|
| `TinyLanguageModel` | `nn.Module` | Three-layer neural network: Embedding → RNN → Linear output |

**TinyLanguageModel — Constructor parameters (hyperparameters):**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vocab_size` | `int` | (required) | Number of unique characters. Determines embedding table rows and output layer size. |
| `embed_size` | `int` | `128` | Dimensions per character embedding vector. Larger = more expressive. |
| `hidden_size` | `int` | `256` | RNN hidden state size. The model's "memory capacity". |
| `num_layers` | `int` | `2` | Stacked RNN layers. Layer 1 learns simple patterns, layer 2 learns higher ones. |

**TinyLanguageModel — Layers (created in `__init__`):**

| Layer | PyTorch class | Shape | What it does |
|-------|--------------|-------|-------------|
| `self.embedding` | `nn.Embedding(48, 128)` | Lookup table `(48, 128)` | Converts character index → 128-number vector |
| `self.rnn` | `nn.RNN(128, 256, 2)` | Multiple weight matrices | Processes sequence left-to-right, builds context |
| `self.output_layer` | `nn.Linear(256, 48)` | Weight `(48, 256)` + bias `(48,)` | Converts context → 48 prediction scores |

**TinyLanguageModel — Methods:**

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `forward(x, hidden=None)` | `x`: char indices `(batch, seq_len)`, `hidden`: optional RNN state | `(logits, hidden)`: scores `(batch, seq_len, 48)` + RNN state | Full forward pass through all 3 layers |

**Standalone functions:**

| Function | Parameters | Description |
|----------|-----------|-------------|
| `demonstrate_model_creation(vocab_size)` | `int` | Creates model, prints architecture, counts parameters per layer |
| `demonstrate_forward_pass(model, vocab_size)` | model, int | Passes fake data through model, shows shapes at each stage |
| `main()` | None | Creates model with vocab_size=48, runs all demos |

**Parameter count: 248,880 total**

| Component | Parameters | Percentage |
|-----------|-----------|-----------|
| Embedding (48 × 128) | 6,144 | 2.5% |
| RNN layer 0 | 98,816 | 39.7% |
| RNN layer 1 | 131,584 | 52.8% |
| Output layer (256 × 48 + 48) | 12,336 | 5.0% |
| **Total** | **248,880** | **100%** |

**Detailed Flow (forward pass):**

```
forward(x, hidden=None)
  │
  │   INPUT: x shape (batch_size, seq_length) = (16, 50)
  │          Each value is an int 0-47 (character index)
  │
  ├── Step 1: self.embedding(x)
  │   │  Looks up each int in the embedding table
  │   │  19 → row 19 of table → [0.3, -0.7, ...(128 floats)]
  │   │
  │   │  INPUT:  (16, 50)        ← ints
  │   └─ OUTPUT: (16, 50, 128)   ← float vectors
  │
  ├── Step 2: self.rnn(embedded, hidden)
  │   │  Processes sequence left-to-right
  │   │  Position 0: context = f(embed[0], initial_hidden)
  │   │  Position 1: context = f(embed[1], hidden_from_pos_0)
  │   │  Position 2: context = f(embed[2], hidden_from_pos_1)
  │   │  ...each position knows about ALL previous positions
  │   │
  │   │  INPUT:  (16, 50, 128)   ← embedded vectors
  │   └─ OUTPUT: (16, 50, 256)   ← context vectors
  │              + hidden (2, 16, 256) ← final hidden state
  │
  ├── Step 3: self.output_layer(rnn_out)
  │   │  Matrix multiply: context × weights + bias
  │   │  Produces 48 scores per position
  │   │
  │   │  INPUT:  (16, 50, 256)   ← context
  │   └─ OUTPUT: (16, 50, 48)    ← logits (prediction scores)
  │
  └── RETURN: (logits, hidden)
```

**Key concepts introduced:**

| Concept | Explanation | Example |
|---------|-------------|---------|
| **Neural network** | A series of math layers that transform input into predictions. Weights are learned. | Our 3-layer model with 248,880 parameters |
| **Embedding** | Converts a plain number into a rich vector of floats that captures meaning | `19` ('T') → `[0.3, -0.7, 0.1, ...]` (128 numbers) |
| **RNN** | Recurrent Neural Network — processes sequences position by position, building up context | After seeing "The", hidden state encodes "I've seen 'The'" |
| **Hidden state** | The RNN's "working memory" — a vector that carries context between positions | Shape: `(2, 16, 256)` — 2 layers, 16 examples, 256 numbers each |
| **Logits** | Raw prediction scores before converting to probabilities. Highest score = best guess. | `[..., 8.5, ..., 2.1, ...]` — 'e' scored highest |
| **Forward pass** | The computation path from input to output through all layers | Input (16,50) → Embed → RNN → Linear → Output (16,50,48) |
| **Parameters/weights** | The learnable numbers inside the model. Training adjusts these. | 248,880 numbers that start random and become meaningful |
| **Hyperparameters** | Settings YOU choose (embed_size, hidden_size). NOT learned by training. | embed_size=128, hidden_size=256, num_layers=2 |

---

#### `src/training_setup.py`

| Property | Value |
|----------|-------|
| **Purpose** | Demonstrate the loss function and optimizer — how the model learns from mistakes |
| **Created in** | Step 08 |
| **Run with** | `PYTHONPATH=src python src/training_setup.py` |
| **Input** | Creates model internally, uses fake data for demos |
| **Output** | Loss computation demo, optimizer explanation, one full learning step with before/after |

**Standalone functions:**

| Function | Parameters | Description |
|----------|-----------|-------------|
| `demonstrate_loss_function(model, vocab_size)` | model, int | Shows CrossEntropyLoss: forward pass → reshape → loss value → interpretation vs random baseline |
| `demonstrate_optimizer(model)` | model | Creates Adam optimizer, explains learning rate and what Adam does per step |
| `demonstrate_one_learning_step(model, loss_fn, optimizer, vocab_size)` | all components | Runs the complete 5-step cycle: forward → loss → backward → step → zero_grad. Shows weights before/after. |
| `main()` | None | Creates model, runs all three demos |

**The 5-line training cycle (the core of ALL neural network training):**

```python
logits, _ = model(input)                    # 1. Forward pass
loss = loss_fn(logits.view(-1, 48), target.view(-1))  # 2. Compute loss
loss.backward()                             # 3. Compute gradients
optimizer.step()                            # 4. Update weights
optimizer.zero_grad()                       # 5. Clear gradients
```

**Key components:**

| Component | PyTorch class | What it does |
|-----------|--------------|-------------|
| Loss function | `nn.CrossEntropyLoss()` | Measures how wrong predictions are. Returns single number (lower = better). Random baseline ≈ 3.87 for 48 chars. |
| Optimizer | `torch.optim.Adam(lr=0.001)` | Reads gradients, adjusts all 248,880 weights to reduce loss. Adaptive learning rate per parameter. |

**Loss scale reference:**

| Loss value | What it means |
|-----------|--------------|
| ~3.87 | Random guessing (untrained model). Expected: -log(1/48). |
| ~2.0 | Model has learned common patterns |
| ~1.0 | Model has learned well |
| → 0.0 | Memorized training data (overfitting) |

---

#### `src/train.py`

| Property | Value |
|----------|-------|
| **Purpose** | Train the model for multiple epochs and save the trained weights to disk |
| **Created in** | Step 09, extended in Step 10 |
| **Run with** | `PYTHONPATH=src python src/train.py` |
| **Input** | Reads `data/input.txt`, imports `Vocabulary`, `TextDataset`, `create_dataloader`, `TinyLanguageModel` |
| **Output** | `outputs/model.pth` (trained weights), `outputs/vocab.pth` (vocabulary), `outputs/loss_history.pth` (loss per epoch) |
| **Imports** | `vocabulary.py`, `dataset.py`, `model.py` |

**Functions:**

| Function | Parameters | Returns | Description |
|----------|-----------|---------|-------------|
| `train_one_epoch(model, dataloader, loss_fn, optimizer, vocab_size, epoch_num)` | model, dataloader, loss_fn, optimizer, vocab_size (int), epoch_num (int, default=0) | `float` (average loss) | Runs the 5-step training cycle on every batch. Returns average loss for the epoch. |
| `train(model, dataloader, loss_fn, optimizer, vocab_size, num_epochs, print_every)` | model, dataloader, loss_fn, optimizer, vocab_size (int), num_epochs (int, default=50), print_every (int, default=5) | `list[float]` (loss history) | Calls train_one_epoch() repeatedly, collecting average loss per epoch into a list. Prints progress at intervals. |
| `save_model(model, filepath)` | model (TinyLanguageModel), filepath (str, default="outputs/model.pth") | None | Saves `model.state_dict()` to disk using `torch.save()`. Reports file size. |
| `save_vocabulary(vocab, filepath)` | vocab (Vocabulary), filepath (str, default="outputs/vocab.pth") | None | Saves chars, char_to_idx, idx_to_char as a dictionary to disk. |
| `main()` | None | None | Full pipeline: load → build → create → train 100 epochs → save model + vocab + loss history. |

**Training configuration:**

| Setting | Value | Why |
|---------|-------|-----|
| `num_epochs` | 100 | Enough passes for our small dataset to converge |
| `batch_size` | 16 | 7 batches per epoch, 700 total weight updates |
| `learning_rate` | 0.003 | Slightly faster than default 0.001 for small model/data |
| `seq_length` | 50 | Characters per training example |

**Detailed Flow:**

```
main()
  │
  ├── Load data/input.txt → text (6201 chars)
  ├── Build Vocabulary(text) → vocab (48 chars)
  ├── Create TextDataset(text, vocab, seq_length=50) → 124 examples
  ├── Create DataLoader(dataset, batch_size=16, shuffle=True) → 7 batches
  ├── Create TinyLanguageModel(vocab_size=48) → 248,880 parameters
  ├── Create CrossEntropyLoss() + Adam(lr=0.003)
  │
  └── train(model, dataloader, loss_fn, optimizer, 48, num_epochs=100)
      │
      ├── Epoch 0:  train_one_epoch() → avg_loss ≈ 3.00
      ├── Epoch 10: train_one_epoch() → avg_loss ≈ 1.28
      ├── Epoch 20: train_one_epoch() → avg_loss ≈ 0.59
      ├── Epoch 30: train_one_epoch() → avg_loss ≈ 0.20
      ├── ...
      ├── Epoch 90: train_one_epoch() → avg_loss ≈ 0.05
      └── Epoch 99: train_one_epoch() → avg_loss ≈ 0.04
          │
          └── RETURN [3.00, ..., 0.04]  (100 loss values)
                │
                ├── save_model(model, "outputs/model.pth")
                │     └── torch.save(model.state_dict(), ...) → 976 KB
                │
                ├── save_vocabulary(vocab, "outputs/vocab.pth")
                │     └── torch.save({chars, char_to_idx, idx_to_char}) → 2 KB
                │
                └── torch.save(loss_history, "outputs/loss_history.pth")
```

**What gets saved to disk:**

```
outputs/
  ├── model.pth          ← 248,880 trained weights (~976 KB)
  │                        Loaded by: generate.py (Step 12)
  │
  ├── vocab.pth          ← {chars, char_to_idx, idx_to_char} (~2 KB)
  │                        Loaded by: generate.py (Step 12)
  │                        Needed to decode model output → text
  │
  └── loss_history.pth   ← [3.00, 2.71, ..., 0.04] (100 floats)
                           Loaded by: plot_loss.py (Step 11)
                           Used to draw the training curve
```

**Key concepts introduced:**

| Concept | Explanation | Example |
|---------|-------------|---------|
| **Multi-epoch training** | Running many passes over the same data. Each pass refines the model further. | 100 epochs = 700 weight updates, loss 3.00 → 0.04 |
| **Loss history** | A list of average loss values, one per epoch. Shows whether training is working. | `[3.00, 2.71, 2.31, ..., 0.04]` |
| **state_dict()** | A dictionary of ALL learnable parameters in the model. The standard way to save/load models. | `{'embedding.weight': tensor(...), 'rnn.weight_ih_l0': ...}` |
| **torch.save()** | Serializes tensors/dicts to a file using Python's pickle. Convention: `.pth` extension. | `torch.save(model.state_dict(), "model.pth")` |
| **Convergence** | When loss stops decreasing meaningfully — the model has learned what it can from the data. | Loss plateaus around 0.04 after ~80 epochs |
| **Serialization** | Converting in-memory objects (tensors) to bytes that can be saved to disk and loaded later. | Model in RAM → `torch.save()` → file on disk |

---

#### `src/plot_loss.py`

| Property | Value |
|----------|-------|
| **Purpose** | Visualize the training loss curve as a PNG chart |
| **Created in** | Step 11 |
| **Run with** | `PYTHONPATH=src python src/plot_loss.py` |
| **Input** | `outputs/loss_history.pth` (list of 100 floats from Step 10) |
| **Output** | `outputs/loss_plot.png` (chart image, ~59 KB) |
| **Imports** | `torch`, `matplotlib` |

**Functions:**

| Function | Parameters | Returns | Description |
|----------|-----------|---------|-------------|
| `plot_training_loss(loss_history, save_path)` | loss_history (list[float]), save_path (str, default="outputs/loss_plot.png") | None | Creates a matplotlib line chart with loss curve, random baseline reference line, start/end annotations, and saves as PNG at 150 DPI. |
| `print_loss_summary(loss_history)` | loss_history (list[float]) | None | Prints text-based summary: first/final/best loss, milestone values at 25%/50%/75%, improvement percentage. |
| `main()` | None | None | Loads loss_history.pth, prints summary, creates and saves the plot. |

**Detailed Flow:**

```
main()
  │
  ├── torch.load("outputs/loss_history.pth")
  │     └── [2.9979, 2.7113, ..., 0.0434]  (100 floats)
  │
  ├── print_loss_summary()
  │     ├── First loss:  2.9979  (epoch 0)
  │     ├── Final loss:  0.0434  (epoch 99)
  │     ├── Best loss:   0.0434  (epoch 99)
  │     └── Improvement: 98.6% reduction
  │
  └── plot_training_loss()
        │
        ├── plt.figure(figsize=(10, 6))           ← blank canvas
        ├── plt.plot(epochs, loss_history)          ← blue loss line
        ├── plt.axhline(y=3.87, linestyle='--')    ← red dashed baseline
        ├── plt.annotate("Start: 3.00", ...)       ← start label
        ├── plt.annotate("Final: 0.04", ...)       ← end label
        ├── plt.xlabel, ylabel, title, legend, grid ← styling
        └── plt.savefig("outputs/loss_plot.png")   ← save to disk (59 KB)
```

**Key concepts introduced:**

| Concept | Explanation | Example |
|---------|-------------|---------|
| **Loss curve** | A chart of loss vs epochs. THE key diagnostic tool in ML — shows if training is working. | Sharp drop then plateau = healthy training |
| **matplotlib** | Python's standard charting library. Creates charts from data arrays. | `plt.plot(x, y)` draws a line |
| **Agg backend** | Non-interactive rendering engine for matplotlib. Writes directly to image files without needing a display window. | `matplotlib.use('Agg')` |
| **DPI** | Dots per inch — controls image resolution. 150 DPI = 1500×900 pixels for a 10×6 inch figure. | `plt.savefig(..., dpi=150)` |

---

#### `src/generate.py`

| Property | Value |
|----------|-------|
| **Purpose** | Generate new text using the trained model (inference), with optional temperature control |
| **Created in** | Step 12, extended in Step 13 |
| **Run with** | `PYTHONPATH=src python src/generate.py` |
| **Input** | `outputs/model.pth` (trained weights), `outputs/vocab.pth` (vocabulary) |
| **Output** | Generated text printed to console |
| **Imports** | `model.py` (TinyLanguageModel) |

**Functions:**

| Function | Parameters | Returns | Description |
|----------|-----------|---------|-------------|
| `load_model(model_path, vocab_size)` | model_path (str, default="outputs/model.pth"), vocab_size (int, default=48) | `TinyLanguageModel` | Creates fresh model, loads trained weights via `load_state_dict()`, sets to eval mode. |
| `load_vocabulary(vocab_path)` | vocab_path (str, default="outputs/vocab.pth") | `dict` | Loads the saved vocabulary dictionary with chars, char_to_idx, idx_to_char. |
| `generate_text(model, vocab_data, seed_text, length)` | model, vocab_data (dict), seed_text (str, default="The"), length (int, default=200) | `str` | Generates text using greedy (argmax) decoding. Deterministic. Returns seed + generated text. |
| `generate_text_with_temperature(model, vocab_data, seed_text, length, temperature)` | model, vocab_data (dict), seed_text (str, default="The"), length (int, default=200), temperature (float, default=1.0) | `str` | Generates text using temperature-controlled sampling. Non-deterministic. Low temp = safe, high temp = creative. |
| `main()` | None | None | Loads model + vocab, compares greedy vs 6 temperature levels, generates with multiple seeds. |

**Detailed Flow (generate_text):**

```
generate_text(model, vocab_data, seed_text="The", length=200)
  │
  ├── Encode seed: "The" → [19, 29, 26]
  │
  ├── hidden = None  (RNN starts with no context)
  │
  └── with torch.no_grad():  ← disable gradient tracking (faster)
        │
        ├── Iteration 0:
        │   ├── input = tensor([[19, 29, 26]])     shape (1, 3)
        │   ├── logits, hidden = model(input)       shape (1, 3, 48)
        │   ├── next_logits = logits[0, -1, :]     shape (48,)
        │   ├── next_idx = argmax(next_logits)      → 1
        │   ├── next_char = idx_to_char[1]          → ' '
        │   └── text = "The "
        │
        ├── Iteration 1:
        │   ├── input = tensor([[1]])               shape (1, 1)
        │   │   (only last char — hidden carries context)
        │   ├── logits, hidden = model(input, hidden)
        │   ├── next_idx = argmax(...)              → 36 ('o')
        │   └── text = "The o"
        │
        ├── Iteration 2:
        │   ├── input = tensor([[36]])              → 'n'
        │   └── text = "The on"
        │
        ├── ... (200 iterations total)
        │
        └── RETURN "The only way to do great work..."
```

**Key concepts introduced:**

| Concept | Explanation | Example |
|---------|-------------|---------|
| **Inference** | Using a trained model to produce output. No learning happens — weights are frozen. | Load model → feed text → get predictions |
| **Autoregressive** | Each generated character becomes input for the next prediction. The model feeds its own output back in. | "The" → predict ' ' → "The " → predict 'o' → ... |
| **Greedy decoding** | Always pick the highest-scoring character. Deterministic but can get repetitive. | argmax([0.1, 0.7, 0.2]) → pick index 1 |
| **Seed text** | Initial text that gives the model starting context. Different seeds produce different outputs. | "The" → quotes about possibility; "Life" → quotes about living |
| **model.eval()** | Sets model to evaluation mode. Disables training-specific behaviors. Always use before inference. | `model.eval()` before generating |
| **torch.no_grad()** | Context manager that disables gradient tracking. Saves memory and speeds up inference. | `with torch.no_grad(): ...` |
| **unsqueeze(0)** | Adds a dimension at position 0. Converts (seq_len,) → (1, seq_len) to add the batch dimension. | `tensor([1,2,3]).unsqueeze(0)` → `tensor([[1,2,3]])` |

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

### Step 04 — Character Vocabulary

**What was built:** A `Vocabulary` class that converts text to numbers and back.

**Why it matters:** Neural networks only understand numbers. Before any text can enter the model, it must be converted to numbers (encoding). After the model produces number outputs, they must be converted back to text (decoding). This is the bridge between human-readable text and machine-processable numbers.

```
What changed:
  + src/vocabulary.py   ← Vocabulary class + encoding/decoding demos

Run:
  python src/vocabulary.py

Expected output:
  Encoding demos ("The" → [19, 29, 26]), decoding demos, round-trip proof,
  complete 48-character vocabulary table
```

**Key takeaway:** Tokenization is lossless — `decode(encode(text)) == text`. The model works entirely with numbers internally; we only convert back to text at the very end.

**The flow so far:**

```
data/input.txt ──(read by)──▶ explore_data.py ──(prints)──▶ statistics
       │
       └──────(read by)──▶ vocabulary.py
                              │
                              ├── Vocabulary class
                              │     .char_to_idx  {'T': 19, 'h': 29, ...}
                              │     .idx_to_char  {19: 'T', 29: 'h', ...}
                              │     .encode()     "The" → [19, 29, 26]
                              │     .decode()     [19, 29, 26] → "The"
                              │
                              └── (imported by dataset.py in Step 05)
```

---

### Step 05 — Training Sequences

**What was built:** A `TextDataset` class that chops text into (input, target) training pairs.

**Why it matters:** The model learns by seeing thousands of examples of "given these characters, predict the next one." This step creates those examples by encoding the full text and slicing it into overlapping input/target pairs where the target is the input shifted by one character.

```
What changed:
  + src/dataset.py      ← TextDataset class + training pair demos

Run:
  PYTHONPATH=src python src/dataset.py

Expected output:
  124 training pairs, character-by-character alignment showing
  how each input position maps to its target, tensor shapes
```

**Key takeaway:** The model's entire learning signal comes from predicting the next character. Input position 0 says "T", target says "h" — so the model learns that "h" often follows "T". Repeated across 124 examples × 50 positions = 6,200 learning signals per epoch.

**The flow so far:**

```
data/input.txt
       │
       ├──(read by)──▶ explore_data.py → stats, frequency chart
       │
       ├──(read by)──▶ vocabulary.py
       │                  └── Vocabulary class
       │                       .encode("The") → [19, 29, 26]
       │                       .decode([19,29,26]) → "The"
       │
       └──(read by)──▶ dataset.py
                          └── TextDataset class (uses Vocabulary)
                               .data = tensor([19, 29, 26, ...])  (6201 ints)
                               dataset[0] → (input[0:50], target[1:51])
                               dataset[1] → (input[50:100], target[51:101])
                               ...124 total training pairs
```

---

### Step 06 — DataLoader Batching

**What was built:** A `create_dataloader()` function and demos showing batching, shuffling, and epoch iteration.

**Why it matters:** Processing examples one at a time is slow. DataLoader groups them into batches of 16, which is dramatically faster because CPUs/GPUs can do math on many examples simultaneously. Shuffling the order each epoch prevents the model from memorizing the sequence of examples.

```
What changed:
  ~ src/dataset.py   ← added create_dataloader(), batching/shuffling/epoch demos

Run:
  PYTHONPATH=src python src/dataset.py

Expected output:
  Training pairs, batch shapes (16, 50), shuffle comparison,
  full epoch walkthrough (7 batches × 16 examples = 112 per epoch)
```

**Key numbers:**
- **124 examples** in the dataset
- **Batch size 16** → 7 complete batches per epoch (112 examples used, 12 dropped)
- **Shape goes from (50,) to (16, 50)** — the new first dimension is the batch

**Key takeaway:** The DataLoader is the last piece of the data pipeline. After this step, data flows from text file → vocabulary → dataset → dataloader → ready for the model. Step 07 builds the model itself.

**The flow so far:**

```
data/input.txt
       │
       └──(read by)──▶ dataset.py
                          │
                          ├── Vocabulary (from vocabulary.py)
                          │     .encode("The") → [19, 29, 26]
                          │
                          ├── TextDataset
                          │     .data = tensor([19, 29, 26, ...]) (6201 ints)
                          │     dataset[0] → (input shape (50,), target shape (50,))
                          │     124 total examples
                          │
                          └── create_dataloader(dataset, batch_size=16, shuffle=True)
                                │
                                └── DataLoader
                                      Yields batches of shape (16, 50)
                                      7 batches per epoch
                                      Shuffled each epoch
                                      │
                                      └── (fed to model in Step 07+)
```

**The shape journey so far:**

```
  Raw text         "The only way to do..."     (6201 characters)
       │
       ▼ encode()
  1D tensor        [19, 29, 26, 1, 36, ...]    shape: (6201,)
       │
       ▼ TextDataset.__getitem__()
  Training pair    input (50,), target (50,)    124 pairs total
       │
       ▼ DataLoader batching
  Batch            input (16, 50), target (16, 50)    7 batches/epoch
                          ↑    ↑
                    batch_dim  seq_dim
```

---

### Step 07 — Neural Network Model

**What was built:** A `TinyLanguageModel` class — the actual neural network with 248,880 learnable parameters across three layers.

**Why it matters:** This is the brain of the project. All previous steps prepared data; this step builds the thing that will actually learn. The model takes character numbers as input and outputs prediction scores for what the next character should be.

```
What changed:
  + src/model.py   ← TinyLanguageModel class + architecture/forward-pass demos

Run:
  PYTHONPATH=src python src/model.py

Expected output:
  Model architecture with layer details, 248,880 parameter count,
  forward pass with shape transformations (16,50) → (16,50,128) → (16,50,256) → (16,50,48)
```

**The three layers:**
| Layer | What it does | Shape change |
|---|---|---|
| **Embedding** | Converts character numbers → rich float vectors | `(16, 50)` → `(16, 50, 128)` |
| **RNN** | Processes sequence, builds up context from left to right | `(16, 50, 128)` → `(16, 50, 256)` |
| **Output** | Produces 48 prediction scores (one per character) | `(16, 50, 256)` → `(16, 50, 48)` |

**Key takeaway:** The model is just three matrix operations chained together, starting with random weights. It currently knows nothing — all 248,880 parameters are random noise. Training (Steps 08-11) will adjust these numbers until the model produces useful predictions.

**The flow so far:**

```
data/input.txt
       │
       └──▶ dataset.py (Vocabulary + TextDataset + DataLoader)
                │
                └── DataLoader yields batches: input (16, 50), target (16, 50)
                        │
                        ▼
                    model.py (TinyLanguageModel)
                        │
                        ├── Embedding:  (16, 50) → (16, 50, 128)
                        ├── RNN:        (16, 50, 128) → (16, 50, 256)
                        └── Linear:     (16, 50, 256) → (16, 50, 48)
                                                              │
                                                              ▼
                                                  48 scores per position
                                                  (one per character)
                                                              │
                                                  (compared to targets in Step 08)
```

---

### Step 08 — Loss Function and Optimizer

**What was built:** Demos of CrossEntropyLoss and Adam optimizer, plus a complete one-step learning cycle showing weights changing.

**Why it matters:** The model has 248,880 random parameters. To make them useful, we need a way to measure mistakes (loss function) and fix them (optimizer). These two components plus the model form the complete learning system.

```
What changed:
  + src/training_setup.py   ← loss function, optimizer, and one-step learning demos

Run:
  PYTHONPATH=src python src/training_setup.py

Expected output:
  Loss ≈ 3.87 (random baseline), optimizer config, one learning step
  with before/after weights, loss improvement proof
```

**Key takeaway:** ALL neural network training is just 5 lines repeated thousands of times: forward pass → compute loss → backward pass → optimizer step → zero gradients. That's it. The rest is infrastructure.

**The learning loop:**

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   DataLoader ──▶ model(input) ──▶ loss_fn(logits, target)      │
│       │              │                    │                     │
│       │         (predictions)         (one number:              │
│       │                                how wrong)               │
│       │                                   │                     │
│       │                            loss.backward()              │
│       │                           (compute gradients)           │
│       │                                   │                     │
│       │                           optimizer.step()              │
│       │                          (adjust weights)               │
│       │                                   │                     │
│       │                         optimizer.zero_grad()           │
│       │                          (clear for next)               │
│       │                                   │                     │
│       └────────── next batch ◀────────────┘                     │
│                                                                 │
│   After many iterations: loss decreases, model improves         │
└─────────────────────────────────────────────────────────────────┘
```

---

### Step 09 — Single Epoch Training

**What was built:** A `train_one_epoch()` function and a `main()` that wires up the entire pipeline for the first time — data through model through training — running one complete epoch.

**Why it matters:** This is the moment everything comes together. Steps 03-08 built individual components in isolation. Step 09 connects them into a working training system that actually learns. After one epoch (7 batches, 7 weight updates), the model shows measurable improvement.

```
What changed:
  + src/train.py   ← train_one_epoch() + full pipeline main()

Run:
  PYTHONPATH=src python src/train.py

Expected output:
  Random baseline: 3.8712
  Batch 0 loss: ~3.90  (near random)
  Batch 1 loss: ~3.70  (improving)
  ...
  Batch 6 loss: ~2.88  (well below random)
  Average loss:  ~3.33  (model is learning!)
```

**Key takeaway:** One epoch (7 weight updates) already drops the loss from random (~3.87) to ~3.33. The model is learning patterns from just one pass through the data. Step 10 will train for many epochs to drive the loss much lower.

**The flow so far:**

```
data/input.txt
       │
       └──▶ train.py  ← NEW: wires everything together
               │
               ├── Vocabulary(text)                → 48 chars
               ├── TextDataset(text, vocab, 50)    → 124 examples
               ├── create_dataloader(dataset, 16)  → 7 batches
               ├── TinyLanguageModel(48)           → 248,880 params
               ├── CrossEntropyLoss()              → measures error
               ├── Adam(lr=0.001)                  → updates weights
               │
               └── train_one_epoch()
                     │
                     ├── Batch 0: forward → loss → backward → step → zero_grad
                     ├── Batch 1: forward → loss → backward → step → zero_grad
                     ├── ...
                     └── Batch 6: forward → loss → backward → step → zero_grad
                           │
                           └── avg_loss ≈ 3.33 (improved from ~3.87!)
```

**The complete data journey (Steps 03-09):**

```
  "The only way..."          ← raw text (Step 03)
         │
         ▼
  [19, 29, 26, 1, ...]      ← encoded numbers (Step 04: Vocabulary)
         │
         ▼
  input (50,), target (50,)  ← training pairs (Step 05: TextDataset)
         │
         ▼
  input (16,50), target (16,50)  ← batched (Step 06: DataLoader)
         │
         ▼
  logits (16,50,48)          ← predictions (Step 07: Model forward pass)
         │
         ▼
  loss = 3.90                ← error measured (Step 08: CrossEntropyLoss)
         │
         ▼
  loss.backward()            ← gradients computed (Step 08: backprop)
         │
         ▼
  optimizer.step()           ← weights updated (Step 08: Adam)
         │
         ▼
  Repeat for 7 batches       ← one epoch complete (Step 09: train.py)
         │
         └── avg_loss ≈ 3.33  (learning confirmed!)
```

---

### Step 10 — Full Training and Model Saving

**What was built:** Extended `train.py` with a `train()` function that runs 100 epochs, plus `save_model()` and `save_vocabulary()` to persist everything to disk.

**Why it matters:** One epoch (Step 09) showed the model can learn, but 7 weight updates aren't enough. Step 10 runs 100 epochs (700 weight updates), driving the loss from ~3.00 down to ~0.04. The trained weights and vocabulary are saved to disk so we can generate text (Step 12) without retraining.

```
What changed:
  ~ src/train.py  ← added train(), save_model(), save_vocabulary()

Run:
  PYTHONPATH=src python src/train.py

Expected output:
  Epoch   0/100 | Avg Loss: 2.9979
  Epoch  10/100 | Avg Loss: 1.2762
  Epoch  20/100 | Avg Loss: 0.5932
  ...
  Epoch  99/100 | Avg Loss: 0.0434
  Model saved to outputs/model.pth (975.7 KB)
  Vocabulary saved to outputs/vocab.pth (2.0 KB)

Generated files:
  outputs/model.pth        ← trained weights (976 KB)
  outputs/vocab.pth        ← vocabulary mappings (2 KB)
  outputs/loss_history.pth ← loss per epoch for plotting (Step 11)
```

**Key numbers:**

| Metric | Value |
|--------|-------|
| Epochs | 100 |
| Weight updates | 700 (100 epochs × 7 batches) |
| Learning rate | 0.003 |
| First epoch loss | ~3.00 |
| Final epoch loss | ~0.04 |
| Reduction | 98.6% |
| Model file size | ~976 KB |

**Key takeaway:** Training is just the 5-line cycle from Step 08, repeated 700 times. The magic is repetition — each pass through the data refines the weights a little more. Saving the model means we never have to retrain to generate text.

**The flow so far:**

```
data/input.txt
       │
       └──▶ train.py
               │
               ├── Pipeline setup (same as Step 09)
               │     Vocabulary → TextDataset → DataLoader → Model → Loss+Optimizer
               │
               └── train(num_epochs=100)  ← NEW: multi-epoch loop
                     │
                     ├── Epoch  0: train_one_epoch() → loss ≈ 3.00
                     ├── Epoch 10: train_one_epoch() → loss ≈ 1.28
                     ├── Epoch 20: train_one_epoch() → loss ≈ 0.59
                     ├── ...
                     └── Epoch 99: train_one_epoch() → loss ≈ 0.04
                           │
                           ├── save_model()      → outputs/model.pth
                           ├── save_vocabulary()  → outputs/vocab.pth
                           └── save loss_history  → outputs/loss_history.pth
```

**What the loss curve looks like:**

```
Loss
 3.0 │*
     │ *
 2.5 │  *
     │   *
 2.0 │    *
     │     *
 1.5 │      *
     │       **
 1.0 │         **
     │           ***
 0.5 │              ****
     │                  *********
 0.0 │                           ****************************
     └───────────────────────────────────────────────────────
     0    10    20    30    40    50    60    70    80    90  100
                              Epoch

  Sharp drop (0-30):  model learns common patterns fast
  Gradual plateau (30-100): diminishing returns, fine-tuning
```

**How model saving works:**

```
SAVING (after training):
  model.state_dict()
    ├── embedding.weight    (48, 128)   ─┐
    ├── rnn.weight_ih_l0    (256, 128)   │
    ├── rnn.weight_hh_l0    (256, 256)   │
    ├── rnn.bias_ih_l0      (256,)       ├── torch.save() → outputs/model.pth
    ├── rnn.weight_ih_l1    (256, 256)   │
    ├── rnn.weight_hh_l1    (256, 256)   │
    ├── rnn.bias_*          (256,) ×2    │
    └── output_layer.*      (48, 256)   ─┘

LOADING (in Step 12 — generation):
  model = TinyLanguageModel(48)          ← fresh model (random weights)
  model.load_state_dict(torch.load("outputs/model.pth"))
                                         ← overwrite with trained weights
  Now model is ready to generate text without retraining!
```

---

### Step 11 — Loss Curve Visualization

**What was built:** A `plot_loss.py` script that loads the loss history from training and creates a professional loss curve chart.

**Why it matters:** The loss curve is the single most important diagnostic tool in machine learning. Just looking at the numbers tells you "loss went down", but the SHAPE of the curve tells you much more — how fast the model learned, when it converged, whether there were issues. Visualization makes patterns obvious that numbers alone can hide.

```
What changed:
  + src/plot_loss.py      ← loss curve plotting script

Run:
  PYTHONPATH=src python src/plot_loss.py

Expected output:
  Loss Summary (text): first 2.9979, final 0.0434, 98.6% reduction
  Loss plot saved to outputs/loss_plot.png (58.8 KB)

Generated files:
  outputs/loss_plot.png  ← the training loss curve chart
```

**Key takeaway:** The curve shows a classic healthy training pattern — sharp initial drop (epochs 0-30) as the model learns common patterns, then a gradual plateau (epochs 30-100) as it fine-tunes. The entire curve stays well below the random baseline (3.87), confirming the model learned successfully.

**Reading the loss curve:**

```
  What the shape tells you:
  ─────────────────────────────────────────────────────
  Sharp drop (epochs 0-10):
    → Model learned the easiest patterns first
      (common chars like space, 'e', 't')

  Gradual decline (epochs 10-30):
    → Learning more complex patterns
      (common words like "the", "is", "to")

  Flat plateau (epochs 30-100):
    → Model has converged — learned most of what it can
      from this data. Further training gives tiny gains.

  Random baseline (dashed red line at 3.87):
    → Where an untrained model would be.
      Everything below this line = actual learning.
```

**The flow so far:**

```
outputs/loss_history.pth   ← saved by train.py (Step 10)
       │
       └──▶ plot_loss.py   ← NEW: loads and visualizes
               │
               ├── print_loss_summary()
               │     └── Text stats: first/final/best loss, milestones
               │
               └── plot_training_loss()
                     │
                     ├── matplotlib creates figure
                     ├── Plots loss curve (blue line)
                     ├── Adds random baseline (red dashed)
                     ├── Adds annotations (start/end labels)
                     └── Saves → outputs/loss_plot.png (59 KB)
```

---

### Step 12 — Text Generation

**What was built:** A `generate.py` script that loads the trained model and generates new text character by character using greedy decoding.

**Why it matters:** This is the moment the model PRODUCES output. Everything up to now was preparation (data, model, training, visualization). Step 12 is where you see what the model actually learned — it generates text that resembles the inspirational quotes it was trained on.

```
What changed:
  + src/generate.py   ← load model + vocab, generate text with greedy decoding

Run:
  PYTHONPATH=src python src/generate.py

Expected output:
  Generated text from 5 different seeds ("The", "Life", "In the",
  "Success", "Be") — each producing ~200 characters of text that
  resembles the training data (inspirational quotes).
```

**Example output:**

```
--- Seed: "The" ---
The only impossible journey is the one you never betarn you are
one most responsive to change...

--- Seed: "Life" ---
Life is what happens when you are busy making others happy too.
The purpose of our lives is to be happy...
```

**Key takeaway:** The model learned real patterns from the training data — it generates recognizable words, phrases, and sentence structures. Greedy decoding is deterministic but repetitive; Step 13 adds temperature for more creative output.

**The flow so far:**

```
outputs/model.pth    ← trained weights (from Step 10)
outputs/vocab.pth    ← vocabulary (from Step 10)
       │
       └──▶ generate.py  ← NEW: text generation
               │
               ├── load_model()
               │     ├── TinyLanguageModel(48)     ← fresh model
               │     ├── load_state_dict(...)       ← load trained weights
               │     └── model.eval()               ← inference mode
               │
               ├── load_vocabulary()
               │     └── {chars, char_to_idx, idx_to_char}
               │
               └── generate_text("The", length=200)
                     │
                     ├── Encode "The" → [19, 29, 26]
                     ├── Model predicts → ' ' → "The "
                     ├── Model predicts → 'o' → "The o"
                     ├── Model predicts → 'n' → "The on"
                     ├── ... (200 iterations)
                     └── "The only way to do great work..."
```

**The complete pipeline (Steps 03-12):**

```
  data/input.txt                      ← raw text
       │
       ▼
  Vocabulary + TextDataset + DataLoader  ← data pipeline (Steps 04-06)
       │
       ▼
  TinyLanguageModel                   ← architecture (Step 07)
       │
       ▼
  train() for 100 epochs              ← training (Steps 08-10)
       │
       ▼
  outputs/model.pth + vocab.pth       ← saved to disk (Step 10)
       │
       ▼
  plot_loss.py → loss_plot.png        ← visualize training (Step 11)
       │
       ▼
  generate.py → "The only way..."    ← text generation (Step 12) ✓
```

---

### Step 13 — Temperature Control

**What was built:** A `generate_text_with_temperature()` function that adds randomness control to text generation using the temperature parameter.

**Why it matters:** Greedy decoding (Step 12) is deterministic and repetitive. Temperature-controlled sampling lets you dial in the creativity level — from safe and predictable (0.3) to wild and experimental (2.0). This is the same "temperature" slider used in ChatGPT, Claude, and all modern language models.

```
What changed:
  ~ src/generate.py  ← added generate_text_with_temperature(), updated main()

Run:
  PYTHONPATH=src python src/generate.py

Expected output:
  Greedy text, then 6 temperature levels (0.3, 0.5, 0.8, 1.0, 1.5, 2.0)
  showing progressively more creative/random output.
  Multiple seeds at temperature 0.8.
```

**Temperature comparison (same seed "The"):**

```
Temp 0.3: "The only way to do great work is to love what you do..."
          → Clean, predictable, close to training data

Temp 0.8: "The only impossible journey is the one you never bet..."
          → Slightly creative, still coherent

Temp 1.5: "The only impossible jound is in the direction..."
          → Unexpected word choices, some errors

Temp 2.0: "The only way to do greapEy to be I and ghe shots..."
          → Mostly nonsense — too much randomness
```

**Key takeaway:** Temperature is a single number that controls the tradeoff between predictability and creativity. 0.8 is a practical sweet spot — creative enough to avoid repetition, safe enough to stay coherent.

**How temperature works (the math):**

```
Model outputs logits:  [8.0, 5.0, 2.0, 1.0]

Step 1: Divide by temperature
  temp=0.3 → [26.7, 16.7, 6.7, 3.3]   (differences amplified)
  temp=1.0 → [8.0, 5.0, 2.0, 1.0]     (unchanged)
  temp=2.0 → [4.0, 2.5, 1.0, 0.5]     (differences compressed)

Step 2: softmax → probabilities
  temp=0.3 → [0.999, 0.001, 0.000, 0.000]  (winner takes all)
  temp=1.0 → [0.933, 0.046, 0.002, 0.001]  (clear favorite)
  temp=2.0 → [0.621, 0.215, 0.049, 0.030]  (more spread out)

Step 3: Sample from distribution
  temp=0.3 → almost always picks index 0
  temp=2.0 → index 1 has 22% chance of being picked
```

**The flow so far:**

```
outputs/model.pth + vocab.pth
       │
       └──▶ generate.py
               │
               ├── generate_text()               ← greedy (Step 12)
               │     └── argmax(logits)           → deterministic
               │
               └── generate_text_with_temperature()  ← NEW (Step 13)
                     ├── logits / temperature      → scale scores
                     ├── F.softmax(scaled)          → probabilities
                     └── torch.multinomial(probs)   → random sample
                           │
                           ├── temp 0.3 → safe, predictable
                           ├── temp 0.8 → creative, coherent
                           └── temp 2.0 → wild, often nonsense
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
| **Tokenization** | Converting text into numbers. Each character is mapped to a unique integer. | Step 04 |
| **Encoding** | The text → numbers direction of tokenization. `"The"` becomes `[19, 29, 26]`. | Step 04 |
| **Decoding** | The numbers → text direction. `[19, 29, 26]` becomes `"The"`. | Step 04 |
| **char_to_idx** | Dictionary that maps each character to its number. The encoding lookup table. | Step 04 |
| **idx_to_char** | Reverse dictionary that maps each number back to its character. The decoding lookup table. | Step 04 |
| **Deterministic** | Same input always gives same output. Important for reproducibility in AI. | Step 04 |
| **Training pair** | An (input, target) example. Model sees input, tries to predict target. Target = input shifted by 1. | Step 05 |
| **Sequence length** | How many characters per training example. Our model uses 50; real LLMs use 4096+. | Step 05 |
| **PyTorch Dataset** | A class with `__len__` and `__getitem__` — the standard way to serve training data in PyTorch. | Step 05 |
| **Input→target shift** | Target is the input shifted right by 1 char. Position 0: input='T', target='h'. | Step 05 |
| **Neural network** | A series of math layers that transform input into predictions. Weights are learned through training. | Step 07 |
| **Embedding** | Converts a plain integer into a rich vector of floats. `19` ('T') → `[0.3, -0.7, ...]` (128 numbers). | Step 07 |
| **RNN** | Recurrent Neural Network — processes sequences position by position, building up context as it goes. | Step 07 |
| **Hidden state** | The RNN's "working memory" — carries context from earlier positions to later ones. | Step 07 |
| **Logits** | Raw prediction scores before probabilities. Highest score = model's best guess for next character. | Step 07 |
| **Forward pass** | The path data takes through all layers: input → embedding → RNN → output → prediction scores. | Step 07 |
| **Parameters/weights** | The learnable numbers inside the model (248,880 in ours). Start random, become meaningful via training. | Step 07 |
| **Hyperparameters** | Settings chosen before training (embed_size, hidden_size, etc.). NOT learned. | Step 07 |
| **nn.Module** | PyTorch base class for all neural networks. Tracks parameters and provides save/load. | Step 07 |
| **Loss** | A number measuring how wrong the model is. Random baseline ≈ 3.87. Good training → ~1.0. | Step 08 |
| **CrossEntropyLoss** | Loss function for classification. Converts logits to probabilities, scores the correct answer. | Step 08 |
| **Epoch** | One complete pass through the entire training dataset. With 124 examples and batch_size=16, one epoch = 7 batches. | Step 09 |
| **Batch** | A group of training examples processed together. Our batches hold 16 examples each. | Step 06 |
| **Batch size** | Number of examples per batch. 16 for us. Larger = faster but more memory. Common: 8-128. | Step 06 |
| **DataLoader** | PyTorch utility that wraps a Dataset to provide batching, shuffling, and iteration. | Step 06 |
| **Shuffling** | Randomizing the order of examples each epoch. Prevents memorizing the order. | Step 06 |
| **drop_last** | DataLoader option to discard the last incomplete batch. Keeps all batches same size. | Step 06 |
| **Gradient** | For each weight: "if I increase this, how does the loss change?" Computed by `loss.backward()`. | Step 08 |
| **Backpropagation** | The algorithm that computes gradients for all weights by tracing back from the loss. The key algorithm in deep learning. | Step 08 |
| **Optimizer** | Algorithm that uses gradients to adjust weights. Adam is the most popular — adaptive learning rate per weight. | Step 08 |
| **Learning rate** | How big each weight adjustment is. lr=0.001 is a common default for Adam. Too high → unstable, too low → slow. | Step 08 |
| **zero_grad()** | Clears accumulated gradients before the next step. Forgetting this is a common PyTorch bug. | Step 08 |
| **Training loop** | The outer loop that iterates over all batches, running the 5-step cycle on each. One iteration of this loop = one epoch. | Step 09 |
| **model.train()** | Sets the model to training mode. Enables training-specific layers (dropout, batchnorm). Always call before training. | Step 09 |
| **Average loss** | Mean loss across all batches in one epoch. Smooths noise for a clearer picture of learning progress. | Step 09 |
| **Random baseline** | Expected loss if the model guessed uniformly: -log(1/vocab_size). For 48 chars ≈ 3.87. Untrained models start here. | Step 09 |
| **Pipeline** | The end-to-end flow from raw text through all components to trained weights. Step 09 wires it up for the first time. | Step 09 |
| **Multi-epoch training** | Running many passes (epochs) over the same data. More epochs = more refined weights. | Step 10 |
| **Loss history** | A list of average loss values, one per epoch: `[3.00, 2.71, ..., 0.04]`. Used for plotting the training curve. | Step 10 |
| **state_dict()** | Dictionary of all learnable parameters in a model. The standard way to save and load PyTorch models. | Step 10 |
| **torch.save()** | Serializes Python objects (tensors, dicts) to a binary file on disk. Convention: `.pth` extension. | Step 10 |
| **Convergence** | When loss stops decreasing — the model has learned what it can from the data with its current capacity. | Step 10 |
| **Serialization** | Converting in-memory objects (like tensors) to bytes that can be written to disk and loaded later. | Step 10 |
| **Loss curve** | A chart plotting loss vs epoch. The key diagnostic: shape reveals learning speed, convergence, and problems. | Step 11 |
| **matplotlib** | Python's standard charting/plotting library. `plt.plot(x, y)` draws a line chart. Used by scientists and engineers worldwide. | Step 11 |
| **Agg backend** | Non-interactive matplotlib renderer. Writes charts directly to image files without needing a GUI window. | Step 11 |
| **Plateau** | When the loss curve flattens — the model has converged and further training yields diminishing returns. | Step 11 |
| **Inference** | Using a trained model to produce output (generate text). No learning happens — weights are frozen. | Step 12 |
| **Autoregressive** | Each output becomes input for the next step. The model generates one char, feeds it back in, generates the next. | Step 12 |
| **Greedy decoding** | Always pick the highest-scoring prediction. Deterministic but can get stuck in repetitive loops. | Step 12 |
| **Seed text** | Starting text that gives the model initial context. Different seeds lead to different generated text. | Step 12 |
| **torch.no_grad()** | Context manager that disables gradient tracking during inference. Saves memory and speeds up generation. | Step 12 |
| **load_state_dict()** | Loads saved weights into a model. Reverses the save process — overwrites random weights with trained ones. | Step 12 |
| **Temperature** | Controls randomness by scaling logits before softmax. Low (0.3) = predictable, high (2.0) = chaotic. Same concept used in ChatGPT/Claude. | Step 13 |
| **Softmax** | Converts raw scores (logits) into probabilities summing to 1.0. `softmax(x_i) = exp(x_i) / sum(exp(x_j))`. | Step 13 |
| **Sampling** | Randomly picking from a probability distribution instead of always choosing the maximum. Makes output non-deterministic. | Step 13 |
| **torch.multinomial** | Draws random samples from a probability distribution. The core of temperature-based generation. | Step 13 |
| **Overfitting** | When a model memorizes training data instead of learning general patterns. | Step 15 (upcoming) |

---

> *This document is updated with each new step. Last updated: Step 13.*
