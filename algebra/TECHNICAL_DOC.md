# Algebra Solver — Technical Documentation

Detailed documentation of every file, method, and concept in the algebra solver. Updated with each step.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [How Seq2Seq Works (Concept)](#how-seq2seq-works)
3. [Data Pipeline](#data-pipeline)
4. [Model Architecture](#model-architecture)
5. [Training Pipeline](#training-pipeline)
6. [Inference Pipeline](#inference-pipeline)
7. [File Reference](#file-reference)
8. [Glossary](#glossary)

---

## Architecture Overview

The algebra solver uses an **encoder-decoder (seq2seq)** architecture — two neural networks working together:

```
                    ENCODER                              DECODER
              ┌──────────────────┐                ┌──────────────────┐
              │                  │  context vector │                  │
"2x + 3 = 7" │  Reads equation  │ ──────────────> │  Writes solution │ "x = 2"
  (input)     │  character by    │   (fixed-size   │  character by    │  (output)
              │  character       │    summary)     │  character       │
              └──────────────────┘                └──────────────────┘
```

### Why two networks?

The input and output have **different lengths and structures**:
- Input: `"2x + 3 = 7"` (10 characters)
- Output: `"x = 2"` (5 characters)

A single RNN (like in the text generator) always produces output the same length as input. The encoder-decoder solves this by:
1. **Encoder**: compresses the entire input into a fixed-size vector (the "context vector")
2. **Decoder**: expands that vector into an output of any length

This is the same architecture used for:
- **Machine translation** (English → French)
- **Text summarization** (long article → short summary)
- **Chatbots** (question → answer)

### Full Data Flow

```
"2x + 3 = 7"
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: Tokenize                                                │
│   "2x + 3 = 7" → [5, 13, 1, 8, 1, 6, 1, 10, 1, 11]          │
│   Each character gets a number from the vocabulary              │
├─────────────────────────────────────────────────────────────────┤
│ STEP 2: Encode                                                  │
│   Embedding: 10 indices → 10 vectors of 64 dimensions          │
│   GRU: reads all 10 vectors → produces context vector (256-dim)│
│   The context vector summarizes the entire equation             │
├─────────────────────────────────────────────────────────────────┤
│ STEP 3: Decode                                                  │
│   Start with <SOS> token + context vector                       │
│   Step 1: <SOS> → predicts "x"                                 │
│   Step 2: "x"   → predicts " "                                 │
│   Step 3: " "   → predicts "="                                 │
│   Step 4: "="   → predicts " "                                 │
│   Step 5: " "   → predicts "2"                                 │
│   Step 6: "2"   → predicts <EOS>  ← stop signal                │
├─────────────────────────────────────────────────────────────────┤
│ STEP 4: Detokenize                                              │
│   [13, 1, 10, 1, 5] → "x = 2"                                 │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
"x = 2"
```

---

## How Seq2Seq Works

### The Encoder — "Reading Comprehension"

The encoder's job is to **read and understand the equation**. It processes the input one character at a time (left to right AND right to left, because it's bidirectional) and produces a single vector that summarizes everything.

```
Forward pass (left to right):
  "2" → "x" → " " → "+" → " " → "3" → " " → "=" → " " → "7"
   h₁    h₂    h₃    h₄    h₅    h₆    h₇    h₈    h₉    h₁₀→

Backward pass (right to left):
  "7" → " " → "=" → " " → "3" → " " → "+" → " " → "x" → "2"
  ←h₁₀  ←h₉   ←h₈   ←h₇   ←h₆   ←h₅   ←h₄   ←h₃   ←h₂   ←h₁

Final hidden state = [forward h₁₀ ; backward h₁] = context vector
```

The context vector (256 dimensions) is the encoder's "summary" of the entire equation. It must capture:
- What is the coefficient of x? (2)
- What is the constant? (+3)
- What does it equal? (7)

### The Decoder — "Answer Writing"

The decoder receives the context vector and generates the answer one character at a time:

```
                context vector
                     │
                     ▼
<SOS> ──→ [GRU] ──→ "x"    (first character of answer)
"x"   ──→ [GRU] ──→ " "    (space)
" "   ──→ [GRU] ──→ "="    (equals sign)
"="   ──→ [GRU] ──→ " "    (space)
" "   ──→ [GRU] ──→ "2"    (the answer!)
"2"   ──→ [GRU] ──→ <EOS>  (stop generating)
```

At each step, the decoder:
1. Takes the previous character (or `<SOS>` for the first step)
2. Combines it with its hidden state (initialized from the context vector)
3. Produces a probability distribution over all characters
4. Picks the most likely character (greedy decoding)

### Teacher Forcing — "Training Wheels"

During training, the decoder might predict the wrong character at step 2. If we feed that wrong character to step 3, the errors compound and the model can't learn.

**Teacher forcing** solves this by always feeding the **correct** character during training:

```
WITHOUT teacher forcing (errors compound):
  <SOS> → predicts "x" ✓ → feed "x"
  "x"   → predicts "+"  ✗ → feed "+" (wrong!)
  "+"   → predicts "3"  ✗ → feed "3" (everything derails)
  ...total garbage output

WITH teacher forcing (correct input regardless):
  <SOS> → predicts "x" ✓ → feed "x" (correct answer)
  "x"   → predicts "+"  ✗ → feed " " (correct answer, ignoring wrong prediction)
  " "   → predicts "="  ✓ → feed "=" (correct answer)
  ...model learns at each step independently
```

We gradually reduce teacher forcing during training so the model learns to use its own predictions.

### Special Tokens

```
<PAD> (index 0): Padding — fills empty space when batching different-length sequences
                  "x = 2<EOS><PAD><PAD>"  ← padded to match longer sequences
                  Ignored during loss calculation (don't penalize model for padding)

<SOS> (index 1): Start Of Sequence — tells decoder "begin generating now"
                  Always the first input to the decoder
                  The decoder never predicts <SOS> — it only receives it

<EOS> (index 2): End Of Sequence — tells decoder "stop generating"
                  Appended to every target sequence during training
                  During inference, generation stops when model predicts <EOS>
```

---

## Data Pipeline

### Data Generation (Step 02) — `algebra/src/generate_data.py`

**Key insight — answer-first generation:**
Instead of generating random equations and solving them (which produces ugly decimals), we generate the **answer first** and work backwards to create a clean equation. This guarantees integer answers every time.

```
Traditional approach (messy):
  Generate a=3, b=10  →  3x = 10  →  x = 3.333...  ← ugly

Our approach (clean):
  Generate a=3, x=4   →  b = 3×4 = 12  →  "3x = 12"  →  "x = 4"  ← clean
```

**Equation types and generation strategy:**

| Type | Template | How it's generated | Example |
|---|---|---|---|
| `ax = b` | Pick a, x_val; b = a×x | `3x = 12` → `x = 4` |
| `ax + b = c` | Pick a, x_val, b; c = a×x+b | `2x + 3 = 7` → `x = 2` |
| `ax - b = c` | Pick a, x_val, b; c = a×x-b | `5x - 10 = 15` → `x = 5` |
| `b + ax = c` | Pick a, x_val, b; c = b+a×x | `3 + 4x = 11` → `x = 2` |
| `b - ax = c` | Pick a, x_val, b; c = b-a×x | `10 - 2x = 4` → `x = 3` |
| `x / a = b` | Pick a, b; x = a×b | `x / 4 = 3` → `x = 12` |
| `ax + b = cx + d` | Pick a, c, x_val, b; d = (a-c)×x+b | `3x + 2 = x + 8` → `x = 3` |

**Dataset statistics (actual output from generation):**

```
Total equations:  50,000
Duplicates skipped: 10,620

Type distribution:
  Type 1 (ax = b)           :  3,357 (6.7%)
  Type 2 (ax + b = c)       : 10,319 (20.6%)
  Type 3 (ax - b = c)       : 10,091 (20.2%)
  Type 4 (b + ax = c)       :  7,008 (14.0%)
  Type 5 (b - ax = c)       :  6,860 (13.7%)
  Type 6 (x / a = b)        :  1,602 (3.2%)
  Type 7 (ax + b = cx + d)  : 10,763 (21.5%)

Answer distribution:
  Positive: 24,466 (48.9%)
  Negative: 24,862 (49.7%)
  Zero:        672 (1.3%)

Sequence lengths:
  Equation: min=5, max=22, avg=14.5
  Solution: min=5, max=8,  avg=6.3

Characters used (16): [' ', '+', '-', '/', '0'-'9', '=', 'x']
```

**Verification:** Every equation is checked by substitution — the x value is plugged back into both sides of the equation, and they must be equal (within 1e-6 tolerance). 0 equations failed verification.

**File format — tab-separated:**
```
2x + 3 = 7\tx = 2
5x - 10 = 15\tx = 5
```
Tab is used as delimiter because equations contain spaces.

### Vocabulary (Step 03) — `algebra/src/vocab.py`

**Token → Index mapping (confirmed from implementation):**

| Token | Index | Purpose |
|---|---|---|
| `<PAD>` | 0 | Padding (ignored in loss) |
| `<SOS>` | 1 | Start of sequence (decoder input) |
| `<EOS>` | 2 | End of sequence (stop signal) |
| ` ` (space) | 3 | Space between terms |
| `+` | 4 | Addition |
| `-` | 5 | Subtraction / negative sign |
| `/` | 6 | Division |
| `0`-`9` | 7-16 | Digits |
| `=` | 17 | Equals sign |
| `x` | 18 | Variable |

**Total: 19 tokens** (3 special + 16 data characters)

Note: `*` (multiplication) and `.` (decimal point) are in the token list in REQUIREMENTS.md but don't appear in the current dataset — our answer-first strategy produces only integer answers, and equations use implicit multiplication (`2x` not `2*x`). They'll be added if needed for future equation types.

**Key methods:**
- `encode("2x + 3 = 7")` → `[9, 18, 3, 4, 3, 10, 3, 17, 3, 14]`
- `encode_with_eos("x = 2")` → `[18, 3, 17, 3, 9, 2]` (appends EOS)
- `decode([18, 3, 17, 3, 9])` → `"x = 2"` (skips special tokens)
- `decode_until_eos([18, 3, 17, 3, 9, 2, 9, 9])` → `"x = 2"` (stops at EOS)

### Dataset (Step 04) — `algebra/src/dataset.py`

**Three tensors per example:**

Each equation-solution pair produces three tensors for the model:

```
Equation: "17x - 14 = -660"    Solution: "x = -38"

Encoder input:  [8,14,18,3,5,3,8,11,3,17,3,5,13,13,7, 2, 0,0,0,0,0,0,0]
                 1  7  x    -    1  4     =    -  6  6  0 EOS  ← padding →

Decoder input:  [1, 18,3,17,3,5,10,15]
                SOS  x    =    -  3  8

Decoder target: [18,3,17,3,5,10,15, 2]
                  x    =    -  3  8 EOS
```

The decoder input/target are offset by 1: at each step the decoder receives a token and must predict the next.

**Train/test split:**
- 40,000 training pairs (80%), 10,000 test pairs (20%)
- Shuffled with fixed seed=42 for reproducibility
- 0 overlap verified (no data leakage)

**Batch shapes (batch_size=64):**
- Encoder input: `(64, max_eq_len)` — padded to longest equation in batch
- Decoder input: `(64, max_sol_len)` — padded to longest solution in batch
- Decoder target: `(64, max_sol_len)` — same shape as decoder input

**Collate function:** Custom `collate_fn()` uses `pad_sequence(padding_value=PAD_IDX)` to handle variable-length sequences within each batch.

**DataLoader config:**
- Train: `shuffle=True, drop_last=True` → 625 batches
- Test: `shuffle=False, drop_last=False` → 157 batches

---

## Model Architecture

### Encoder (Step 05) — `algebra/src/encoder.py`

```
Encoder(
  embedding:  Embedding(19, 64, padding_idx=0)
  gru:        GRU(64, 128, num_layers=2, bidirectional=True, dropout=0.1)
  fc_hidden:  Linear(256, 256)         ← bridge to decoder hidden size
)
Total: 512,448 parameters
```

**Key design choices:**
- `padding_idx=0`: PAD tokens produce zero vectors (no contribution)
- Bidirectional: reads equation left→right AND right→left
- Hidden bridge: reshapes bidirectional output (2×128=256) for unidirectional decoder
- `tanh` activation on bridge to bound values to [-1, 1]

**Output:** `(batch, seq_len, 256)` per-position context + `(2, batch, 256)` final hidden

### Decoder (Step 06) — `algebra/src/decoder.py`

```
Decoder(
  embedding:    Embedding(19, 64, padding_idx=0)
  gru:          GRU(64, 256, num_layers=2, dropout=0.1)
  output_layer: Linear(256, 19)
)
Total: 648,147 parameters
```

**Key design choices:**
- Unidirectional (left→right only — can't peek at future tokens)
- `hidden_size=256` matches encoder's bidirectional output (128×2)
- Teacher forcing ratio configurable per call (1.0 → 0.5 during training)
- `max_length=20` safety limit during inference

**Two modes:**
- Training: `forward(hidden, target, tf_ratio)` → logits `(batch, target_len, 19)`
- Inference: `generate(hidden)` → predictions `(batch, max_length)`

### Seq2Seq Combined (Step 07) — `algebra/src/seq2seq.py`

```
Seq2Seq(
  encoder: Encoder(512,448 params)
  decoder: Decoder(648,147 params)
)
Total: 1,160,595 parameters (~4.7× the text generator's 248K)
```

**Gradient verification:** Loss backpropagates through both decoder AND encoder — both networks learn simultaneously.

**Interface:**
- `model(src, target, tf_ratio)` → training (returns logits)
- `model.solve(src)` → inference (returns predicted indices)

---

## Training Pipeline

### Configuration
```
Epochs:           50
Batch size:       64
Learning rate:    0.002 (initial)
Optimizer:        Adam
LR scheduler:    StepLR(step_size=15, gamma=0.5)
                  → LR halves every 15 epochs: 0.002 → 0.001 → 0.0005
Loss function:   CrossEntropyLoss(ignore_index=PAD_IDX)
Teacher forcing:  1.0 → 0.3 (linear decrease over 50 epochs)
Gradient clip:    1.0 (prevents exploding gradients)
Training time:   ~15 minutes on CPU
```

### Loss Calculation

The loss is computed **per character of the output**, ignoring padding:

```
Target:     [x,  =,  2, <EOS>, <PAD>, <PAD>]
Prediction: [x,  =,  3, <EOS>, <PAD>, <PAD>]
Loss:        ✓   ✓   ✗   ✓     skip   skip
                      ↑
              penalty for predicting "3" instead of "2"
```

### Training Curve (Actual Results)

```
Epoch 10: Loss 0.1229, Accuracy 68.9%, TF 0.87    (learning structure)
Epoch 20: Loss 0.0538, Accuracy 85.4%, TF 0.73    (most equations correct)
Epoch 30: Loss 0.0484, Accuracy 87.7%, TF 0.59    (fine-tuning)
Epoch 40: Loss 0.0225, Accuracy 92.1%, TF 0.44    (above 90% target!)
Epoch 50: Loss 0.0098, Accuracy 94.6%, TF 0.30    (final — target exceeded)
```

### Key Training Decisions

1. **LR Scheduler**: StepLR halves the learning rate every 15 epochs. Early epochs use large steps for fast learning, later epochs use smaller steps for fine-tuning.

2. **Teacher Forcing 1.0 → 0.3**: Starts with full teacher forcing (always feed correct tokens) and gradually forces the model to rely on its own predictions. The linear decrease is more gradual than a step schedule.

3. **Gradient Clipping at 1.0**: RNNs can produce very large gradients during backpropagation (exploding gradients). Clipping caps the total gradient norm, keeping weight updates stable.

### Training History

Previous runs that informed the final configuration:
```
Run 1: 30 epochs, LR=0.001, TF 1.0→0.5  → 73.8%  (TF dropped too fast)
Run 2: 50 epochs, LR=0.001, TF 1.0→0.3  → 83.8%  (LR too low for later epochs)
Run 3: 50 epochs, LR=0.002, StepLR       → 94.6%  ← final (target met!)
```

---

## Evaluation Results

### Overall Accuracy
```
Overall accuracy: 94.6% (9,458 / 10,000)    ← TARGET MET (≥ 90%)
```

### Per-Type Accuracy
```
Type 1: ax = b              99.4%  (693/697)    ✓ ≥ 80%
Type 2: ax + b = c           97.8%  (2120/2167)  ✓ ≥ 80%
Type 3: ax - b = c           97.1%  (1887/1943)  ✓ ≥ 80%
Type 4: b + ax = c           98.3%  (1360/1383)  ✓ ≥ 80%
Type 5: b - ax = c           97.7%  (1311/1342)  ✓ ≥ 80%
Type 6: x / a = b            65.9%  (203/308)    ✗ below 80%
Type 7: ax + b = cx + d      87.2%  (1884/2160)  ✓ ≥ 80%
```

### By Answer Sign
```
Positive answers:  94.7%  (4639/4901)
Negative answers:  94.5%  (4686/4960)
Zero answers:      95.7%  (133/139)
```

### Known Limitation: Type 6 (Division)
Type 6 (x / a = b) achieves only 65.9% accuracy. This is because:
1. **Few training examples**: only 1,602 out of 50,000 (3.2%)
2. **Large answers**: x = a × b can reach 280+, making character prediction harder
3. **Multiplication is harder**: the model must learn to multiply (reverse the division), which produces multi-digit numbers it hasn't seen enough

### Substitution Verification
100% of correct predictions were verified by plugging the answer back into the equation.

### Error Patterns
Most errors are small numeric mistakes (off-by-one or off-by-a-few), not garbled output. The model understands the format perfectly — it just sometimes gets the exact digit wrong.

---

## Inference Pipeline

### Greedy Decoding

```
Input: "2x + 3 = 7"

1. Encoder processes "2x + 3 = 7" → context vector (256-dim)
2. Decoder starts with <SOS> + context vector
3. Loop:
   a. Decoder predicts probability for each character
   b. Pick the character with highest probability (argmax)
   c. If it's <EOS>, stop
   d. Otherwise, feed it back and repeat
4. Concatenate all predicted characters → "x = 2"
```

### Verification by Substitution

```
Equation: 2x + 3 = 7
Model says: x = 2

Verify:
  Left side:  2(2) + 3 = 4 + 3 = 7
  Right side: 7
  7 == 7 → ✓ CORRECT
```

---

## Experiment Results

Four experiments were run (10 epochs each) to understand what affects accuracy:

### Experiment 1: Data Size
```
 2,000 examples →  2% accuracy   (not enough to learn patterns)
10,000 examples → 37% accuracy   (learns structure, struggles with digits)
40,000 examples → 63% accuracy   (strong learning, would reach 94%+ with more epochs)
```
**Takeaway:** More data helps significantly. The model needs at least 10K examples to start learning, and 40K+ to reach high accuracy.

### Experiment 2: Model Size
```
Small  (enc=64,  dec=128)    319K params → 57% accuracy
Medium (enc=128, dec=256)  1,160K params → 63% accuracy  ← our default
Large  (enc=256, dec=512)  4,415K params → 61% accuracy
```
**Takeaway:** Bigger is not always better. The large model is 4x bigger but slightly worse than medium — it needs more training time to leverage its extra capacity. Medium is the sweet spot for this task.

### Experiment 3: Equation Complexity
```
Type 1: ax = b           → 99.4%  (easiest — simple pattern)
Type 2: ax + b = c        → 97.8%
Type 3: ax - b = c        → 97.1%
Type 4: b + ax = c        → 98.3%
Type 5: b - ax = c        → 97.7%
Type 6: x / a = b         → 65.9%  (hardest — large multi-digit answers)
Type 7: ax + b = cx + d   → 87.2%  (harder — x on both sides)
```
**Takeaway:** Division is the hardest because x = a×b produces large numbers. Types 1-5 are all above 97%. Type 7 (x on both sides) requires more reasoning.

### Experiment 4: Teacher Forcing
```
Always (TF=1.0) → 67% accuracy
Half   (TF=0.5) → 68% accuracy  ← best in 10 epochs
Never  (TF=0.0) → 62% accuracy
```
**Takeaway:** Teacher forcing is critical for learning. Without it (TF=0), the model learns much slower. TF=0.5 slightly outperforms TF=1.0 because the model also practices generating on its own. Our scheduled approach (1.0→0.3) gets the best of both.

---

## File Reference

*(Updated as files are created)*

| File | Step | Purpose |
|---|---|---|
| `algebra/REQUIREMENTS.md` | 01 | Execution plan and requirements |
| `algebra/TECHNICAL_DOC.md` | 01 | This file — technical documentation |
| `algebra/src/generate_data.py` | 02 | Generate synthetic equation data |
| `algebra/src/vocab.py` | 03 | Character vocabulary with special tokens |
| `algebra/src/dataset.py` | 04 | PyTorch Dataset with padding and splitting |
| `algebra/src/encoder.py` | 05 | Encoder network (reads equation) |
| `algebra/src/decoder.py` | 06 | Decoder network (writes solution) |
| `algebra/src/seq2seq.py` | 07 | Combined encoder-decoder model |
| `algebra/src/train.py` | 08 | Training loop with teacher forcing |
| `algebra/src/evaluate.py` | 09 | Accuracy evaluation on test set |
| `algebra/src/plot_loss.py` | 10 | Loss and accuracy visualization |
| `algebra/src/interactive.py` | 11 | Interactive equation solver prompt |
| `algebra/tests/test_e2e.py` | 12 | End-to-end test suite |
| `algebra/src/experiments.py` | 13 | Hyperparameter experiments |

---

## Glossary

New terms introduced in this project (terms from the text generator are not repeated):

| # | Term | Definition |
|---|---|---|
| 1 | **Seq2Seq** | Sequence-to-sequence — an architecture that transforms one sequence into another (e.g., equation → solution) |
| 2 | **Encoder** | The neural network that reads the input and compresses it into a fixed-size context vector |
| 3 | **Decoder** | The neural network that takes the context vector and generates the output one token at a time |
| 4 | **Context Vector** | A fixed-size vector (e.g., 256 dimensions) that summarizes the entire input — the "bridge" between encoder and decoder |
| 5 | **GRU** | Gated Recurrent Unit — an improved RNN that uses gates to control information flow, better at remembering long sequences |
| 6 | **Bidirectional** | Processing a sequence in both directions (left→right and right→left) to capture full context |
| 7 | **Teacher Forcing** | Training technique where the decoder receives the correct previous token (not its own prediction) to speed up learning |
| 8 | **`<PAD>`** | Padding token — fills shorter sequences to match the longest in a batch |
| 9 | **`<SOS>`** | Start Of Sequence token — signals the decoder to begin generating |
| 10 | **`<EOS>`** | End Of Sequence token — signals the decoder to stop generating |
| 11 | **Synthetic Data** | Training data generated programmatically (by code) rather than collected from the real world |
| 12 | **Train/Test Split** | Dividing data into a training set (model learns from) and test set (model is evaluated on, never seen during training) |
| 13 | **Exact-Match Accuracy** | Percentage of predictions that are exactly correct (character-for-character match with the expected answer) |
| 14 | **Gradient Clipping** | Capping gradient values during training to prevent "exploding gradients" that destabilize learning |
| 15 | **Collate Function** | Custom function that assembles individual data samples into a padded batch for the DataLoader |
| 16 | **Data Leakage** | When test data accidentally appears in training data, giving falsely optimistic accuracy numbers |
| 17 | **Teacher Forcing Ratio** | The probability of using the correct token vs the model's own prediction during training (1.0 = always correct, 0.0 = always own prediction) |
| 18 | **Substitution Check** | Verifying an answer by plugging the value back into the original equation and checking both sides are equal |
| 19 | **Learning Rate Scheduler** | Automatically reduces the learning rate during training (e.g., halve every 15 epochs) so early epochs learn fast and later epochs fine-tune |
| 20 | **StepLR** | A PyTorch scheduler that multiplies the learning rate by gamma every step_size epochs (e.g., 0.002 → 0.001 → 0.0005) |
| 21 | **CrossEntropyLoss** | Loss function that measures how wrong the model's probability predictions are — penalizes confident wrong answers heavily |
| 22 | **ignore_index** | Parameter in CrossEntropyLoss that tells it to skip certain positions (like `<PAD>`) when computing the loss |

---

*This document is updated with each step. Last updated: Step 13 (experiments — final step).*
