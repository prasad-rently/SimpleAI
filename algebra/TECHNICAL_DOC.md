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

*(Updated as steps are implemented)*

### Data Generation (Step 02)

**Equation types and generation strategy:**

| Type | Template | How x is solved | Example |
|---|---|---|---|
| `ax = b` | Generate a,b; x = b/a | Division | `3x = 12` → `x = 4` |
| `ax + b = c` | Generate a,b,c; x = (c-b)/a | Subtract, divide | `2x + 3 = 7` → `x = 2` |
| `ax - b = c` | Generate a,b,c; x = (c+b)/a | Add, divide | `5x - 10 = 15` → `x = 5` |
| `b + ax = c` | Generate a,b,c; x = (c-b)/a | Same math, different order | `3 + 4x = 11` → `x = 2` |
| `b - ax = c` | Generate a,b,c; x = (b-c)/a | Subtract from b | `10 - 2x = 4` → `x = 3` |
| `x / a = b` | Generate a,b; x = a*b | Multiply | `x / 4 = 3` → `x = 12` |
| `ax + b = cx + d` | Generate a,b,c,d; x = (d-b)/(a-c) | Collect terms | `3x + 2 = x + 8` → `x = 3` |

### Vocabulary (Step 03)

**Token → Index mapping:**

| Token | Index | Purpose |
|---|---|---|
| `<PAD>` | 0 | Padding |
| `<SOS>` | 1 | Start of sequence |
| `<EOS>` | 2 | End of sequence |
| `0`-`9` | 3-12 | Digits |
| `x` | 13 | Variable |
| `+` | 14 | Addition |
| `-` | 15 | Subtraction / negative sign |
| `*` | 16 | Multiplication |
| `/` | 17 | Division |
| `=` | 18 | Equals |
| `.` | 19 | Decimal point |
| ` ` | 20 | Space |

*(Exact indices will be confirmed during implementation)*

### Dataset (Step 04)

**Encoding an equation-solution pair:**

```
Equation: "2x + 3 = 7"
Solution: "x = 2"

Encoder input:  [2, x, +, 3, =, 7, <EOS>, <PAD>, <PAD>]  ← padded to max length
Decoder input:  [<SOS>, x, =, 2]                           ← starts with <SOS>
Decoder target: [x, =, 2, <EOS>]                           ← shifted by 1, ends with <EOS>
```

---

## Model Architecture

*(Updated as steps are implemented)*

### Encoder (Step 05)

```
Encoder(
  embedding:  Embedding(~23, 64)       ← character → 64-dim vector
  gru:        GRU(64, 128, 2 layers, bidirectional=True)
  fc_hidden:  Linear(256, 256)         ← bridge: encoder hidden → decoder hidden
)

Parameter estimate:
  Embedding:    23 × 64     =   1,472
  GRU Layer 1:  (forward + backward) ≈ 148,000
  GRU Layer 2:  (forward + backward) ≈ 394,000
  Bridge FC:    256 × 256   =  65,536
  Total:        ~609,000 parameters
```

### Decoder (Step 06)

```
Decoder(
  embedding:    Embedding(~23, 64)     ← character → 64-dim vector
  gru:          GRU(64, 256, 2 layers)
  output_layer: Linear(256, ~23)       ← hidden → character scores
)

Parameter estimate:
  Embedding:    23 × 64     =   1,472
  GRU Layer 1:              ≈ 247,000
  GRU Layer 2:              ≈ 394,000
  Output:       256 × 23    =   5,888
  Total:        ~648,000 parameters
```

### Seq2Seq Combined (Step 07)

```
Seq2Seq(
  encoder: Encoder(...)    ← ~609K params
  decoder: Decoder(...)    ← ~648K params
)
Total: ~1.26M parameters

(~5× larger than the text generator's 248K, because the task is harder)
```

---

## Training Pipeline

*(Updated when Step 08 is implemented)*

### Configuration
```
Epochs:         30-50
Batch size:     64
Learning rate:  0.001
Optimizer:      Adam
Loss function:  CrossEntropyLoss(ignore_index=PAD_IDX)
Teacher forcing: 1.0 → 0.5 (scheduled decrease)
Gradient clip:  1.0 (prevents exploding gradients)
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

### Expected Training Curve

```
Epoch  1: Loss ~2.8, Accuracy ~5%    (random guessing)
Epoch  5: Loss ~1.5, Accuracy ~30%   (learning structure "x = ...")
Epoch 10: Loss ~0.8, Accuracy ~60%   (getting digits right sometimes)
Epoch 20: Loss ~0.3, Accuracy ~85%   (most equations correct)
Epoch 30: Loss ~0.1, Accuracy ~93%   (fine-tuning edge cases)
```

---

## Inference Pipeline

*(Updated when Step 09 is implemented)*

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

---

*This document is updated with each step. Last updated: Step 01 (skeleton).*
