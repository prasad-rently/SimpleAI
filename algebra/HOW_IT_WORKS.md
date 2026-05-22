# How It Works: Step-by-Step Inside the Algebra Solver

What happens when you type **"2x + 3 = 7"** into the model? This document traces every internal step — from raw equation to predicted solution — with real values from the trained model.

This is a completely different architecture from the text generator. The text generator used one RNN to predict the next character. The algebra solver uses **two networks** (encoder + decoder) because the input and output have different lengths and structures.

---

## The Big Picture

```
You type "2x + 3 = 7"
     │
     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  STEP 1: Tokenization     "2","x"," ","+",... → [9,18,3,4,3,10,3,17,3,14,2]│
│                                                                              │
│  STEP 2: Embedding        [9,18,3,4,...] → 11 vectors of 64 numbers each    │
│                                                                              │
│  STEP 3: Encoder (GRU)    Reads equation BOTH directions → context vector   │
│           Bidirectional    "2x+3=7" forward + "7=3+x2" backward             │
│                            Result: 256-dim summary of the equation          │
│                                                                              │
│  ─ ─ ─ ─ ─ ─ ─ ─ context vector passes from encoder to decoder ─ ─ ─ ─ ─  │
│                                                                              │
│  STEP 4: Decoder (GRU)    Starts with <SOS> token + context vector          │
│           One direction    Generates one character at a time:               │
│                            <SOS> → "x" → " " → "=" → " " → "2" → <EOS>   │
│                                                                              │
│  STEP 5: Decode Tokens    [18, 3, 17, 3, 9, 2] → "x = 2"                  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
Output: "x = 2"
```

### How this differs from the text generator

| Feature | Text Generator | Algebra Solver |
|---|---|---|
| Networks | 1 RNN | 2 networks (encoder + decoder) |
| Direction | Left-to-right only | Encoder: bidirectional, Decoder: left-to-right |
| RNN type | Simple RNN | GRU (Gated Recurrent Unit) |
| Input/Output | Same length | Different lengths |
| Special tokens | None | `<PAD>`, `<SOS>`, `<EOS>` |
| Training trick | None | Teacher forcing |
| Task | Predict next character | Map equation → solution |

---

## Step 1: Tokenization (Characters → Numbers)

Neural networks only understand numbers. The first step converts each character into a unique integer using a **vocabulary** of 19 tokens.

### The Vocabulary

The model uses only **16 unique characters** from the equations, plus **3 special tokens**:

```
Index │ Token    │ Purpose
──────┼──────────┼─────────────────────────────────
   0  │ <PAD>    │ Padding — fills shorter sequences to equal length
   1  │ <SOS>    │ Start Of Sequence — tells decoder to begin
   2  │ <EOS>    │ End Of Sequence — signals "I'm done"
   3  │ " "      │ Space
   4  │ "+"      │ Plus sign
   5  │ "-"      │ Minus sign
   6  │ "/"      │ Division sign
   7  │ "0"      │ Digit zero
   8  │ "1"      │ Digit one
   9  │ "2"      │ Digit two
  10  │ "3"      │ Digit three
  11  │ "4"      │ Digit four
  12  │ "5"      │ Digit five
  13  │ "6"      │ Digit six
  14  │ "7"      │ Digit seven
  15  │ "8"      │ Digit eight
  16  │ "9"      │ Digit nine
  17  │ "="      │ Equals sign
  18  │ "x"      │ Variable x
```

Compare this to the text generator's 48 tokens (all lowercase + uppercase letters). Here we only need 19 because math uses a small character set.

### Encoding "2x + 3 = 7"

Each character is replaced by its index number, then `<EOS>` is appended:

```
"2"  →  9
"x"  →  18
" "  →  3
"+"  →  4
" "  →  3
"3"  →  10
" "  →  3
"="  →  17
" "  →  3
"7"  →  14
<EOS> → 2

"2x + 3 = 7"  →  [9, 18, 3, 4, 3, 10, 3, 17, 3, 14, 2]
                   2   x  _  +  _   3  _   =  _   7  END
```

### Why `<EOS>`?

The `<EOS>` (End Of Sequence) token tells the model "this is where the equation ends." Without it, the model wouldn't know where the meaningful content stops and padding begins.

### Encoding the solution "x = 2"

The solution is also encoded, but with TWO versions for training:

```
Decoder INPUT:  [1, 18, 3, 17, 3, 9]     ← starts with <SOS>
                 ^   x  _   =  _  2
                 │
                 └── <SOS> = "start generating now"

Decoder TARGET: [18, 3, 17, 3, 9, 2]     ← ends with <EOS>
                  x  _   =  _  2  ^
                                  │
                                  └── <EOS> = "stop generating"
```

Notice the **offset by 1**: the input starts with `<SOS>` and the target ends with `<EOS>`. At each step, the decoder receives the current token and must predict the next one:

```
Step │ Decoder Input │ Should Predict
─────┼───────────────┼───────────────
  0  │ <SOS>         │ x
  1  │ x             │ (space)
  2  │ (space)       │ =
  3  │ =             │ (space)
  4  │ (space)       │ 2
  5  │ 2             │ <EOS>
```

### Padding

Equations have different lengths. In a batch of 64 equations, the shorter ones are padded with `<PAD>` (index 0) to match the longest:

```
Equation 1: "3x = 12"      → [10,18,3,17,3,8,9,2, 0, 0, 0, 0, 0]
Equation 2: "2x + 3 = 7"   → [ 9,18,3,4,3,10,3,17,3,14, 2, 0, 0]
Equation 3: "5x - 10 = 15" → [12,18,3,5,3,8,7,3,17,3,8,12, 2]
                                                    padded to length 13 ──┘
```

The `<PAD>` token has a special property: its embedding is always zeros, and the loss function ignores it. This way, padding doesn't affect training.

---

## Step 2: Embedding (Numbers → Vectors)

A single number like `9` (for "2") doesn't capture any meaning. The **embedding layer** converts each index into a **64-dimensional vector** — a list of 64 learned numbers.

### How it works

The embedding layer is a lookup table: a **19 x 64 matrix** (19 tokens, 64 numbers each). To embed index `9` ("2"), grab row 9:

```
Embedding weight matrix: 19 rows x 64 columns

Index 9 ("2") → row 9 → [-0.211, -0.140, 0.089, 0.083, 0.006, -0.077,
                           1.630, -0.311, 0.683, -0.411, ...]
                          (showing first 10 of 64 dimensions)
```

### Real values for "2x + 3 = 7"

```
"2" (idx  9) → [-0.211, -0.140,  0.089,  0.083,  0.006, -0.077,  1.630, -0.311, ...]
"x" (idx 18) → [ 0.xxx,  0.xxx,  0.xxx,  0.xxx,  0.xxx,  0.xxx,  0.xxx,  0.xxx, ...]
" " (idx  3) → [ 0.xxx,  0.xxx,  0.xxx,  0.xxx,  0.xxx,  0.xxx,  0.xxx,  0.xxx, ...]
... (11 characters total, each becoming a 64-number vector)
```

### Shape transformation

```
Input:   (1, 25)       →  25 integers (11 real + 14 padding)
Output:  (1, 25, 64)   →  25 vectors of 64 numbers each

Each integer got "expanded" into a 64-number description.
The <PAD> tokens (index 0) always map to all-zeros vectors.
```

### Why 64 dimensions?

Compare to the text generator's 128 dimensions for 48 characters. Here we use 64 for 19 tokens — fewer tokens need fewer dimensions. These numbers are **learned during training**. Similar tokens end up with similar vectors:
- Digits "0"-"9" develop similar patterns
- Operators "+", "-" develop their own patterns
- "x" gets a unique representation

**Parameter count:** 19 tokens x 64 dimensions = **1,216 learnable numbers** (per embedding layer)

---

## Step 3: Encoder — Reading the Equation

The encoder is a **bidirectional GRU** that reads the embedded equation and compresses it into a fixed-size **context vector**. This is the key difference from the text generator — instead of producing output at every position, the encoder produces a single summary vector.

### What is a GRU?

The text generator used a simple RNN, which suffers from the "forgetting problem" — by the end of a long sequence, it forgets what was at the start. The **GRU (Gated Recurrent Unit)** fixes this with two gates:

```
┌─────────────────────────────────────────────────┐
│                    GRU Cell                      │
│                                                  │
│  Input: current character + previous memory      │
│                                                  │
│  ┌──────────────┐                                │
│  │ Reset Gate   │  "How much old memory to use?" │
│  │ (0.0 to 1.0) │  0.0 = forget everything      │
│  └──────────────┘  1.0 = remember everything     │
│                                                  │
│  ┌──────────────┐                                │
│  │ Update Gate  │  "How much new info to add?"   │
│  │ (0.0 to 1.0) │  0.0 = keep old memory        │
│  └──────────────┘  1.0 = replace with new        │
│                                                  │
│  Output: updated memory (256 numbers)            │
└─────────────────────────────────────────────────┘
```

The gates are **learned during training** — the model discovers what to remember and what to forget for each type of character.

### Why Bidirectional?

A regular RNN reads left-to-right only. When it reaches "7" at the end of "2x + 3 = 7", it has context from the left but not the right. A **bidirectional** GRU reads in BOTH directions:

```
Forward GRU:   "2" → "x" → " " → "+" → " " → "3" → " " → "=" → " " → "7"
                ──────────────────────────────────────────────────────────────>
                At "=", knows "2x + 3" came before

Backward GRU:  "2" ← "x" ← " " ← "+" ← " " ← "3" ← " " ← "=" ← " " ← "7"
                <──────────────────────────────────────────────────────────────
                At "=", knows "7" comes after
```

At every position, the model sees context from **both sides**. This is critical for math — the "=" sign needs to know what's on both its left AND right to understand the equation.

### Real encoder processing

```
Input: "2x + 3 = 7" (padded to length 25)

┌──────────┐                              ┌──────────┐
│ Forward  │ "2"→"x"→" "→"+"→...→"7"     │ Backward │ "7"→...→"+"→" "→"x"→"2"
│ GRU      │ ─────────────────────────>   │ GRU      │ <─────────────────────────
│ 128-dim  │                              │ 128-dim  │
└──────────┘                              └──────────┘
      │                                         │
      └─────────── concatenate ─────────────────┘
                        │
                   256-dim output
              (128 forward + 128 backward)
```

### The context vector

After processing all characters, the encoder produces its final hidden state — the **context vector**. This is a 256-dimensional vector that summarizes the entire equation:

```
Context vector shape: (2 layers, 1 batch, 256 dimensions)

Layer 0, first 10 values:
  [0.974, -0.897, 0.975, 1.000, -0.669, -0.996, 1.000, 1.000, 0.543, 1.000]
```

These 256 numbers are all the decoder gets. The entire equation "2x + 3 = 7" has been compressed into 256 numbers that (hopefully) capture:
- The coefficient is 2
- The constant is 3
- The right side is 7
- The operation is addition
- The answer should be x = (7-3)/2 = 2

The model doesn't literally compute this formula — but it has learned to encode the information needed to produce the right answer.

### Hidden state bridge

There's a subtlety: the bidirectional GRU produces hidden states in a specific shape, but the decoder (which is unidirectional) expects a different shape. A **linear layer + tanh** transforms between them:

```
Encoder hidden: (4, batch, 128)    ← 2 layers x 2 directions x 128 dims
       ↓  reshape
                  (2, batch, 256)  ← 2 layers x 256 dims (128 fwd + 128 bwd)
       ↓  Linear(256, 256) + tanh
Decoder hidden: (2, batch, 256)    ← ready for unidirectional decoder
```

The `tanh` squashes values to [-1, 1], preventing extreme values from destabilizing the decoder.

---

## Step 4: Decoder — Writing the Solution

The decoder receives the context vector and generates the solution **one character at a time**. Unlike the encoder, the decoder is **unidirectional** (left-to-right only) because it can't peek at characters it hasn't generated yet.

### Step-by-step generation (with the trained model)

Starting with the context vector from "2x + 3 = 7":

```
Step 0: Input = <SOS> (start token)
        ┌──────────────────────────────────────────────────────────────┐
        │  GRU processes <SOS> + context vector                       │
        │  Output layer produces 19 scores (one per token)            │
        │                                                              │
        │  Token probabilities:                                        │
        │    "x"   : 100.0%  ██████████████████████████████████████████│
        │    "1"   :   0.0%                                            │
        │    <EOS> :   0.0%                                            │
        │    "2"   :   0.0%                                            │
        │                                                              │
        │  Predicted: "x"  (index 18)                                  │
        └──────────────────────────────────────────────────────────────┘

Step 1: Input = "x" (previous prediction)
        ┌──────────────────────────────────────────────────────────────┐
        │  Token probabilities:                                        │
        │    " "   : 100.0%  ██████████████████████████████████████████│
        │    "2"   :   0.0%                                            │
        │    "1"   :   0.0%                                            │
        │                                                              │
        │  Predicted: " "  (index 3)                                   │
        └──────────────────────────────────────────────────────────────┘

Step 2: Input = " " → Predicted: "="  (100.0%)
Step 3: Input = "=" → Predicted: " "  (100.0%)

Step 4: Input = " " (this is the critical step — predicting the actual digit)
        ┌──────────────────────────────────────────────────────────────┐
        │  Token probabilities:                                        │
        │    "2"   : 100.0%  ██████████████████████████████████████████│
        │    "3"   :   0.0%                                            │
        │    "1"   :   0.0%                                            │
        │    "4"   :   0.0%                                            │
        │                                                              │
        │  Predicted: "2"  (index 9)                                   │
        └──────────────────────────────────────────────────────────────┘

Step 5: Input = "2"
        ┌──────────────────────────────────────────────────────────────┐
        │  Token probabilities:                                        │
        │    <EOS> : 100.0%  ██████████████████████████████████████████│
        │    "0"   :   0.0%                                            │
        │                                                              │
        │  Predicted: <EOS>  → STOP generating                        │
        └──────────────────────────────────────────────────────────────┘
```

### Result: `[18, 3, 17, 3, 9, 2]` → decode → **"x = 2"**

The model is extremely confident (near 100% at every step) because this is a simple equation. For harder equations with larger numbers, the probabilities become more spread out and the model sometimes picks the wrong digit.

### Why is Step 4 the hardest?

Steps 0-3 are "easy" — the model learns that every answer starts with "x = ". The real challenge is Step 4 (and beyond for multi-digit answers): predicting the actual numeric value. This requires the model to have encoded the equation's mathematical relationships into the context vector.

---

## Step 5: Decoding (Numbers → Text)

The decoder's output `[18, 3, 17, 3, 9, 2]` is converted back to characters:

```
18 → "x"
 3 → " "
17 → "="
 3 → " "
 9 → "2"
 2 → <EOS> ← stop here, don't include in output

Result: "x = 2"
```

The `decode_until_eos()` function reads tokens until it hits `<EOS>` (index 2), then stops and returns the decoded string.

---

## How Training Works

The model starts with **random weights** — it has no idea how to solve equations. Training teaches it by showing 40,000 examples and adjusting weights to reduce errors.

### Before training (random weights)

```
Input:  "2x + 3 = 7"
Output: "1 6 / + + /"    ← complete gibberish

Loss: 2.94  (very high — model is maximally wrong)
```

The untrained model assigns near-equal probability to all 19 tokens at each step. It has no concept of equations or solutions.

### The training loop

For each of the 40,000 training equations, the model:

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  1. FORWARD PASS                                                 │
│     "2x + 3 = 7" → Encoder → context → Decoder → predictions   │
│                                                                  │
│  2. COMPARE (Loss)                                               │
│     Predicted: [0.1, 0.1, 0.05, ..., 0.2, ...]  (19 scores)    │
│     Correct:   "x" (index 18)                                    │
│     Loss = how wrong the prediction is                           │
│     Higher score on wrong token = higher loss                    │
│                                                                  │
│  3. BACKWARD PASS (Backpropagation)                              │
│     Compute gradients: "which weights caused the error?"         │
│     Gradients flow backward through decoder AND encoder          │
│                                                                  │
│  4. GRADIENT CLIPPING                                            │
│     Cap gradients at 1.0 to prevent wild weight updates          │
│     (RNNs are prone to "exploding gradients")                    │
│                                                                  │
│  5. OPTIMIZER STEP                                               │
│     Adjust all 1,160,595 weights slightly to reduce the loss    │
│     Small nudge in the direction that makes the answer better    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

Repeat 50 times over all 40,000 equations = 2,000,000 weight updates
```

### Teacher forcing — training wheels for the decoder

During training, the decoder needs to learn to produce each character. But there's a chicken-and-egg problem: if it predicts Step 0 wrong, Step 1 gets wrong input, Step 2 gets worse input, and so on. The entire sequence collapses.

**Teacher forcing** solves this by feeding the **correct** previous character regardless of what the model predicted:

```
WITHOUT teacher forcing (bad for early training):
  Step 0: input=<SOS> → predicts "9" (wrong!)
  Step 1: input="9"   → predicts "x" (wrong context, builds on error)
  Step 2: input="x"   → predicts "+" (completely lost)
  Result: "9x+..." — garbage that the model can't learn from

WITH teacher forcing (good for early training):
  Step 0: input=<SOS> → predicts "9" (wrong, but we compute loss)
  Step 1: input="x"   → predicts " " (correct input from teacher!)
  Step 2: input=" "   → predicts "=" (learning from correct context)
  Step 3: input="="   → predicts " " (getting better!)
  Step 4: input=" "   → predicts "3" (almost right — target is "2")
  Step 5: input="2"   → predicts <EOS> (correct!)
  Result: model learns at EVERY position, not just position 0
```

### The teacher forcing schedule

We start with full teacher forcing and gradually reduce it:

```
Epoch  1: TF = 1.00  → always give correct answer (training wheels ON)
Epoch 10: TF = 0.86  → 86% correct, 14% own predictions
Epoch 20: TF = 0.73  → mixing correct and own predictions
Epoch 30: TF = 0.59  → more independence
Epoch 40: TF = 0.44  → mostly on its own
Epoch 50: TF = 0.30  → almost fully independent (training wheels almost OFF)
```

This gradual transition is critical. If we remove teacher forcing too early, the model collapses. If we never remove it, the model never learns to handle its own mistakes.

### Loss computation — ignoring padding

The loss function (`CrossEntropyLoss`) measures how wrong each prediction is. But we tell it to **ignore padding tokens** with `ignore_index=PAD_IDX`:

```
Target:     [x,   =,  2,  <EOS>, <PAD>, <PAD>]
Prediction: [x,   =,  3,  <EOS>, <PAD>, <PAD>]
Loss:        ✓    ✓   ✗    ✓     skip   skip
                       ↑
             penalty for predicting "3" instead of "2"
```

Without `ignore_index`, the model would be rewarded for predicting `<PAD>` in padding positions — which teaches it nothing useful.

### Gradient clipping — preventing explosions

RNNs (including GRUs) process sequences step by step. During backpropagation, gradients multiply at each step. Over a long sequence, gradients can become astronomically large — called **exploding gradients**:

```
Gradient at step 10 = gradient × weight × weight × ... × weight  (10 times)

If each multiplication is 1.1x: 1.1^10 = 2.6   (manageable)
If each multiplication is 2.0x: 2.0^10 = 1024   (EXPLOSION!)
```

**Gradient clipping** caps the total gradient at 1.0:

```
Before clipping: gradients = [1000.0, -500.0, 800.0, ...]
                 total norm = 1,389
After clipping:  gradients = [0.72, -0.36, 0.58, ...]
                 total norm = 1.0  (scaled down proportionally)
```

This prevents wild weight updates that would destabilize training.

### Learning rate scheduler

The learning rate controls how big each weight update is:

```
Epoch  1-14: LR = 0.002    (big steps — learn fast)
Epoch 15-29: LR = 0.001    (medium steps — refine)
Epoch 30-44: LR = 0.0005   (small steps — fine-tune)
Epoch 45-50: LR = 0.00025  (tiny steps — polish)
```

This is like a sculptor: start with a hammer for the rough shape, then switch to finer tools for details.

### Training progress

```
Epoch 10: Loss 0.1229  Accuracy 68.9%  ← learns "x = " format, some digits
Epoch 20: Loss 0.0538  Accuracy 85.4%  ← gets most equations right
Epoch 30: Loss 0.0484  Accuracy 87.7%  ← plateau — LR drop helps
Epoch 40: Loss 0.0225  Accuracy 92.1%  ← breaks through 90% target!
Epoch 50: Loss 0.0098  Accuracy 94.6%  ← final accuracy
```

---

## What the Model Actually Learned

The model doesn't "know" algebra. It has no concept of addition, subtraction, or solving for x. It learned **statistical patterns** from 40,000 training examples:

### Pattern 1: Output format
Every answer starts with "x = ". The model learns this in the first few epochs because it's 100% consistent across all 50,000 examples. By epoch 3, the model always generates "x = " as the first 4 tokens.

### Pattern 2: Single-digit answers
For "3x = 12", the model learned that when the coefficient divides evenly into the constant, the answer is a single digit. It doesn't compute 12/3 — it recognizes the pattern from thousands of similar examples in training.

### Pattern 3: Negative answers
For "2x + 10 = 4", the answer is negative. The model learned that when the constant on the left is larger than the right side (for positive coefficients), the answer is negative. It outputs "-" before the digit.

### Pattern 4: Multi-digit answers
For "-15x + 20 = -280", the model must output "x = 20" — two digits. It learned to predict each digit left-to-right: first "2", then "0", then `<EOS>`.

### The fundamental limitation

**The model is a pattern matcher, not a calculator.** When it sees an equation it has seen similar versions of during training, it does well. When it sees a pattern it has **never** encountered, it fails — sometimes confidently.

---

## Where the Model Fails — Error Analysis

### Overall error statistics

Out of 10,000 test equations, the model gets **542 wrong** (5.4% error rate). Here's what those errors look like:

```
Error Type          Count    % of Errors
─────────────────────────────────────────
Off by 1              414       76%       ← most common by far
Off by 2-5            101       19%
Off by large amount    27        5%
Wrong sign              1       <1%
Garbled output          0        0%
```

**Key insight: 76% of all errors are off by exactly 1.** The model almost always gets the right ballpark — it just sometimes picks the neighboring digit. There are zero garbled outputs — the model always produces valid "x = N" format.

### Error examples by equation type

```
Type 1 (ax = b) — 99.4% accurate, errors are rare:
  3x = 60           predicted: x = 19    expected: x = 20    (off by 1)

Type 2 (ax + b = c) — 97.8% accurate:
  x + 8 = -14       predicted: x = -23   expected: x = -22   (off by 1)
  3x + 15 = -93     predicted: x = -35   expected: x = -36   (off by 1)

Type 6 (x / a = b) — 65.9% accurate, worst type:
  x / -12 = -20     predicted: x = 260   expected: x = 240   (off by 20!)
  x / 4 = 16        predicted: x = 68    expected: x = 64    (off by 4)

Type 7 (ax + b = cx + d) — 87.2% accurate:
  2x + 19 = -4x + 127   predicted: x = 19   expected: x = 18   (off by 1)
```

### Why off-by-one errors happen

The model predicts each digit independently, left to right. For "x = 20", it must:
1. Predict "2" correctly (tens digit)
2. Predict "0" correctly (ones digit)

When the model gets the tens digit right but the ones digit wrong by 1, the answer is off by 1. For example:

```
Target: x = 20
Model predicts:
  "2" → correct (90% confident)
  "0" → wrong, picks "1" instead (48% vs 45% confidence)
Result: x = 21 (off by 1)
```

The ones digit is hardest because small changes in the equation can flip it between adjacent values.

### Why Type 6 (division) is the worst

Division equations like `x / 4 = 3` require the answer `x = 12` (multiply 4 x 3). Problems:

1. **Few training examples**: only 1,602 out of 50,000 (3.2%) are division
2. **Large answers**: x = a x b can reach 280+, producing 3-digit numbers
3. **Digit prediction compounds errors**: for "x = 240", the model must get 3 digits right in sequence — each digit has maybe 70% accuracy, so overall: 0.7 x 0.7 x 0.7 = 34% chance of getting all three right

---

## A Real Failure: "5x + 5 = 10x - 20"

This equation exposes a critical limitation. The correct answer is **x = 5**, but the model predicts **x = -4** with 100% confidence at every step:

```
Equation: 5x + 5 = 10x - 20
Correct:  x = 5

Model step-by-step:
  Step 0: <SOS> → "x"   (100% confident)
  Step 1: "x"   → " "   (100% confident)
  Step 2: " "   → "="   (100% confident)
  Step 3: "="   → " "   (100% confident)
  Step 4: " "   → "-"   (100% confident)  ← WRONG! Should be no "-"
  Step 5: "-"   → "4"   (100% confident)  ← WRONG! Should be "5"
  Step 6: "4"   → <EOS> (100% confident)

Result: "x = -4"  ← completely wrong, but model is 100% sure
```

### Why does this happen?

The model is **100% confident** and **100% wrong**. This is the most dangerous type of error. Here's why:

**The training data never contains this equation format.**

All 10,763 Type 7 equations in the training data are formatted as:
```
ax + b = cx + d      ← always uses "+" on the right side
```

When `d` is negative, the generator formats it as:
```
3x + 2 = x + -8     ← writes "+ -8", NOT "- 8"
```

But the user typed:
```
5x + 5 = 10x - 20   ← uses "-" on the right side
```

The model has **never seen a minus sign on the right side** of a Type 7 equation. This is called an **out-of-distribution** input — it falls outside the patterns the model was trained on.

### The root cause in code

In `generate_data.py`, the Type 7 generator always uses this format:

```python
equation = f"{ax} + {b} = {cx} + {format_number(d)}"
#                                  ^^^^^^^^^^^^^^^^^
#                                  always "+" even when d is negative
#                                  produces: "x + -8" not "x - 8"
```

So the model learned:
- `"+ -20"` → means negative 20 on the right (from training data)
- `"- 20"` → model has NO IDEA what this means (never seen it)

When the model encounters `"- 20"`, it doesn't understand the subtraction and produces a wrong answer with full confidence — because within its training distribution, it has no reason to be uncertain.

### This is a pattern recognition problem, not a math problem

A human looks at `5x + 5 = 10x - 20` and `5x + 5 = 10x + -20` and sees the same equation. The model sees two **completely different character sequences** and treats them as different problems. It has no understanding of mathematics — only character patterns.

---

## Why the Model Is Confidently Wrong

This is one of the most important lessons in machine learning:

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   High confidence does NOT mean the answer is correct.          │
│                                                                 │
│   The model can be 100% sure and 100% wrong when the input     │
│   is outside its training distribution.                         │
│                                                                 │
│   This is why real AI systems need:                             │
│   1. Diverse training data covering edge cases                  │
│   2. Verification systems (like substitution checking)          │
│   3. Uncertainty estimation (knowing when you don't know)       │
│   4. Human oversight for critical decisions                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

The model's `:verify` mode catches this error immediately:
```
>> :verify
>> 5x + 5 = 10x - 20
   x = -4
   Check: 5x + 5 → -15,  10x - 20 → -60  ✗   ← caught!
```

This is why verification systems matter — the model can't tell you when it's wrong, but math can.

---

## Comparing Three Equations

### Easy: "3x = 12" (Type 1, 99.4% accuracy)

```
Encoder reads:  3, x, =, 1, 2
Context vector: [0.99, -0.87, 0.95, ...]  (256 numbers summarizing "3x=12")

Decoder generates:
  <SOS> → "x" (100%)  → " " (100%)  → "=" (100%)  → " " (100%)
       → "4" (99.8%)  → <EOS> (100%)

Result: "x = 4" ✓
Why it works: Simple pattern, single digit, seen thousands of similar examples.
```

### Medium: "x / 12 = -9" (Type 6, 65.9% accuracy)

```
Encoder reads:  x, /, 1, 2, =, -, 9
Context vector: [0.78, -0.45, 0.92, ...]  (must encode "multiply 12 by -9")

Decoder generates:
  <SOS> → "x" (100%)  → " " (100%)  → "=" (100%)  → " " (100%)
       → "-" (99%)     → "1" (70%)   → "0" (65%)   → "8" (55%)
       → <EOS> (90%)

Result: "x = -108" ✓ (but only 55% confident on the "8")
Why it's hard: 3-digit answer, each digit has decreasing confidence.
```

### Failure: "5x + 5 = 10x - 20" (out of distribution)

```
Encoder reads:  5, x, +, 5, =, 1, 0, x, -, 2, 0
Context vector: [?, ?, ?, ...]  (confused by never-seen "- 20" pattern)

Decoder generates:
  <SOS> → "x" (100%)  → " " (100%)  → "=" (100%)  → " " (100%)
       → "-" (100%)    → "4" (100%)   → <EOS> (100%)

Result: "x = -4" ✗ (100% confident, 100% wrong!)
Why it fails: "10x - 20" format never appeared in training data.
              Training data only has "10x + -20" format.
```

---

## How to Fix These Inaccuracies

### Fix 1: Diversify the training data format

The biggest fix is generating equations in BOTH formats:

```python
# CURRENT (always uses "+"):
equation = f"{ax} + {b} = {cx} + {format_number(d)}"
# Produces: "3x + 2 = x + -8"

# FIXED (use "-" when d is negative):
if d >= 0:
    equation = f"{ax} + {b} = {cx} + {d}"
else:
    equation = f"{ax} + {b} = {cx} - {abs(d)}"
# Produces: "3x + 2 = x - 8"
```

Apply this to ALL equation types — currently Types 2-5 also only use one format. Adding both `"+ -8"` and `"- 8"` variants to the training data would teach the model that both mean the same thing.

### Fix 2: Add more Type 6 (division) examples

Type 6 has only 3.2% of training data but the hardest answers (large multi-digit numbers). Solutions:

```
Current:  1,602 Type 6 equations  (3.2%)  → 65.9% accuracy
Better:   5,000 Type 6 equations  (10%)   → likely ~85% accuracy
Best:    10,000 Type 6 equations  (20%)   → likely ~90% accuracy
```

Also limit divisor range to keep answers smaller (e.g., `x / 4 = 3` → x = 12 is easier than `x / 19 = 18` → x = 342).

### Fix 3: Train for more epochs with lower learning rate

The model reached 94.6% at epoch 50. Training longer could push accuracy higher:

```
Current:   50 epochs  → 94.6%
Extended: 100 epochs  → likely ~96-97%
Extended: 200 epochs  → likely ~97-98% (diminishing returns)
```

Combine with a cosine learning rate schedule instead of StepLR for smoother convergence.

### Fix 4: Data augmentation — rewrite equations in multiple formats

For each equation, generate multiple equivalent representations:

```
Original:  "2x + 3 = 7"
Augmented: "3 + 2x = 7"     (swap terms on left)
           "7 = 2x + 3"     (swap sides)
           "7 = 3 + 2x"     (swap sides + swap terms)
```

This teaches the model that format doesn't change the answer — making it robust to variations like `"10x - 20"` vs `"10x + -20"`.

### Fix 5: Add an attention mechanism

Currently the decoder only sees the context vector — a 256-number summary of the entire equation. With **attention**, the decoder can "look back" at specific parts of the equation while generating each digit:

```
Without attention:
  Decoder sees: [0.97, -0.89, ...]  (compressed summary — lost details)

With attention:
  Generating "2": decoder looks at the "7" and "3" positions in the encoder
  Generating "-": decoder looks at the coefficient signs
```

Attention is what made seq2seq models dramatically better and eventually led to the Transformer architecture used in ChatGPT and Claude.

### Fix 6: Use a Transformer instead of GRU

Modern models use Transformers, which:
- Process all positions in parallel (faster)
- Have built-in attention at every layer (better accuracy)
- Scale better to larger datasets and harder problems

For this 50K-equation task, a small Transformer would likely achieve 98%+ accuracy.

---

## Verification by Substitution

After the model predicts "x = 2" for "2x + 3 = 7", we can verify mathematically:

```
Equation: 2x + 3 = 7
Model says: x = 2

Substitute x = 2:
  Left side:  2(2) + 3 = 4 + 3 = 7
  Right side: 7
  
  7 == 7  → ✓ CORRECT
```

The interactive solver's `:verify` mode does this automatically:

```
>> :verify
   Verification mode: ON

>> 2x + 3 = 7
   x = 2
   Check: 2x + 3 → 7,  7 → 7  ✓

>> 5x + 5 = 10x - 20
   x = -4
   Check: 5x + 5 → -15,  10x - 20 → -60  ✗   ← WRONG! Caught by verification
```

In the evaluation, **100% of the model's correct predictions pass substitution verification** — when the model gets the right answer, it's genuinely right. But verification is most valuable for catching the 5.4% of answers that are wrong.

**Lesson: Never trust a model's confidence. Always verify with an independent check when possible.**

---

## The Complete Data Flow (Summary)

```
INPUT: "2x + 3 = 7"

1. TOKENIZE:     "2x + 3 = 7"  →  [9, 18, 3, 4, 3, 10, 3, 17, 3, 14, 2]
2. PAD:          [9, 18, 3, 4, 3, 10, 3, 17, 3, 14, 2, 0, 0, ..., 0]  (length 25)
3. EMBED:        (1, 25) → (1, 25, 64)  — each token becomes 64-dim vector
4. ENCODE:       Bidirectional GRU reads all 25 positions
                 Output: context vector (2, 1, 256)
5. DECODE:       GRU generates tokens one at a time:
                 <SOS> → x → (space) → = → (space) → 2 → <EOS>
6. DECODE TEXT:  [18, 3, 17, 3, 9, 2] → "x = 2"

OUTPUT: "x = 2"

Total parameters: 1,160,595
Training time: ~15 minutes
Final accuracy: 94.6% (9,458 / 10,000 correct)
Error rate: 5.4% (542 wrong — 76% are off by just 1)
```

---

## Key Takeaways

1. **The model is a pattern matcher, not a mathematician.** It learns from examples, not from rules. When it sees familiar patterns, it does great. When it sees unfamiliar patterns, it fails.

2. **High confidence doesn't mean correctness.** The model can be 100% confident and completely wrong on out-of-distribution inputs. Always verify.

3. **Training data quality is everything.** The model can only learn what's in the data. Missing equation formats (like `"- 20"` vs `"+ -20"`) create blind spots.

4. **76% of errors are off by just 1.** The model is remarkably close even when wrong — it understands the structure, just sometimes picks the adjacent digit.

5. **Verification catches errors that confidence doesn't.** The `:verify` mode uses substitution to mathematically verify answers — this catches all errors regardless of model confidence.

6. **More data and diverse formats are the best fixes.** Most of the model's weaknesses come from limited training data variety, not from model architecture.

---

## Running It Yourself

```bash
# Setup
cd SimpleAI
source venv/bin/activate

# Solve equations interactively
PYTHONPATH=algebra/src python3 algebra/src/interactive.py

# Run all 76 tests
PYTHONPATH=algebra/src pytest algebra/tests/test_e2e.py -v

# See the training loss plot
open algebra/outputs/loss_plot.png
```

---

*This document traces the internals of the algebra solver with real values from the trained model. For the text generator's internals, see [../HOW_IT_WORKS.md](../HOW_IT_WORKS.md).*
