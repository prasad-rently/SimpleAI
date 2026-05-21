# How It Works: Step-by-Step Inside SimpleAI

What happens when you type **"The"** or **"Success"** into the model? This document traces every internal step — from raw characters to generated text — with real values from the trained model.

---

## The Big Picture

```
You type "The"
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Tokenization         "T", "h", "e"  →  [19, 29, 26]  │
│  STEP 2: Embedding            [19, 29, 26]  →  3 vectors of    │
│                                                  128 numbers    │
│  STEP 3: RNN Processing       3 × 128-dim  →  3 × 256-dim     │
│                                (reads left to right, builds     │
│                                 context at each position)       │
│  STEP 4: Output Layer         256-dim  →  48 scores            │
│                                (one score per character)        │
│  STEP 5: Softmax              48 scores  →  48 probabilities   │
│                                (scores become percentages)      │
│  STEP 6: Pick Next Char       probabilities  →  " " (space)    │
│                                                                 │
│  STEP 7: Loop                 Feed " " back in, repeat         │
│                                "The " → "The o" → "The on"...  │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
Output: "The only way to do great work is to love what you do today."
```

---

## Step 1: Tokenization (Characters → Numbers)

Neural networks can only process numbers, not letters. So the first step converts each character into a unique integer using a **vocabulary** — a lookup table built from the training data.

### The Vocabulary

Our model learned from 99 inspirational quotes containing **48 unique characters**:

```
['\n', ' ', ',', '.', ':', 'A', 'B', 'C', 'D', 'E', 'G', 'H', 'I', 'K', 'L',
 'M', 'N', 'R', 'S', 'T', 'W', 'Y', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
 'x', 'y', 'z']
```

Each character gets a fixed index (sorted alphabetically, with special characters first):

| Character | Index | Character | Index | Character | Index |
|---|---|---|---|---|---|
| `\n` (newline) | 0 | `A` | 5 | `a` | 22 |
| ` ` (space) | 1 | `B` | 6 | `b` | 23 |
| `,` | 2 | `C` | 7 | `e` | 26 |
| `.` | 3 | `S` | 18 | `h` | 29 |
| `:` | 4 | `T` | 19 | `s` | 40 |

### Encoding "The"

```
"T"  →  char_to_idx["T"]  =  19
"h"  →  char_to_idx["h"]  =  29
"e"  →  char_to_idx["e"]  =  26

"The"  →  [19, 29, 26]
```

### Encoding "Success"

```
"S"  →  18
"u"  →  42
"c"  →  24
"c"  →  24      ← same character = same number (always)
"e"  →  26
"s"  →  40
"s"  →  40

"Success"  →  [18, 42, 24, 24, 26, 40, 40]
```

### As a PyTorch Tensor

The list becomes a **tensor** — PyTorch's version of an array that can run on GPUs:

```
"The"     →  tensor([[19, 29, 26]])          shape: (1, 3)
"Success" →  tensor([[18, 42, 24, 24, 26, 40, 40]])   shape: (1, 7)
                      │   │
                      │   └── 3 or 7 characters
                      └── 1 = batch size (processing 1 input at a time)
```

---

## Step 2: Embedding (Numbers → Vectors)

A single number like `19` doesn't tell the network much about what "T" means. The **embedding layer** converts each index into a rich **128-dimensional vector** — a list of 128 numbers that capture the character's "meaning" in a way the network can process.

### How it works

The embedding layer is a big lookup table: a **48 × 128 matrix** (48 characters, 128 numbers each). To embed character index `19` ("T"), it simply grabs row 19 from the table.

```
Embedding weight matrix: 48 rows × 128 columns
                         (one row per character)

Index 19 ("T") → row 19 → [-0.4093, 0.2465, 0.4073, 0.2483, -2.2228,
                             0.1594, 0.6341, 1.8311, -0.6375, 3.0199,
                             ... 118 more numbers ...]
```

### Real values for "The"

```
"T" (idx 19) → [-0.4093,  0.2465,  0.4073,  0.2483, -2.2228,  0.1594,  0.6341,  1.8311, -0.6375,  3.0199, ...]
"h" (idx 29) → [-0.9635, -0.1020,  0.1264, -0.8741,  0.2488, -1.3906,  1.2727,  1.7870, -2.0849,  0.0303, ...]
"e" (idx 26) → [ 1.6687, -0.0966, -1.1840,  1.1613,  1.8374, -2.5836,  0.5949,  0.7173, -1.2540, -0.7586, ...]
                 (showing first 10 of 128 dimensions)
```

### Shape transformation

```
Input:   (1, 3)       →  3 integers
Output:  (1, 3, 128)  →  3 vectors of 128 numbers each

Each integer got "expanded" into a 128-number description.
```

### Why 128 dimensions?

These numbers are **learned during training** — the model discovered what values help it predict the next character. Similar characters (like vowels) end up with similar vectors. The 128 dimensions capture relationships like:
- Is this a vowel or consonant?
- Does this character typically start words?
- Does this character often appear after spaces?

**Parameter count:** 48 characters × 128 dimensions = **6,144 learnable numbers**

---

## Step 3: RNN Processing (Vectors → Context)

The **Recurrent Neural Network** is the brain of the model. It reads the embedded characters **one at a time, left to right**, building up an understanding of the context.

### What the RNN does

```
        "T"              "h"              "e"
         │                │                │
    [128-dim vector] [128-dim vector] [128-dim vector]
         │                │                │
         ▼                ▼                ▼
    ┌─────────┐      ┌─────────┐      ┌─────────┐
    │  RNN    │─────>│  RNN    │─────>│  RNN    │
    │ Cell    │ h₁   │ Cell    │ h₂   │ Cell    │ h₃
    └─────────┘      └─────────┘      └─────────┘
         │                │                │
    [256-dim output] [256-dim output] [256-dim output]
         │                │                │
    "seen T"         "seen Th"       "seen The"
    (no useful       (maybe a word    (definitely a
     context yet)     starting)        common word)
```

The same RNN cell is reused at every step, but it carries forward a **hidden state** — a 256-number vector that acts as the model's "memory" of everything it has read so far.

### The hidden state — the model's memory

At each step, the RNN takes two inputs:
1. The current character's embedding (128 numbers)
2. The previous hidden state (256 numbers — what it remembers so far)

And produces two outputs:
1. An output vector (256 numbers — its understanding at this position)
2. A new hidden state (256 numbers — updated memory for the next step)

```
After reading "T":    hidden state encodes "I've seen a capital T"
After reading "Th":   hidden state encodes "I've seen Th — probably 'The' or 'This'"
After reading "The":  hidden state encodes "I've seen 'The' — next is likely a space"
```

### Real values

```
RNN output after "e" (first 10 of 256 dimensions):
[0.9802, -0.9089, 0.9648, -0.9840, -0.2529, 0.5798, -0.9414, 0.9983, -0.6024, 0.9872]

Hidden state (h) shape: (2 layers, 1 batch, 256)
  → 2 layers because our RNN is stacked (layer 1 feeds into layer 2)
  → 256 numbers = the model's ENTIRE memory of the input so far
```

### Two layers deep

Our RNN has **2 stacked layers**. The first layer reads the embeddings; the second layer reads the first layer's output. This gives the model more capacity to learn complex patterns:

```
                 "T"           "h"           "e"
                  │             │             │
            [128-dim embed] [128-dim]    [128-dim]
                  │             │             │
                  ▼             ▼             ▼
Layer 1:    ┌──────────┐  ┌──────────┐  ┌──────────┐
            │ RNN      │→ │ RNN      │→ │ RNN      │   (learns basic patterns)
            └──────────┘  └──────────┘  └──────────┘
                  │             │             │
            [256-dim]     [256-dim]     [256-dim]
                  │             │             │
                  ▼             ▼             ▼
Layer 2:    ┌──────────┐  ┌──────────┐  ┌──────────┐
            │ RNN      │→ │ RNN      │→ │ RNN      │   (learns higher-level patterns)
            └──────────┘  └──────────┘  └──────────┘
                  │             │             │
            [256-dim]     [256-dim]     [256-dim]
                                             │
                                        FINAL OUTPUT
                                        (used for prediction)
```

### Parameter count for RNN

| Parameter | Shape | Count |
|---|---|---|
| Layer 1: input→hidden weights | 256 × 128 | 32,768 |
| Layer 1: hidden→hidden weights | 256 × 256 | 65,536 |
| Layer 1: biases (×2) | 256 + 256 | 512 |
| Layer 2: input→hidden weights | 256 × 256 | 65,536 |
| Layer 2: hidden→hidden weights | 256 × 256 | 65,536 |
| Layer 2: biases (×2) | 256 + 256 | 512 |
| **RNN total** | | **230,400** |

---

## Step 4: Output Layer (Context → Scores)

The RNN gave us a 256-dimensional vector that encodes "everything the model understands about the text so far." Now we need to convert that into a **prediction**: which of the 48 characters should come next?

The **linear output layer** multiplies the 256-dim vector by a **48 × 256 weight matrix**, producing **48 raw scores** (one per character). These scores are called **logits**.

### The math

```
logits = (48×256 weight matrix) × (256-dim RNN output) + (48-dim bias)
       = 48 scores, one for each possible next character
```

### Real logits after "The"

Here are the actual scores the model produces for every character:

```
Character  Logit      Character  Logit      Character  Logit
─────────  ─────      ─────────  ─────      ─────────  ─────
"\n"       -1.68      "A"        -2.54      "a"        -3.97
" "        18.99 ★    "B"        -0.64      "b"        -2.86
","         5.16      "C"        -2.55      "c"         1.86
"."         8.61      "D"        -9.55      "d"         1.99
":"         2.02      "E"        -0.67      "e"        -6.05
                      "T"         2.70      "h"        -6.86
                      "S"        -7.38      "l"         5.83
                      "N"         0.78      "n"         2.63
                                            "o"        -4.90
                                            "r"         5.39
                                            "y"         2.17
```

**The space character (" ") has the highest score at 18.99** — the model is very confident that a space follows "The". This makes sense: "The" is almost always followed by a space in English.

### Top 10 predictions ranked

| Rank | Character | Logit | Interpretation |
|---|---|---|---|
| 1 | `" "` (space) | 18.99 | "The " — overwhelmingly likely |
| 2 | `"."` | 8.61 | "The." — end of sentence |
| 3 | `"l"` | 5.83 | Doesn't make sense for "The" |
| 4 | `"r"` | 5.39 | "Ther" — start of "There"? |
| 5 | `","` | 5.16 | "The," — pause after "The" |
| 6 | `"T"` | 2.70 | Doesn't make sense |
| 7 | `"n"` | 2.63 | "Then"? |
| 8 | `"y"` | 2.17 | "They"? |
| 9 | `":"` | 2.02 | "The:" — unlikely |
| 10 | `"d"` | 2.00 | Doesn't make sense |

**Parameter count:** 48 × 256 weights + 48 biases = **12,336**

---

## Step 5: Softmax (Scores → Probabilities)

Raw logits are hard to interpret — what does "18.99" actually mean? **Softmax** converts the 48 scores into **probabilities** that sum to 100%.

### The formula

```
probability(char) = e^(logit for char) / sum of e^(logit) for ALL 48 chars
```

The exponential (`e^x`) amplifies differences: a logit of 18.99 becomes a huge number, while a logit of 5.83 becomes comparatively tiny.

### Real probabilities after "The"

```
" " (space)  →  e^18.99 / (e^18.99 + e^8.61 + e^5.83 + ... )  =  100.00%
"."          →  e^8.61  / (same sum)                            =    0.00%
"l"          →  e^5.83  / (same sum)                            =    0.00%
"r"          →  e^5.39  / (same sum)                            =    0.00%
(all others) →                                                      ~0.00%

Sum of all 48 probabilities = 1.000000 (exactly 100%)
```

The model is **100% confident** that a space comes after "The". It learned this pattern perfectly from the training data.

### After "Success" — a more interesting spread

```
" " (space)  →  98.71%     ← very likely, but not 100%
","          →   0.70%     ← "Success," is reasonable
"i"          →   0.15%     ← "Successi..." doesn't make sense
"."          →   0.14%     ← "Success." would work
"d"          →   0.12%
"t"          →   0.07%
```

---

## Step 6: Picking the Next Character

Now we have probabilities. How do we pick the actual next character? There are two methods:

### Method A: Greedy Decoding (temperature = 0)

**Always pick the character with the highest probability.**

```
After "The":     probabilities → " " has 100.00% → pick " "
After "Success": probabilities → " " has 98.71%  → pick " "
```

Greedy decoding is **deterministic** — same input always gives the same output. This is good for consistency but can be repetitive.

### Method B: Temperature Sampling (temperature > 0)

**Randomly sample from the probabilities**, with temperature controlling how random:

```
temperature < 1.0  →  sharpen probabilities (more confident, less random)
temperature = 1.0  →  use probabilities as-is
temperature > 1.0  →  flatten probabilities (less confident, more random)
```

The formula: divide logits by temperature BEFORE softmax:

```
adjusted_logits = original_logits / temperature
probabilities = softmax(adjusted_logits)
next_char = random_sample(probabilities)
```

### Real temperature effect after "Success"

| Temperature | `" "` | `","` | `"i"` | `"."` | `"d"` | Behavior |
|---|---|---|---|---|---|---|
| 0.3 | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | Almost greedy |
| 0.5 | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | Still very focused |
| 0.8 | 99.7% | 0.2% | 0.0% | 0.0% | 0.0% | Tiny chance of comma |
| 1.0 | 98.7% | 0.7% | 0.2% | 0.1% | 0.1% | Small variety possible |
| 1.5 | 90.6% | 3.3% | 1.2% | 1.1% | 1.0% | Real variety |

At **temperature 1.5**, there's a 3.3% chance of picking "," instead of " " — which would give "Success," instead of "Success ". This is how temperature creates variety.

### How random sampling works

```python
# Temperature = 1.5, after "Success"
probabilities = [" "=90.6%, ","=3.3%, "i"=1.2%, "."=1.1%, ...]

# torch.multinomial draws ONE random sample from these probabilities
# Think of it as a weighted dice roll:
#   90.6% chance → " "
#    3.3% chance → ","
#    1.2% chance → "i"
#   etc.

next_char = random_sample(probabilities)  # usually " ", sometimes "," or others
```

---

## Step 7: The Autoregressive Loop

Here's the key insight: **text generation is a loop**. After predicting one character, we feed it back into the model as input to predict the next character. This is called **autoregressive generation**.

### Traced: 10 steps from "The" (greedy)

```
Step 1:  Model reads "The"
         → top predictions: " "=100.0%, "."=0.0%, "l"=0.0%
         → picks: " " (space)
         → text so far: "The "

Step 2:  Model reads " " (with hidden state remembering "The")
         → top predictions: "o"=65.4%, "f"=34.2%, "g"=0.2%
         → picks: "o"
         → text so far: "The o"

Step 3:  Model reads "o" (remembering "The ")
         → top predictions: "n"=99.8%, "t"=0.1%, "u"=0.1%
         → picks: "n"
         → text so far: "The on"

Step 4:  Model reads "n" (remembering "The o")
         → top predictions: "l"=99.8%, "t"=0.1%, "c"=0.1%
         → picks: "l"
         → text so far: "The onl"

Step 5:  Model reads "l" (remembering "The on")
         → top predictions: "y"=99.9%, "l"=0.0%, "o"=0.0%
         → picks: "y"
         → text so far: "The only"

Step 6:  Model reads "y" (remembering "The onl")
         → top predictions: " "=99.9%, "."=0.0%, ","=0.0%
         → picks: " " (space)
         → text so far: "The only "

Step 7:  Model reads " " (remembering "The only")
         → top predictions: "w"=69.8%, "i"=29.3%, "h"=0.4%
         → picks: "w"
         → text so far: "The only w"

Step 8:  Model reads "w" (remembering "The only ")
         → top predictions: "a"=99.8%, "e"=0.1%, "o"=0.1%
         → picks: "a"
         → text so far: "The only wa"

Step 9:  Model reads "a" (remembering "The only w")
         → top predictions: "y"=99.9%, "r"=0.0%, "l"=0.0%
         → picks: "y"
         → text so far: "The only way"

Step 10: Model reads "y" (remembering "The only wa")
         → top predictions: " "=99.9%, "."=0.0%, "s"=0.0%
         → picks: " " (space)
         → text so far: "The only way "
```

After 10 steps: **"The only way "** — the beginning of the quote "The only way to do great work is to love what you do."

### What the hidden state carries

Notice how at Step 7, the model saw only the single character " " (space), yet it predicted "w" with 69.8% confidence. How?

Because the hidden state (256 numbers) carries the **memory of everything before**. The model isn't just seeing " " — it's seeing " " in the context of "The only". That context is encoded in the 256-number hidden state vector.

```
Hidden state after 10 steps (first 5 of 256 values):
[0.9326, 0.9406, 0.9853, -0.9945, -0.9803]

These 256 numbers are the model's ENTIRE memory of "The only way ".
All the context is compressed into this fixed-size vector.
```

### The loop visualized

```
       ┌──────────────────────────────────────────────────┐
       │                                                  │
       ▼                                                  │
  ┌─────────┐    ┌───────────┐    ┌────────┐    ┌───────────┐
  │ Embed   │ →  │   RNN     │ →  │ Linear │ →  │ Softmax + │
  │ char    │    │ + hidden  │    │ layer  │    │   Pick    │
  └─────────┘    └───────────┘    └────────┘    └───────────┘
       ▲              │                               │
       │              ▼                               │
       │         new hidden                           │
       │         state (256)                          │
       │                                              │
       └──────────── next character ◄─────────────────┘
```

Each iteration:
1. Embed the latest character → 128-dim vector
2. Feed it + previous hidden state into RNN → new 256-dim output + updated hidden state
3. Linear layer converts 256-dim → 48 logits
4. Softmax → probabilities, pick a character
5. Feed that character back to step 1

This repeats for however many characters you want to generate (default: 200).

---

## Complete Model Architecture

```
TinyLanguageModel
├── Embedding(48, 128)          ←  6,144 parameters
├── RNN(128, 256, 2 layers)     ←  230,400 parameters
└── Linear(256, 48)             ←  12,336 parameters
                                   ─────────────────
                            TOTAL: 248,880 parameters
```

### Parameter breakdown

| Layer | Weight Shape | Bias Shape | Parameters | Purpose |
|---|---|---|---|---|
| `embedding.weight` | 48 × 128 | — | 6,144 | Character → vector lookup |
| `rnn.weight_ih_l0` | 256 × 128 | 256 | 32,768 + 256 | Layer 1: input to hidden |
| `rnn.weight_hh_l0` | 256 × 256 | 256 | 65,536 + 256 | Layer 1: hidden to hidden |
| `rnn.weight_ih_l1` | 256 × 256 | 256 | 65,536 + 256 | Layer 2: input to hidden |
| `rnn.weight_hh_l1` | 256 × 256 | 256 | 65,536 + 256 | Layer 2: hidden to hidden |
| `output_layer.weight` | 48 × 256 | 48 | 12,288 + 48 | Hidden → character scores |
| **Total** | | | **248,880** | |

---

## Full Worked Example: "The" → "The only way to do great work..."

### Phase 1: Process the seed text

```
Input: "The"

Step 1 — Tokenize:
  "The"  →  [19, 29, 26]

Step 2 — Embed:
  [19, 29, 26]  →  3 vectors of 128 dimensions
  Shape: (1, 3, 128)

Step 3 — RNN reads all 3 characters:
  Position 0 ("T"):  embedding + zeros  →  output₀ + hidden₁
  Position 1 ("h"):  embedding + hidden₁ →  output₁ + hidden₂
  Position 2 ("e"):  embedding + hidden₂ →  output₂ + hidden₃
  Shape: (1, 3, 256)

  We only need output₂ (the last position) — it encodes "The".

Step 4 — Linear layer on output₂:
  (256-dim)  →  (48 logits)
  Top logit: " " = 18.99

Step 5 — Softmax:
  " " = 100.00%

Step 6 — Pick:
  Greedy: " " (space)
  Result: "The "
```

### Phase 2: Autoregressive generation (character by character)

```
"The "  →  feed " " + hidden₃  →  "o" (65.4%)   →  "The o"
"The o" →  feed "o" + hidden₄  →  "n" (99.8%)   →  "The on"
"The on" → feed "n" + hidden₅  →  "l" (99.8%)   →  "The onl"
"The onl" → feed "l" + hidden₆  → "y" (99.9%)   →  "The only"
"The only" → feed "y" + hidden₇  → " " (99.9%)  →  "The only "
"The only " → feed " " + hidden₈ → "w" (69.8%)  →  "The only w"
"The only w" → feed "w" + hidden₉ → "a" (99.8%) →  "The only wa"
"The only wa" → feed "a" + hidden₁₀→ "y" (99.9%)→  "The only way"
"The only way" → feed "y" + hidden₁₁→ " " (99.9%)→ "The only way "
"The only way " → feed " " + hidden₁₂→ "t" (...)  → "The only way t"
... continues for 200 characters total ...
```

### Final output (greedy, 200 characters)

```
The only way to do great work is to love what you do today.
Go confide of fear.
The only think, we become.
Happiness is not
```

---

## Full Worked Example: "Success"

### Phase 1: Process the seed

```
Input: "Success"

Step 1 — Tokenize:
  "Success"  →  [18, 42, 24, 24, 26, 40, 40]

Step 2 — Embed:
  7 characters  →  7 vectors of 128 dimensions
  Shape: (1, 7, 128)

Step 3 — RNN reads all 7 characters left to right:
  Position 0 ("S"):  RNN builds context for "S"
  Position 1 ("u"):  RNN builds context for "Su"
  Position 2 ("c"):  RNN builds context for "Suc"
  Position 3 ("c"):  RNN builds context for "Succ"
  Position 4 ("e"):  RNN builds context for "Succe"
  Position 5 ("s"):  RNN builds context for "Succes"
  Position 6 ("s"):  RNN builds context for "Success"  ← this is what we use
  Shape: (1, 7, 256)

Step 4 — Linear layer on position 6:
  (256-dim)  →  (48 logits)

Step 5 — Softmax:
  " " = 98.71%
  "," =  0.70%
  "i" =  0.15%
  "." =  0.14%

Step 6 — Pick:
  Greedy: " " (space)
  Result: "Success "
```

### Phase 2: Continue generating

```
"Success "    → " " → next prediction → continues character by character
```

### Final output (greedy)

```
Success, but realize how conft, that is real weadous.
What you get by achieving your goals is not to be happy.
Life is what hap
```

Notice the model generates some nonsense words ("conft", "weadous") — this is because it learned character-level patterns, not word-level understanding. With only 6,201 characters of training data and 248,880 parameters, it captures the style but not perfect grammar.

---

## How Temperature Changes Everything

The same seed "The" produces very different outputs depending on temperature:

### Temperature = 0 (greedy)

```
The only way to do great work is to love what you do today.
```
- Always the same output
- Directly reproduces the most likely training pattern
- Safe but predictable

### Temperature = 0.5 (conservative)

```
The future depends on what you do today.
Go confide of fear.
```
- Very close to greedy, occasionally picks the 2nd-most-likely character
- Still recognizable quotes

### Temperature = 1.0 (balanced)

```
The only way to do great work is to love what you dreame.
It is better towe ot onever loved make it...
```
- Mix of real words and invented ones
- Sentence structure mostly intact
- Different output each run

### Temperature = 1.5 (wild)

```
The only impossible journey is the one you never beautinue that counts.
Neturn your wounds into wisdom.
```
- Mostly garbled but with occasional real phrases
- Very different each run
- Shows the model's learned character patterns breaking down

### Why this happens

```
Temperature 0.3:  " "=100.0%  →  always picks space (certain)
Temperature 1.0:  " "=98.7%   →  usually picks space (very likely)
Temperature 1.5:  " "=90.6%   →  often picks space but sometimes comma or period (uncertain)
Temperature 3.0:  " "=40.0%   →  almost random between many characters (chaotic)
```

Lower temperature = sharper probabilities = more predictable text.
Higher temperature = flatter probabilities = more random/creative text.

---

## Summary: The Complete Pipeline

```
 INPUT                STEP 1          STEP 2           STEP 3
 "The"          →   [19, 29, 26]  →  3×128 vectors  →  RNN processes
 (text)             (tokenize)       (embed)            left to right

        STEP 4          STEP 5            STEP 6         STEP 7
     →  48 logits    →  48 probabilities → pick " "   →  feed back
        (linear)        (softmax)          (sample)       and REPEAT
                                                          200 times
                                                             │
                                                             ▼
                                                          OUTPUT
         "The only way to do great work is to love what you do today."
```

Every large language model (GPT-4, Claude, etc.) follows this same fundamental loop — they just do it with billions of parameters instead of 248,880, and with words/subwords instead of characters.
