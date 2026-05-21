# Tiny AI Model — Proof of Concept

## Goal
Build a small neural network from scratch to understand how AI models learn, then interact with it to see it produce output — like a mini version of what large language models (LLMs) do.

## What We're Building
A **character-level text generator** — a tiny neural network that learns patterns from a small text dataset and generates new text one character at a time.

Think of it like this: if you show it hundreds of Shakespeare lines, it learns patterns like "after 'th', the letter 'e' often comes next" and can generate Shakespeare-like text.

## Why This Approach
- Simple enough to understand every piece
- Runs on any laptop (no GPU needed)
- Shows the same core concepts used in ChatGPT, Claude, etc. (just massively scaled down)
- You can see the model improve in real-time during training

---

## Requirements

### Functional
1. **Training**: Feed a small text file into the model and have it learn patterns
2. **Generation**: After training, give it a starting word/character and watch it generate new text
3. **Visualization**: Show training progress (how the model's "loss" decreases over time)
4. **Interactive**: A simple script to chat with / prompt the trained model

### Technical
- **Language**: Python 3.10+
- **Library**: PyTorch (industry-standard, great for learning)
- **Model type**: Simple RNN (Recurrent Neural Network) or Transformer (tiny version)
- **Dataset**: A small text file (~100KB) — e.g., a collection of quotes, short stories, or song titles
- **Hardware**: CPU only (no GPU required)

### Non-Functional
- Code should be heavily commented for learning purposes
- Each step should be runnable independently
- Total training time: under 5 minutes on a laptop

---

## Key AI Concepts You'll Learn

| Concept | What It Means (Simply) |
|---|---|
| **Neural Network** | A program that learns patterns from data, loosely inspired by how brain neurons connect |
| **Training** | Showing the model examples so it adjusts its internal numbers (weights) to get better |
| **Loss** | A score of how wrong the model is — training tries to make this number smaller |
| **Tokens** | The pieces we break text into (in our case, individual characters) |
| **Inference** | Using the trained model to generate new output |
| **Epoch** | One full pass through the entire training dataset |
| **Overfitting** | When the model memorizes the training data instead of learning general patterns |

---

## Execution Plan (Granular Steps)

Each step below is a separate branch and PR, so you can review exactly what changed and why.

---

### Step 01 — Project skeleton (`step-01-project-skeleton`)
> *What you'll learn: How Python projects are organized*
- [ ] Create the folder structure: `src/`, `data/`, `outputs/`
- [ ] Add `requirements.txt` with dependencies (torch, matplotlib)
- [ ] Add a `.gitignore` to keep generated files out of git
- [ ] Add a `README.md` with a one-liner about the project

### Step 02 — Verify PyTorch works (`step-02-verify-pytorch`)
> *What you'll learn: How to import and test a library, what tensors are*
- [ ] Create `src/hello_pytorch.py` — a tiny script that:
  - Imports PyTorch and prints the version
  - Creates a "tensor" (a number/array that PyTorch uses)
  - Does a simple math operation on tensors
  - Prints results so you can see it working

### Step 03 — Add training data (`step-03-training-data`)
> *What you'll learn: What a "dataset" looks like for AI*
- [ ] Add `data/input.txt` with a small collection of text (~100 lines)
- [ ] Create `src/explore_data.py` that:
  - Opens and reads the file
  - Counts total characters, unique characters, and lines
  - Prints a preview of the data

### Step 04 — Character vocabulary (`step-04-char-vocabulary`)
> *What you'll learn: How text becomes numbers (tokenization)*
- [ ] Create `src/vocabulary.py` that:
  - Reads the text and finds every unique character
  - Builds a `char_to_index` dictionary (e.g., 'a' → 0, 'b' → 1)
  - Builds an `index_to_char` dictionary (reverse: 0 → 'a')
  - Demonstrates encoding a sentence into numbers
  - Demonstrates decoding numbers back to text

### Step 05 — Create training sequences (`step-05-training-sequences`)
> *What you'll learn: How AI sees "input → expected output" pairs*
- [ ] Create `src/dataset.py` that:
  - Takes the text and slices it into chunks (e.g., 50 characters each)
  - For each chunk: input = characters 1-49, target = characters 2-50
  - Converts these to PyTorch tensors
  - Wraps it in a PyTorch Dataset class (the standard way to feed data)
  - Prints a few example input/target pairs so you can see the pattern

### Step 06 — DataLoader: batching the data (`step-06-dataloader`)
> *What you'll learn: What "batches" are and why models don't see all data at once*
- [ ] Update `src/dataset.py` to:
  - Create a PyTorch DataLoader that groups sequences into batches
  - Print the shape of one batch (batch_size x sequence_length)
  - Explain what each dimension means
- [ ] Add a runnable `if __name__ == "__main__"` block to test it end-to-end

### Step 07 — Define the neural network (`step-07-model-skeleton`)
> *What you'll learn: What a neural network looks like in code*
- [ ] Create `src/model.py` with a `TinyLanguageModel` class containing:
  - **Embedding layer** — turns character numbers into rich vectors
  - **RNN layer** — processes sequences and remembers context
  - **Output layer** — predicts which character comes next
  - A `forward()` method that chains them together
- [ ] Add a test at the bottom: create the model, pass in fake data, print output shape

### Step 08 — Loss function and optimizer (`step-08-loss-and-optimizer`)
> *What you'll learn: How the model measures "how wrong am I" and improves*
- [ ] Create `src/training_setup.py` that:
  - Imports the model from step 07
  - Sets up `CrossEntropyLoss` (the scoring function) — explain what it does
  - Sets up `Adam` optimizer (the improvement engine) — explain what it does
  - Runs one fake forward pass, computes loss, does one backward pass
  - Prints the loss value to show it works

### Step 09 — Training loop: one epoch (`step-09-single-epoch`)
> *What you'll learn: What happens in one full pass through the data*
- [ ] Create `src/train.py` with a `train_one_epoch()` function that:
  - Loops through every batch in the DataLoader
  - For each batch: forward pass → compute loss → backward pass → update weights
  - Prints loss every N batches so you can watch it
  - Returns the average loss for the epoch

### Step 10 — Full training with progress tracking (`step-10-full-training`)
> *What you'll learn: How models get better over many epochs*
- [ ] Update `src/train.py` to:
  - Run multiple epochs (e.g., 50)
  - Track loss at each epoch in a list
  - Print a summary after training (start loss vs. end loss)
  - Save the trained model weights to `outputs/model.pth`

### Step 11 — Plot the training curve (`step-11-loss-plot`)
> *What you'll learn: How to visualize whether training is working*
- [ ] Add plotting to `src/train.py` (using matplotlib):
  - Plot epoch number (x-axis) vs. loss (y-axis)
  - Save the chart to `outputs/loss_plot.png`
  - The curve should go downward — that means the model is learning

### Step 12 — Text generation (inference) (`step-12-text-generation`)
> *What you'll learn: How a trained model produces new text*
- [ ] Create `src/generate.py` that:
  - Loads the saved model from `outputs/model.pth`
  - Takes a starting string (e.g., "The ")
  - Predicts the next character, appends it, repeats N times
  - Prints the generated text

### Step 13 — Temperature control (`step-13-temperature`)
> *What you'll learn: How "creativity" is controlled in AI models*
- [ ] Update `src/generate.py` to:
  - Accept a `temperature` parameter (0.1 to 2.0)
  - Low temperature (0.2) → very predictable, repetitive text
  - High temperature (1.5) → creative but sometimes nonsensical
  - Generate and print samples at 3 different temperatures to compare

### Step 14 — Interactive prompt mode (`step-14-interactive`)
> *What you'll learn: How chatbots work — it's just a loop of "you type → model responds"*
- [ ] Create `src/interactive.py` that:
  - Loads the trained model
  - Starts a loop: asks you to type a prompt
  - Generates a continuation of your prompt
  - Keeps going until you type "quit"

### Step 15 — Experiment and document (`step-15-experiments`)
> *What you'll learn: How changing settings affects AI output*
- [ ] Create `src/experiment.py` that runs comparisons:
  - Train with different amounts of data and compare output quality
  - Train with different model sizes (more/fewer neurons)
  - Show side-by-side generated text for each config
- [ ] Update `README.md` with what you learned and sample outputs

---

## Project Structure (What We'll Build)

```
SimpleAI/
├── REQUIREMENTS.md          # This file
├── data/
│   └── input.txt            # Training text data
├── src/
│   ├── prepare_data.py      # Step 1: Load and prepare text data
│   ├── model.py             # Step 2: Define the neural network
│   ├── train.py             # Step 3: Train the model
│   └── generate.py          # Step 4: Generate text from trained model
├── outputs/
│   ├── model.pth            # Saved trained model
│   └── loss_plot.png        # Training progress chart
└── requirements.txt         # Python dependencies
```

---

## Success Criteria
The POC is complete when:
1. The model trains successfully on the input text
2. It generates coherent (even if imperfect) new text that resembles the training data
3. You understand what each part of the code does and why
