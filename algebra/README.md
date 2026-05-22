# AlgebrAI — Equation Solver

A neural network that learns to solve single-variable linear equations by pattern recognition, not coded math rules.

## What is this?

This project builds a **seq2seq (encoder-decoder) model** (1.16M parameters) that:
1. Reads 50,000 synthetic equations like `"2x + 3 = 7"`
2. Learns to map equations to solutions like `"x = 2"`
3. Achieves **94.6% exact-match accuracy** on unseen test equations

The model learns purely from examples — it has no knowledge of algebra. It discovers the patterns on its own.

## Sample output

```
>> 2x + 3 = 7
   x = 2  ✓

>> 5x - 10 = 15
   x = 5  ✓

>> 3x + 2 = x + 8
   x = 3  ✓

>> 10 - 2x = 4
   x = 3  ✓

>> x / 4 = 3
   x = 12  ✓

>> :batch 10
  10/10 correct (100%)
```

## How to run

All commands assume you are in the `SimpleAI/` root directory with the virtual environment activated.

```bash
# 1. Setup (if not already done)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Generate training data (50,000 equations)
PYTHONPATH=algebra/src python3 algebra/src/generate_data.py

# 3. Train the model (~15 minutes on CPU)
PYTHONPATH=algebra/src python3 algebra/src/train.py

# 4. Evaluate accuracy on test set
PYTHONPATH=algebra/src python3 algebra/src/evaluate.py

# 5. Generate training plots
PYTHONPATH=algebra/src python3 algebra/src/plot_loss.py

# 6. Solve equations interactively
PYTHONPATH=algebra/src python3 algebra/src/interactive.py

# 7. Run experiments
PYTHONPATH=algebra/src python3 algebra/src/experiments.py

# 8. Run all tests (76 tests)
PYTHONPATH=algebra/src pytest algebra/tests/test_e2e.py -v
```

## Quick start (already trained)

If the model is already trained (files exist in `algebra/outputs/`), you can skip steps 2-3 and go straight to the interactive solver:

```bash
source venv/bin/activate
PYTHONPATH=algebra/src python3 algebra/src/interactive.py
```

### Interactive commands

| Command | What it does |
|---|---|
| `2x + 3 = 7` | Solve any equation — just type it |
| `:verify` | Toggle verification mode (substitutes answer back into equation) |
| `:batch N` | Test N random equations from the dataset |
| `:help` | Show all commands |
| `:quit` | Exit |

## Equation types supported

| Type | Example | Accuracy |
|---|---|---|
| `ax = b` | `3x = 12` | 99.4% |
| `ax + b = c` | `2x + 3 = 7` | 97.8% |
| `ax - b = c` | `5x - 10 = 15` | 97.1% |
| `b + ax = c` | `4 + 3x = 13` | 98.3% |
| `b - ax = c` | `10 - 2x = 4` | 97.7% |
| `x / a = b` | `x / 4 = 3` | 65.9% |
| `ax + b = cx + d` | `3x + 2 = x + 8` | 87.2% |

## Architecture

```
"2x + 3 = 7" → Encoder (Bidirectional GRU) → Context Vector → Decoder (GRU) → "x = 2"
  equation       reads both directions        256-dim summary    generates answer    solution
```

### Model details

```
Component        Architecture                  Parameters
Encoder          Embedding(19,64)                  1,216
                 GRU(64,128,2,bidirectional)      442,880
                 Linear(256,256) bridge            65,792
                 Dropout(0.1)                       2,560
                                               ---------
                 Subtotal                        512,448

Decoder          Embedding(19,64)                  1,216
                 GRU(64,256,2)                   494,592
                 Linear(256,19) output           148,339
                 Dropout(0.1)                       4,000
                                               ---------
                 Subtotal                        648,147

Total                                          1,160,595
```

## Project structure

```
algebra/
├── data/
│   └── equations.txt          <- 50,000 equation-solution pairs (1.1 MB)
├── src/
│   ├── generate_data.py       <- Step 02: Generate synthetic equations
│   ├── vocab.py               <- Step 03: Character vocabulary (19 tokens)
│   ├── dataset.py             <- Step 04: PyTorch Dataset + DataLoader
│   ├── encoder.py             <- Step 05: Bidirectional GRU encoder
│   ├── decoder.py             <- Step 06: GRU decoder with teacher forcing
│   ├── seq2seq.py             <- Step 07: Combined encoder-decoder model
│   ├── train.py               <- Step 08: Training loop (50 epochs)
│   ├── evaluate.py            <- Step 09: Accuracy evaluation + per-type breakdown
│   ├── plot_loss.py           <- Step 10: Loss and accuracy visualization
│   ├── interactive.py         <- Step 11: Interactive equation solver
│   └── experiments.py         <- Step 13: Hyperparameter experiments
├── tests/
│   └── test_e2e.py            <- Step 12: 76 E2E tests
├── outputs/
│   ├── model.pth              <- Trained model weights (4.5 MB)
│   ├── vocab.pth              <- Vocabulary mappings
│   ├── loss_history.pth       <- Training history
│   └── loss_plot.png          <- Training progress chart
├── README.md                  <- This file
├── REQUIREMENTS.md            <- 13-step execution plan
├── TECHNICAL_DOC.md           <- Detailed technical documentation
└── VALIDATION_PLAN.md         <- Test criteria for each step
```

## The 13-step learning path

| Step | File | What you learn |
|---|---|---|
| 01 | Planning docs | Execution plan, architecture design |
| 02 | `generate_data.py` | Synthetic data generation, answer-first strategy |
| 03 | `vocab.py` | Character tokenization with special tokens (PAD, SOS, EOS) |
| 04 | `dataset.py` | Variable-length padding, train/test split, DataLoader |
| 05 | `encoder.py` | Bidirectional GRU, context vectors |
| 06 | `decoder.py` | Teacher forcing, autoregressive generation |
| 07 | `seq2seq.py` | Combining encoder + decoder, gradient flow |
| 08 | `train.py` | Training loop, LR scheduler, gradient clipping |
| 09 | `evaluate.py` | Accuracy metrics, per-type analysis, error patterns |
| 10 | `plot_loss.py` | Loss curves, training visualization |
| 11 | `interactive.py` | REPL interface, substitution verification |
| 12 | `test_e2e.py` | Testing AI systems (76 tests) |
| 13 | `experiments.py` | Hyperparameter experiments |

## Key concepts covered

- **Seq2Seq architecture** — encoder-decoder for variable-length input/output
- **GRU (Gated Recurrent Unit)** — improved RNN with memory gates
- **Bidirectional processing** — reading equations in both directions
- **Teacher forcing** — training technique with scheduled reduction
- **Gradient clipping** — preventing exploding gradients in RNNs
- **Learning rate scheduling** — StepLR for fine-tuning
- **Synthetic data generation** — answer-first strategy for clean data
- **Substitution verification** — mathematically verifying model answers

## Training results

```
Final accuracy:  94.6% (9,458 / 10,000 test equations)
Final loss:      0.0098
Training time:   ~15 minutes on CPU
Epochs:          50
```

## Documentation

- [TECHNICAL_DOC.md](TECHNICAL_DOC.md) — detailed docs for every file, method, and concept
- [REQUIREMENTS.md](REQUIREMENTS.md) — 13-step execution plan
- [VALIDATION_PLAN.md](VALIDATION_PLAN.md) — test criteria for each step
