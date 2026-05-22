# Algebra Solver — Testing Guide

How to run the algebra solver, test it, and what to expect at each stage.

---

## 1. Setup

```bash
# From the SimpleAI root directory
cd SimpleAI

# Create virtual environment (skip if already done)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install pytest
```

---

## 2. Generate training data

```bash
PYTHONPATH=algebra/src python3 algebra/src/generate_data.py
```

**Expected output:**
```
============================================================
STEP 02: GENERATING EQUATION DATA
============================================================

  Generating 50,000 equations...
  Generated 50,000 unique equations.
  Verified: 50,000/50,000 solutions correct.

  Saved to algebra/data/equations.txt
  File size: 1.1 MB

  Sample equations:
    -10x + 1 = -6x + 37    → x = -9
    9 + -4x = -191          → x = 50
    9x + 14 = 302           → x = 32
    ...
```

**What to check:**
- File `algebra/data/equations.txt` exists (1.1 MB)
- 50,000 lines, each with `equation\tsolution` format
- All solutions are integers
- All solutions are verified by substitution

---

## 3. Train the model (~15 minutes)

```bash
PYTHONPATH=algebra/src python3 algebra/src/train.py
```

**Expected output:**
```
============================================================
STEP 08: TRAINING THE ALGEBRA SOLVER
============================================================

Building vocabulary...
  Vocabulary: 19 tokens

Creating dataloaders...
  Train set: 40,000 pairs
  Test set:  10,000 pairs

Creating model...
  Parameters: 1,160,595

Training...
------------------------------------------------------------
  Epoch   9/50 | Loss: 0.1229 | Accuracy: 68.9% | TF: 0.87 | Time: 179s
  Epoch  19/50 | Loss: 0.0538 | Accuracy: 85.4% | TF: 0.73 | Time: 359s
  Epoch  29/50 | Loss: 0.0484 | Accuracy: 87.7% | TF: 0.59 | Time: 538s
  Epoch  39/50 | Loss: 0.0225 | Accuracy: 92.1% | TF: 0.44 | Time: 718s
  Epoch  49/50 | Loss: 0.0098 | Accuracy: 94.6% | TF: 0.30 | Time: 905s

  Training complete in 905.0s (15.1 minutes)
  Final loss:     0.0098
  Final accuracy: 94.6%
------------------------------------------------------------

Saving...
  Model saved to algebra/outputs/model.pth (4541.4 KB)
  Vocabulary saved to algebra/outputs/vocab.pth (1.6 KB)
  History saved to algebra/outputs/loss_history.pth
```

**What to check:**
- Final accuracy is above 90%
- Loss decreases steadily across epochs
- Three files saved in `algebra/outputs/`
- Training completes in ~15 minutes on CPU

---

## 4. Evaluate accuracy

```bash
PYTHONPATH=algebra/src python3 algebra/src/evaluate.py
```

**Expected output:**
```
============================================================
EVALUATION REPORT
============================================================

  Overall accuracy: 94.6% (9,458 / 10,000)
  Target (>= 90%):   PASS

  Per-type accuracy:
  --------------------------------------------------
    Type 1: ax = b                99.4%  ( 693/697 ) ✓
    Type 2: ax + b = c            97.8%  (2120/2167) ✓
    Type 3: ax - b = c            97.1%  (1887/1943) ✓
    Type 4: b + ax = c            98.3%  (1360/1383) ✓
    Type 5: b - ax = c            97.7%  (1311/1342) ✓
    Type 6: x / a = b             65.9%  ( 203/308 ) ✗
    Type 7: ax + b = cx + d       87.2%  (1884/2160) ✓

  Substitution verification: 9458/9458 correct predictions verified (100.0%)
```

**What to check:**
- Overall accuracy >= 90%
- 6 out of 7 types pass the 80% threshold
- Type 6 (division) is a known limitation at ~66%
- 100% substitution verification on correct predictions

---

## 5. Generate loss plot

```bash
PYTHONPATH=algebra/src python3 algebra/src/plot_loss.py
```

**Expected output:**
```
  Plot saved to algebra/outputs/loss_plot.png (116.5 KB)
```

**What to check:**
- File `algebra/outputs/loss_plot.png` exists
- Loss curve trends downward
- Accuracy curve crosses the 90% line
- Teacher forcing decreases from 1.0 to 0.3

---

## 6. Interactive solver

```bash
PYTHONPATH=algebra/src python3 algebra/src/interactive.py
```

**Test inputs and expected outputs:**

```
>> 2x + 3 = 7
   x = 2

>> 5x - 10 = 15
   x = 5

>> 3x + 2 = x + 8
   x = 3

>> 10 - 2x = 4
   x = 3

>> x / 4 = 3
   x = 12

>> 3x = 12
   x = 4
```

**Test commands:**

```
>> :verify
   Verification mode: ON

>> 7x + 1 = 50
   x = 7
   Check: 7x + 1 → 50,  50 → 50  ✓

>> :batch 10
   (solves 10 random equations, shows accuracy)

>> :help
   (shows all available commands)

>> :quit
   Goodbye!
```

**Edge cases to test:**

```
>> hello
   That doesn't look like an equation (missing '=')

>> 5 = 10
   No variable 'x' found in equation.
```

---

## 7. Run experiments (~20 minutes)

```bash
PYTHONPATH=algebra/src python3 algebra/src/experiments.py
```

This runs 4 experiments (10 epochs each):
1. Data size: 2K vs 10K vs 40K examples
2. Model size: small vs medium vs large
3. Equation complexity: per-type accuracy
4. Teacher forcing: always vs half vs never

---

## 8. Run all tests

```bash
PYTHONPATH=algebra/src pytest algebra/tests/test_e2e.py -v
```

**Expected output:**
```
algebra/tests/test_e2e.py::TestDataGeneration::test_data_file_exists PASSED
algebra/tests/test_e2e.py::TestDataGeneration::test_data_has_50000_lines PASSED
...
algebra/tests/test_e2e.py::TestEvaluation::test_trained_model_accuracy_above_90 PASSED
...
algebra/tests/test_e2e.py::TestFullPipeline::test_substitution_verification_on_correct_predictions PASSED

============================== 76 passed in ~7s ==============================
```

**What to check:**
- All 76 tests pass
- No failures or errors
- Accuracy threshold test confirms >= 90%

---

## 9. Run individual step demos

Each source file can be run independently to see its step in action:

```bash
# Step 02: Data generation
PYTHONPATH=algebra/src python3 algebra/src/generate_data.py

# Step 03: Vocabulary
PYTHONPATH=algebra/src python3 algebra/src/vocab.py

# Step 04: Dataset and DataLoader
PYTHONPATH=algebra/src python3 algebra/src/dataset.py

# Step 05: Encoder
PYTHONPATH=algebra/src python3 algebra/src/encoder.py

# Step 06: Decoder
PYTHONPATH=algebra/src python3 algebra/src/decoder.py

# Step 07: Seq2Seq model
PYTHONPATH=algebra/src python3 algebra/src/seq2seq.py

# Step 08: Training
PYTHONPATH=algebra/src python3 algebra/src/train.py

# Step 09: Evaluation
PYTHONPATH=algebra/src python3 algebra/src/evaluate.py

# Step 10: Loss plot
PYTHONPATH=algebra/src python3 algebra/src/plot_loss.py

# Step 11: Interactive solver
PYTHONPATH=algebra/src python3 algebra/src/interactive.py

# Step 13: Experiments
PYTHONPATH=algebra/src python3 algebra/src/experiments.py
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'vocab'"
You forgot `PYTHONPATH=algebra/src` before the command. This tells Python where to find the source files.

### "FileNotFoundError: algebra/data/equations.txt"
Run Step 2 first (`generate_data.py`) to create the training data.

### "FileNotFoundError: algebra/outputs/model.pth"
Run Step 3 first (`train.py`) to train the model and save weights.

### "bad interpreter" error with pip
Your virtual environment path changed. Recreate it:
```bash
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Accuracy below 90%
Training is stochastic — results vary slightly each run. The model should converge to 90%+ with the current settings (50 epochs, lr=0.002, StepLR). If it doesn't, try running training again.
