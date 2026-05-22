# Algebra Solver — Validation Plan

How we verify that every component works correctly, from individual functions to the full pipeline.

---

## Overview

Unlike the text generator (where output quality is subjective), the algebra solver produces **mathematically verifiable answers**. Every prediction can be checked by plugging the answer back into the equation.

This gives us three levels of validation:

```
Level 1: Unit Validation       — does each piece work on its own?
Level 2: Integration Validation — do the pieces work together?
Level 3: Acceptance Validation  — does the final product meet the goals?
```

---

## Level 1: Unit Validation (per step)

### Step 01 — Skeleton
| Check | How to verify | Pass criteria |
|---|---|---|
| Directory structure | `ls -R algebra/` | All directories exist |
| REQUIREMENTS.md | Read file | Complete execution plan |
| TECHNICAL_DOC.md | Read file | Initial skeleton with architecture overview |

### Step 02 — Data Generation
| Check | How to verify | Pass criteria |
|---|---|---|
| Equation count | `wc -l algebra/data/equations.txt` | 50,000 lines |
| Format correctness | Parse each line as `equation\tsolution` | All lines parseable |
| Solution correctness | Substitute x back into equation | Both sides equal for all 50K |
| No duplicates | `sort \| uniq -d` | 0 duplicates |
| Type coverage | Count each equation type | All 7 types present |
| Coefficient range | Parse coefficients | All within [-20, 20] |
| Answer range | Parse solutions | All within [-50, 50] |
| Negative answers present | Filter for "x = -" | At least 5,000 negative answers |
| Zero answers present | Filter for "x = 0" | At least 500 zero answers |
| Decimal answers present | Filter for "x = *.5" | At least 2,000 decimal answers |

**Automated test:**
```python
def test_all_solutions_correct():
    """Verify every generated equation by substitution."""
    for equation, solution in load_data():
        x_value = parse_solution(solution)
        left, right = evaluate_equation(equation, x_value)
        assert abs(left - right) < 1e-6, f"Wrong: {equation} -> {solution}"
```

### Step 03 — Vocabulary
| Check | How to verify | Pass criteria |
|---|---|---|
| Vocab covers all chars | Encode every equation | No KeyError |
| Special tokens exist | Check indices 0, 1, 2 | `<PAD>`=0, `<SOS>`=1, `<EOS>`=2 |
| Encode roundtrip | `decode(encode(text)) == text` | True for 100 random equations |
| Deterministic | Encode same text twice | Identical output |
| No collisions | Check all indices unique | `len(set(indices)) == len(indices)` |

**Automated test:**
```python
def test_roundtrip():
    """Encoding then decoding should recover the original text."""
    for text in ["2x + 3 = 7", "x = -15", "x / 4 = 3.5"]:
        assert vocab.decode(vocab.encode(text)) == text
```

### Step 04 — Dataset
| Check | How to verify | Pass criteria |
|---|---|---|
| Total size | `len(dataset)` | 50,000 |
| Train size | `len(train_dataset)` | 40,000 |
| Test size | `len(test_dataset)` | 10,000 |
| No overlap | `set(train) & set(test)` | Empty set |
| Sequence ends with EOS | Check last non-pad token | `== EOS_IDX` for all |
| Padding correct | Check padded positions | All `== PAD_IDX` |
| Batch shapes | Iterate one batch | Encoder: `(batch, max_eq_len)`, Decoder: `(batch, max_sol_len)` |
| DataLoader exhaustion | Iterate full epoch | `num_batches == ceil(40000/64)` |

**Automated test:**
```python
def test_no_data_leakage():
    """Training and test sets must not overlap."""
    train_eqs = {eq for eq, _ in train_dataset.pairs}
    test_eqs = {eq for eq, _ in test_dataset.pairs}
    assert len(train_eqs & test_eqs) == 0
```

### Step 05 — Encoder
| Check | How to verify | Pass criteria |
|---|---|---|
| Output shape | Forward pass with batch | `(batch, seq_len, 256)` |
| Hidden shape | Forward pass | `(2 layers, batch, 256)` for decoder |
| Bidirectional | Check raw hidden | `(2*2 layers, batch, 128)` before transform |
| Gradient flow | `loss.backward()` | No errors, gradients non-zero |
| Handles padding | Pass padded input | No errors |

**Automated test:**
```python
def test_encoder_output_shape():
    batch = torch.randint(0, 23, (8, 15))  # 8 sequences, length 15
    output, hidden = encoder(batch)
    assert output.shape == (8, 15, 256)
    assert hidden.shape == (2, 8, 256)
```

### Step 06 — Decoder
| Check | How to verify | Pass criteria |
|---|---|---|
| Output shape | Forward pass | `(batch, target_len, vocab_size)` |
| Accepts encoder hidden | Pass encoder's output | No size mismatch |
| Teacher forcing | Pass target sequence | Output matches target length |
| Generation mode | Pass only `<SOS>` | Generates until `<EOS>` or max length |
| EOS stops generation | Generate from trained model | Output contains `<EOS>` |

**Automated test:**
```python
def test_decoder_teacher_forcing():
    target = torch.randint(0, 23, (8, 10))  # 8 sequences, length 10
    output = decoder(encoder_hidden, target)
    assert output.shape == (8, 10, vocab_size)
```

### Step 07 — Seq2Seq
| Check | How to verify | Pass criteria |
|---|---|---|
| End-to-end forward | Pass equation batch | Returns predicted sequence |
| Training mode | Forward with target | Returns loss-ready output |
| Inference mode | Forward without target | Returns generated sequence |
| Gradient flow | `loss.backward()` | Gradients in both encoder and decoder |

### Step 08 — Training
| Check | How to verify | Pass criteria |
|---|---|---|
| Loss decreases | Compare epoch 1 vs epoch 30 | Final < 0.3 × initial |
| Model saves | Check file exists | `algebra/outputs/model.pth` created |
| Vocab saves | Check file exists | `algebra/outputs/vocab.pth` created |
| Loss history saves | Check file exists | `algebra/outputs/loss_history.pth` created |
| Training time | Time the run | Under 10 minutes on CPU |
| No NaN loss | Check all epoch losses | No NaN or Inf values |
| Teacher forcing decreases | Log ratio per epoch | Starts 1.0, ends ≤ 0.5 |

**Automated test:**
```python
def test_loss_decreases():
    losses = train(model, dataloader, epochs=10)
    assert losses[-1] < losses[0] * 0.5  # at least 50% reduction
```

### Step 09 — Evaluation
| Check | How to verify | Pass criteria |
|---|---|---|
| Overall accuracy | Test on 10K held-out equations | ≥ 90% |
| Per-type accuracy | Break down by equation type | All types ≥ 80% |
| No data leakage | Verify test set was never trained on | Train/test sets disjoint |
| Substitution verification | Plug answers back in | Verified answers match model output |
| Error analysis | Inspect wrong answers | Errors are understandable (not random) |

**Automated test:**
```python
def test_accuracy_threshold():
    accuracy = evaluate(model, test_dataloader)
    assert accuracy >= 0.90, f"Accuracy {accuracy:.1%} below 90% threshold"
```

### Step 10 — Loss Plot
| Check | How to verify | Pass criteria |
|---|---|---|
| Plot file created | `os.path.exists()` | `algebra/outputs/loss_plot.png` exists |
| Loss curve trends down | Visual inspection | Clear downward trend |
| Accuracy curve trends up | Visual inspection | Clear upward trend |

### Step 11 — Interactive Solver
| Check | How to verify | Pass criteria |
|---|---|---|
| Solves simple equation | Type `"2x + 3 = 7"` | Returns `"x = 2"` |
| Solves negative answer | Type `"2x + 10 = 4"` | Returns `"x = -3"` |
| Solves x-on-both-sides | Type `"3x + 2 = x + 8"` | Returns `"x = 3"` |
| Verification mode | `:verify` then solve | Shows ✓ or ✗ with substitution |
| Batch testing | `:batch 10` | Shows 10 equations with accuracy |
| Invalid input handling | Type `"hello"` | Graceful error message |
| Quit command | `:quit` | Exits cleanly |

### Step 12 — E2E Tests
| Check | How to verify | Pass criteria |
|---|---|---|
| All tests pass | `pytest -v` | 0 failures |
| Coverage | Count test functions | ≥ 40 tests |
| Edge cases covered | Check for negative, zero, decimal tests | All present |

### Step 13 — Experiments
| Check | How to verify | Pass criteria |
|---|---|---|
| All experiments complete | Run script | No errors |
| Results are printed | Check output | Comparison tables with accuracy |
| Conclusions make sense | Read output | Logical explanations |

---

## Level 2: Integration Validation (checkpoints)

These verify that multiple steps work together correctly.

### Checkpoint A: Data Pipeline (after Step 04)
```bash
# Generate data, build vocab, create dataset, iterate batches
python algebra/src/generate_data.py
python -c "
from vocab import AlgebraVocab
from dataset import AlgebraDataset, create_dataloaders
# Load data, encode, batch, verify shapes
"
```
**Pass criteria:** Data generates → encodes → pads → batches without errors.

### Checkpoint B: Model Forward Pass (after Step 07)
```bash
# Full forward pass: equation → encoded → decoded → predicted solution
python -c "
from seq2seq import Seq2Seq
# Create model, pass one batch, get output
"
```
**Pass criteria:** Random equation batch → encoder → decoder → output tensor with correct shape.

### Checkpoint C: Training Pipeline (after Step 08)
```bash
python algebra/src/train.py
```
**Pass criteria:** Training completes, loss decreases from ~2.8 to < 0.5, model saved.

### Checkpoint D: Full Pipeline (after Step 11)
```bash
python algebra/src/interactive.py
# Type: 2x + 3 = 7
# Expect: x = 2
```
**Pass criteria:** Interactive solver loads model, solves equations, verification works.

---

## Level 3: Acceptance Validation (final)

The project is **complete and accepted** when ALL of the following are true:

### Accuracy Requirements
- [ ] Overall accuracy ≥ 90% on 10,000 unseen test equations
- [ ] Accuracy ≥ 80% on every individual equation type
- [ ] Negative answer accuracy ≥ 80%
- [ ] Decimal answer accuracy ≥ 75%

### Functional Requirements
- [ ] All 7 equation types are supported
- [ ] Interactive solver handles valid and invalid input
- [ ] Verification by substitution works correctly
- [ ] Batch testing mode works

### Quality Requirements
- [ ] All E2E tests pass (≥ 40 tests)
- [ ] Training completes under 10 minutes on CPU
- [ ] Code is heavily commented for educational purposes
- [ ] TECHNICAL_DOC.md documents every file and concept
- [ ] Experiments provide insights into model behavior

### Deliverables
- [ ] All 13 steps implemented and documented
- [ ] Each step on its own branch with a PR
- [ ] Demo branch contains all steps merged
- [ ] README updated with algebra solver section

---

## Correctness Verification: The Substitution Method

The gold standard for checking if a solution is correct:

```
Given: equation and predicted x_value

1. Parse the equation into left side and right side
   "2x + 3 = 7"  →  left = "2x + 3",  right = "7"

2. Replace "x" with the predicted value
   x = 2  →  left = "2(2) + 3",  right = "7"

3. Evaluate both sides
   left  = 2 × 2 + 3 = 7
   right = 7

4. Compare
   |left - right| < 0.001  →  ✓ CORRECT
```

This runs automatically during:
- Data generation (verify all 50K equations)
- Evaluation (verify model predictions)
- Interactive mode (when `:verify` is enabled)
- E2E tests (verify correctness programmatically)

---

## Error Categories

When the model gets an answer wrong, it typically falls into one of these categories:

| Error Type | Example | Why it happens |
|---|---|---|
| **Off by one** | Expected `x = 5`, got `x = 4` | Model almost learned the pattern but made an arithmetic error |
| **Wrong sign** | Expected `x = -3`, got `x = 3` | Model struggled with negative numbers |
| **Truncated decimal** | Expected `x = 3.5`, got `x = 3` | Model didn't generate the decimal part |
| **Garbled output** | Expected `x = 12`, got `x = 1x` | Model confused digits — usually happens with large numbers |
| **Incomplete** | Expected `x = 15`, got `x = 1` | Model generated `<EOS>` too early |

Tracking these error types helps us understand the model's weaknesses and improve it.

---

## Testing Commands Quick Reference

```bash
# Run all tests
PYTHONPATH=algebra/src python -m pytest algebra/tests/test_e2e.py -v

# Run specific test class
PYTHONPATH=algebra/src python -m pytest algebra/tests/test_e2e.py::TestDataGeneration -v

# Run with verbose output
PYTHONPATH=algebra/src python -m pytest algebra/tests/test_e2e.py -v -s

# Check accuracy only
PYTHONPATH=algebra/src python algebra/src/evaluate.py

# Quick manual test
PYTHONPATH=algebra/src python algebra/src/interactive.py
```
