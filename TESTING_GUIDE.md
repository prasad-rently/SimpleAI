# SimpleAI Testing Guide

How to run the project, test it, and what to expect.

---

## 1. Setup

```bash
# Clone the repo
git clone https://github.com/prasad-rently/SimpleAI.git
cd SimpleAI

# Create virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install pytest
```

---

## 2. Train the model (~30 seconds)

```bash
PYTHONPATH=src python3 src/train.py
```

**Expected output:**

```
============================================================
STEP 10: FULL TRAINING — 100 EPOCHS
============================================================

  Starting training: 100 epochs, 7 batches/epoch = 700 updates

  Epoch   0/100 | Avg Loss: 3.0012
  Epoch  10/100 | Avg Loss: 1.5678
  Epoch  20/100 | Avg Loss: 0.7543
  ...
  Epoch  90/100 | Avg Loss: 0.0456
  Epoch  99/100 | Avg Loss: 0.0398

  Training complete!
    Start loss : ~3.00
    Final loss : ~0.04
    Reduction  : ~98.6%

  Model saved to outputs/model.pth
  Vocabulary saved to outputs/vocab.pth
  Loss history saved to outputs/loss_history.pth
```

**What to verify:**
- Final loss should be below `0.10` (typically around `0.04`)
- Three files created in `outputs/`: `model.pth`, `vocab.pth`, `loss_history.pth`

---

## 3. Generate text interactively

```bash
PYTHONPATH=src python3 src/interactive.py
```

This opens a chat-like prompt. Type any seed text and the model continues it.

### Test inputs and expected outputs

#### Greedy mode (deterministic — same output every run)

These outputs are **exact** because greedy decoding always picks the highest-probability character.

| Seed | Expected output starts with |
|---|---|
| `The` | `The only way to do great work is to love what you do today.` |
| `Life is` | `Life is what happens when you are busy making others happy too.` |
| `Be` | `Be you set your goals ridiculously high and what lies before us are tiny matters compared` |
| `In the` | `In the silence of our friends.` |
| `Success` | `Success, but realize how conft...` |

To use greedy mode, set temperature to 0:
```
>> :temp 0
>> The
The only way to do great work is to love what you do today.
Go confide of fear.
The only think, we become.
Happiness is not...
```

#### Temperature 0.5 (conservative — close to greedy with slight variation)

```
>> :temp 0.5
>> The
The future depends on what you do today.
Go confide of fear.
The only think, we become...

>> Life is
Life is what happens when you are busy making others happy too.
The only way to have re pand overcoming them is what makes life...
```

**What to expect:** Mostly recognizable quotes from the training data. Minor variations between runs, but the structure is coherent.

#### Temperature 1.0 (balanced — creative but readable)

```
>> :temp 1.0
>> The
The only way to do great work is to love what you dreame.
It is better towe ot onever loved make it...

>> Be
Be is change.
The mind is everything. What you to be one...
```

**What to expect:** Recognizable words and sentence fragments. Some made-up words appear. Output varies between runs.

#### Temperature 1.5 (wild — mostly nonsensical)

```
>> :temp 1.5
>> The
The only impossible journey is the one you never beautinue that counts.
Neturn your wounds into wisdom...

>> Be
Be wearn.
It is better tour goals is not to be how your goals is not tho un thousand ways...
```

**What to expect:** Mostly garbled text with occasional real words. Very different each run. This demonstrates how high temperature adds randomness.

### Interactive commands to test

| Command | What it does | Expected behavior |
|---|---|---|
| `:temp 0.5` | Set temperature to 0.5 | Prints `Temperature set to 0.5` |
| `:temp 0` | Set temperature to 0 (greedy) | Prints `Temperature set to 0 (greedy mode)` |
| `:temp 5.0` | Try to set too high | Prints `Temperature clamped to 3.0 (maximum)` |
| `:length 50` | Set output length to 50 chars | Prints `Length set to 50` |
| `:length 5` | Try to set too low | Prints `Length clamped to minimum: 10` |
| `:settings` | Show current settings | Prints temperature and length |
| `:help` | Show commands | Prints help text |
| `:quit` | Exit the program | Prints `Goodbye! Happy generating.` |

### Invalid input to test

| Input | Expected behavior |
|---|---|
| `Hello!` | `Unknown characters: ['!']` — exclamation mark isn't in the vocabulary |
| `@test` | `Unknown characters: ['@']` — @ isn't in the vocabulary |
| (empty, just press Enter) | No output, prompt reappears |

---

## 4. Run the E2E test suite

```bash
PYTHONPATH=src python3 -m pytest tests/test_e2e.py -v
```

**Expected output:**

```
tests/test_e2e.py::TestDataLoading::test_file_exists PASSED
tests/test_e2e.py::TestDataLoading::test_data_not_empty PASSED
tests/test_e2e.py::TestDataLoading::test_data_character_count PASSED
tests/test_e2e.py::TestDataLoading::test_data_line_count PASSED
tests/test_e2e.py::TestDataLoading::test_data_unique_characters PASSED
tests/test_e2e.py::TestVocabulary::test_vocab_size PASSED
tests/test_e2e.py::TestVocabulary::test_chars_sorted PASSED
tests/test_e2e.py::TestVocabulary::test_encode_returns_list_of_ints PASSED
tests/test_e2e.py::TestVocabulary::test_encode_known_values PASSED
tests/test_e2e.py::TestVocabulary::test_decode_returns_string PASSED
tests/test_e2e.py::TestVocabulary::test_decode_known_values PASSED
tests/test_e2e.py::TestVocabulary::test_roundtrip_encode_decode PASSED
tests/test_e2e.py::TestVocabulary::test_roundtrip_decode_encode PASSED
tests/test_e2e.py::TestVocabulary::test_every_char_has_mapping PASSED
tests/test_e2e.py::TestVocabulary::test_encode_deterministic PASSED
tests/test_e2e.py::TestDataset::test_dataset_length PASSED
tests/test_e2e.py::TestDataset::test_getitem_returns_tuple PASSED
tests/test_e2e.py::TestDataset::test_input_target_shapes PASSED
tests/test_e2e.py::TestDataset::test_input_target_shift PASSED
tests/test_e2e.py::TestDataset::test_input_target_dtype PASSED
tests/test_e2e.py::TestDataset::test_values_in_vocab_range PASSED
tests/test_e2e.py::TestDataLoader::test_batch_count PASSED
tests/test_e2e.py::TestDataLoader::test_batch_shapes PASSED
tests/test_e2e.py::TestDataLoader::test_full_epoch_iteration PASSED
tests/test_e2e.py::TestDataLoader::test_shuffling_changes_order PASSED
tests/test_e2e.py::TestModel::test_parameter_count PASSED
tests/test_e2e.py::TestModel::test_forward_pass_shape PASSED
tests/test_e2e.py::TestModel::test_hidden_state_shape PASSED
tests/test_e2e.py::TestModel::test_forward_with_hidden PASSED
tests/test_e2e.py::TestModel::test_single_input PASSED
tests/test_e2e.py::TestModel::test_output_not_all_same PASSED
tests/test_e2e.py::TestTraining::test_one_epoch_reduces_loss PASSED
tests/test_e2e.py::TestTraining::test_multi_epoch_loss_decreases PASSED
tests/test_e2e.py::TestTraining::test_save_and_load_model PASSED
tests/test_e2e.py::TestTraining::test_save_and_load_vocabulary PASSED
tests/test_e2e.py::TestLossPlot::test_plot_creates_file PASSED
tests/test_e2e.py::TestLossPlot::test_saved_loss_history_exists PASSED
tests/test_e2e.py::TestGeneration::test_greedy_returns_string PASSED
tests/test_e2e.py::TestGeneration::test_greedy_starts_with_seed PASSED
tests/test_e2e.py::TestGeneration::test_greedy_is_deterministic PASSED
tests/test_e2e.py::TestGeneration::test_greedy_generates_correct_length PASSED
tests/test_e2e.py::TestGeneration::test_temperature_returns_string PASSED
tests/test_e2e.py::TestGeneration::test_temperature_correct_length PASSED
tests/test_e2e.py::TestGeneration::test_temperature_generates_valid_chars PASSED
tests/test_e2e.py::TestGeneration::test_different_seeds_different_output PASSED
tests/test_e2e.py::TestGeneration::test_high_temperature_produces_variety PASSED
tests/test_e2e.py::TestInteractive::test_interactive_generator_creation PASSED
tests/test_e2e.py::TestInteractive::test_command_temp PASSED
tests/test_e2e.py::TestInteractive::test_command_length PASSED
tests/test_e2e.py::TestInteractive::test_command_temp_clamp_high PASSED
tests/test_e2e.py::TestInteractive::test_command_length_clamp_low PASSED
tests/test_e2e.py::TestInteractive::test_command_quit_returns_false PASSED
tests/test_e2e.py::TestInteractive::test_command_help_returns_true PASSED
tests/test_e2e.py::TestInteractive::test_generate_with_low_temp PASSED
tests/test_e2e.py::TestFullPipeline::test_train_and_generate PASSED
tests/test_e2e.py::TestFullPipeline::test_save_load_generate_cycle PASSED
tests/test_e2e.py::TestFullPipeline::test_pretrained_model_generates_coherent_text PASSED

57 passed in ~5s
```

**All 57 tests should pass.** If any fail, check:
- You ran `train.py` first (some tests need `outputs/model.pth` and `outputs/vocab.pth`)
- The virtual environment is activated (`source venv/bin/activate`)
- `PYTHONPATH=src` is set so imports resolve

---

## 5. Run experiments (~2 minutes)

```bash
PYTHONPATH=src python3 src/experiment.py
```

This trains 9 separate models with different settings and compares results. You'll see three experiments:

1. **Data size** (25%, 50%, 100%) — more data = lower loss
2. **Model size** (tiny, medium, large) — bigger models learn more but can overfit
3. **Epoch count** (10, 50, 100) — more training helps, then plateaus

---

## 6. Plot the training loss curve

```bash
PYTHONPATH=src python3 src/plot_loss.py
```

Creates `outputs/loss_plot.png` showing loss dropping from ~3.0 to ~0.04 over 100 epochs.

---

## Quick reference

| Command | What it does |
|---|---|
| `PYTHONPATH=src python3 src/train.py` | Train the model (generates outputs/) |
| `PYTHONPATH=src python3 src/interactive.py` | Interactive chat-like generation |
| `PYTHONPATH=src python3 src/plot_loss.py` | Plot training loss curve |
| `PYTHONPATH=src python3 src/experiment.py` | Run hyperparameter experiments |
| `PYTHONPATH=src python3 -m pytest tests/test_e2e.py -v` | Run all 57 E2E tests |
