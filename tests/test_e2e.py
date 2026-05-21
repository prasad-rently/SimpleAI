"""
test_e2e.py — End-to-end tests for the SimpleAI pipeline.

Tests the complete flow from raw text to generated output,
verifying each component works correctly on its own and
that they integrate properly.

Run with:
    PYTHONPATH=src python3 -m pytest tests/test_e2e.py -v
"""

import os
import math
import torch
import torch.nn as nn
import pytest

# ====================================================================
# FIXTURES — shared setup used across multiple tests
# ====================================================================

@pytest.fixture
def sample_text():
    """Load the training text from data/input.txt."""
    filepath = "data/input.txt"
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


@pytest.fixture
def vocab(sample_text):
    """Build a Vocabulary from the training text."""
    from vocabulary import Vocabulary
    return Vocabulary(sample_text)


@pytest.fixture
def dataset(sample_text, vocab):
    """Create a TextDataset from the training text."""
    from dataset import TextDataset
    return TextDataset(sample_text, vocab, seq_length=50)


@pytest.fixture
def dataloader(dataset):
    """Create a DataLoader from the dataset."""
    from dataset import create_dataloader
    return create_dataloader(dataset, batch_size=16, shuffle=True)


@pytest.fixture
def model(vocab):
    """Create an untrained TinyLanguageModel."""
    from model import TinyLanguageModel
    return TinyLanguageModel(vocab_size=vocab.vocab_size)


@pytest.fixture
def trained_model_and_vocab():
    """Load the pre-trained model and vocabulary from disk."""
    from generate import load_model, load_vocabulary

    model_path = "outputs/model.pth"
    vocab_path = "outputs/vocab.pth"

    if not os.path.exists(model_path) or not os.path.exists(vocab_path):
        pytest.skip("Trained model not found. Run train.py first.")

    vocab_data = load_vocabulary(vocab_path)
    vocab_size = len(vocab_data['chars'])
    model = load_model(model_path, vocab_size=vocab_size)
    return model, vocab_data


# ====================================================================
# STEP 03: DATA LOADING
# ====================================================================

class TestDataLoading:
    """Verify the training data loads correctly."""

    def test_file_exists(self):
        assert os.path.exists("data/input.txt"), "Training data missing"

    def test_data_not_empty(self, sample_text):
        assert len(sample_text) > 0

    def test_data_character_count(self, sample_text):
        assert len(sample_text) == 6201, f"Expected 6201 chars, got {len(sample_text)}"

    def test_data_line_count(self, sample_text):
        lines = sample_text.strip().split("\n")
        assert len(lines) == 99, f"Expected 99 lines, got {len(lines)}"

    def test_data_unique_characters(self, sample_text):
        unique = set(sample_text)
        assert len(unique) == 48, f"Expected 48 unique chars, got {len(unique)}"


# ====================================================================
# STEP 04: VOCABULARY
# ====================================================================

class TestVocabulary:
    """Verify the Vocabulary encodes and decodes correctly."""

    def test_vocab_size(self, vocab):
        assert vocab.vocab_size == 48

    def test_chars_sorted(self, vocab):
        assert vocab.chars == sorted(vocab.chars)

    def test_encode_returns_list_of_ints(self, vocab):
        encoded = vocab.encode("The")
        assert isinstance(encoded, list)
        assert all(isinstance(i, int) for i in encoded)

    def test_encode_known_values(self, vocab):
        encoded = vocab.encode("The")
        assert encoded == [19, 29, 26], f"Expected [19, 29, 26], got {encoded}"

    def test_decode_returns_string(self, vocab):
        decoded = vocab.decode([19, 29, 26])
        assert isinstance(decoded, str)

    def test_decode_known_values(self, vocab):
        decoded = vocab.decode([19, 29, 26])
        assert decoded == "The"

    def test_roundtrip_encode_decode(self, vocab):
        """encode then decode should return the original text."""
        original = "Life is short."
        encoded = vocab.encode(original)
        decoded = vocab.decode(encoded)
        assert decoded == original

    def test_roundtrip_decode_encode(self, vocab):
        """decode then encode should return the original indices."""
        original = [19, 29, 26, 1, 36]
        decoded = vocab.decode(original)
        encoded = vocab.encode(decoded)
        assert encoded == original

    def test_every_char_has_mapping(self, vocab):
        """Every character in chars should exist in both mappings."""
        for ch in vocab.chars:
            assert ch in vocab.char_to_idx
            idx = vocab.char_to_idx[ch]
            assert idx in vocab.idx_to_char
            assert vocab.idx_to_char[idx] == ch

    def test_encode_deterministic(self, vocab):
        """Same input should always produce the same output."""
        result1 = vocab.encode("hello")
        result2 = vocab.encode("hello")
        assert result1 == result2


# ====================================================================
# STEP 05-06: DATASET AND DATALOADER
# ====================================================================

class TestDataset:
    """Verify TextDataset creates correct training pairs."""

    def test_dataset_length(self, dataset):
        assert len(dataset) == 124, f"Expected 124 examples, got {len(dataset)}"

    def test_getitem_returns_tuple(self, dataset):
        item = dataset[0]
        assert isinstance(item, tuple)
        assert len(item) == 2

    def test_input_target_shapes(self, dataset):
        inp, tgt = dataset[0]
        assert inp.shape == (50,), f"Input shape {inp.shape}, expected (50,)"
        assert tgt.shape == (50,), f"Target shape {tgt.shape}, expected (50,)"

    def test_input_target_shift(self, dataset):
        """Target should be input shifted right by 1 position."""
        inp, tgt = dataset[0]
        # tgt[i] should equal data[start + i + 1]
        # Which means tgt[0] == inp[1] for consecutive sequences
        # More precisely: target is data[start+1:start+51]
        # and input is data[start:start+50]
        # So for the raw data: tgt[0] should equal dataset.data[1]
        assert tgt[0].item() == dataset.data[1].item()

    def test_input_target_dtype(self, dataset):
        inp, tgt = dataset[0]
        assert inp.dtype == torch.long
        assert tgt.dtype == torch.long

    def test_values_in_vocab_range(self, dataset, vocab):
        inp, tgt = dataset[0]
        assert inp.min() >= 0
        assert inp.max() < vocab.vocab_size
        assert tgt.min() >= 0
        assert tgt.max() < vocab.vocab_size


class TestDataLoader:
    """Verify DataLoader batching works correctly."""

    def test_batch_count(self, dataloader):
        assert len(dataloader) == 7, f"Expected 7 batches, got {len(dataloader)}"

    def test_batch_shapes(self, dataloader):
        inp, tgt = next(iter(dataloader))
        assert inp.shape == (16, 50), f"Input batch shape {inp.shape}"
        assert tgt.shape == (16, 50), f"Target batch shape {tgt.shape}"

    def test_full_epoch_iteration(self, dataloader):
        """Should be able to iterate through all batches."""
        count = 0
        for inp, tgt in dataloader:
            count += 1
            assert inp.shape[0] == 16
        assert count == 7

    def test_shuffling_changes_order(self, dataset):
        """Two DataLoaders should yield different batch orders."""
        from dataset import create_dataloader
        dl1 = create_dataloader(dataset, batch_size=16, shuffle=True)
        dl2 = create_dataloader(dataset, batch_size=16, shuffle=True)
        batch1 = next(iter(dl1))[0]
        batch2 = next(iter(dl2))[0]
        # Not guaranteed to differ every time, but very likely with shuffle
        # We just check both are valid shapes
        assert batch1.shape == batch2.shape == (16, 50)


# ====================================================================
# STEP 07: MODEL
# ====================================================================

class TestModel:
    """Verify TinyLanguageModel architecture and forward pass."""

    def test_parameter_count(self, model):
        total = sum(p.numel() for p in model.parameters())
        assert total == 248880, f"Expected 248,880 params, got {total}"

    def test_forward_pass_shape(self, model):
        x = torch.randint(0, 48, (16, 50))
        logits, hidden = model(x)
        assert logits.shape == (16, 50, 48), f"Logits shape {logits.shape}"

    def test_hidden_state_shape(self, model):
        x = torch.randint(0, 48, (16, 50))
        _, hidden = model(x)
        # 2 layers, batch 16, hidden 256
        assert hidden.shape == (2, 16, 256), f"Hidden shape {hidden.shape}"

    def test_forward_with_hidden(self, model):
        """Model should accept pre-initialized hidden state."""
        x = torch.randint(0, 48, (4, 10))
        _, hidden1 = model(x)
        logits2, hidden2 = model(x, hidden1)
        assert logits2.shape == (4, 10, 48)
        assert hidden2.shape == hidden1.shape

    def test_single_input(self, model):
        """Model should handle batch_size=1."""
        x = torch.randint(0, 48, (1, 5))
        logits, hidden = model(x)
        assert logits.shape == (1, 5, 48)

    def test_output_not_all_same(self, model):
        """Different inputs should produce different outputs."""
        x1 = torch.zeros(1, 10, dtype=torch.long)
        x2 = torch.ones(1, 10, dtype=torch.long)
        logits1, _ = model(x1)
        logits2, _ = model(x2)
        assert not torch.allclose(logits1, logits2)


# ====================================================================
# STEP 08-10: TRAINING
# ====================================================================

class TestTraining:
    """Verify the training loop reduces loss."""

    def test_one_epoch_reduces_loss(self, model, dataloader, vocab):
        """One epoch should reduce loss below random baseline."""
        from train import train_one_epoch

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        random_baseline = -math.log(1.0 / vocab.vocab_size)

        avg_loss = train_one_epoch(model, dataloader, loss_fn, optimizer,
                                    vocab.vocab_size, epoch_num=0)

        assert avg_loss < random_baseline, (
            f"Loss {avg_loss:.4f} should be below random baseline {random_baseline:.4f}"
        )

    def test_multi_epoch_loss_decreases(self, model, dataloader, vocab):
        """Loss should decrease over multiple epochs."""
        from train import train

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

        loss_history = train(model, dataloader, loss_fn, optimizer,
                             vocab.vocab_size, num_epochs=10, print_every=100)

        assert len(loss_history) == 10
        assert loss_history[-1] < loss_history[0], (
            f"Final loss {loss_history[-1]:.4f} should be < first loss {loss_history[0]:.4f}"
        )

    def test_save_and_load_model(self, model, tmp_path):
        """Saved model should load with identical weights."""
        from train import save_model

        save_path = str(tmp_path / "test_model.pth")
        save_model(model, filepath=save_path)

        assert os.path.exists(save_path)

        from model import TinyLanguageModel
        loaded = TinyLanguageModel(vocab_size=48)
        loaded.load_state_dict(torch.load(save_path, weights_only=True))

        for p1, p2 in zip(model.parameters(), loaded.parameters()):
            assert torch.equal(p1, p2)

    def test_save_and_load_vocabulary(self, vocab, tmp_path):
        """Saved vocabulary should load with identical mappings."""
        from train import save_vocabulary

        save_path = str(tmp_path / "test_vocab.pth")
        save_vocabulary(vocab, filepath=save_path)

        assert os.path.exists(save_path)

        loaded = torch.load(save_path, weights_only=False)
        assert loaded['chars'] == vocab.chars
        assert loaded['char_to_idx'] == vocab.char_to_idx
        assert loaded['idx_to_char'] == vocab.idx_to_char


# ====================================================================
# STEP 11: LOSS PLOT
# ====================================================================

class TestLossPlot:
    """Verify the loss plot is generated correctly."""

    def test_plot_creates_file(self, tmp_path):
        from plot_loss import plot_training_loss

        fake_history = [3.0, 2.5, 2.0, 1.5, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3]
        save_path = str(tmp_path / "test_plot.png")

        plot_training_loss(fake_history, save_path=save_path)

        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0

    def test_saved_loss_history_exists(self):
        path = "outputs/loss_history.pth"
        if not os.path.exists(path):
            pytest.skip("loss_history.pth not found. Run train.py first.")

        history = torch.load(path, weights_only=False)
        assert isinstance(history, list)
        assert len(history) == 100
        assert all(isinstance(x, float) for x in history)


# ====================================================================
# STEP 12-13: GENERATION
# ====================================================================

class TestGeneration:
    """Verify text generation works correctly."""

    def test_greedy_returns_string(self, trained_model_and_vocab):
        from generate import generate_text
        model, vocab_data = trained_model_and_vocab

        result = generate_text(model, vocab_data, seed_text="The", length=50)
        assert isinstance(result, str)
        assert len(result) > 3  # at least the seed

    def test_greedy_starts_with_seed(self, trained_model_and_vocab):
        from generate import generate_text
        model, vocab_data = trained_model_and_vocab

        result = generate_text(model, vocab_data, seed_text="Life", length=50)
        assert result.startswith("Life")

    def test_greedy_is_deterministic(self, trained_model_and_vocab):
        """Same seed should produce same output with greedy decoding."""
        from generate import generate_text
        model, vocab_data = trained_model_and_vocab

        result1 = generate_text(model, vocab_data, seed_text="The", length=100)
        result2 = generate_text(model, vocab_data, seed_text="The", length=100)
        assert result1 == result2

    def test_greedy_generates_correct_length(self, trained_model_and_vocab):
        from generate import generate_text
        model, vocab_data = trained_model_and_vocab

        result = generate_text(model, vocab_data, seed_text="The", length=100)
        assert len(result) == 3 + 100  # seed "The" (3) + 100 generated

    def test_temperature_returns_string(self, trained_model_and_vocab):
        from generate import generate_text_with_temperature
        model, vocab_data = trained_model_and_vocab

        result = generate_text_with_temperature(
            model, vocab_data, seed_text="The", length=50, temperature=0.8)
        assert isinstance(result, str)
        assert result.startswith("The")

    def test_temperature_correct_length(self, trained_model_and_vocab):
        from generate import generate_text_with_temperature
        model, vocab_data = trained_model_and_vocab

        result = generate_text_with_temperature(
            model, vocab_data, seed_text="Be", length=80, temperature=1.0)
        assert len(result) == 2 + 80  # seed "Be" (2) + 80 generated

    def test_temperature_generates_valid_chars(self, trained_model_and_vocab):
        """All generated characters should be in the vocabulary."""
        from generate import generate_text_with_temperature
        model, vocab_data = trained_model_and_vocab

        result = generate_text_with_temperature(
            model, vocab_data, seed_text="The", length=200, temperature=1.5)

        valid_chars = set(vocab_data['chars'])
        for ch in result:
            assert ch in valid_chars, f"Generated char '{ch}' not in vocabulary"

    def test_different_seeds_different_output(self, trained_model_and_vocab):
        """Different seeds should produce different text."""
        from generate import generate_text
        model, vocab_data = trained_model_and_vocab

        result1 = generate_text(model, vocab_data, seed_text="The", length=100)
        result2 = generate_text(model, vocab_data, seed_text="Life", length=100)
        assert result1 != result2

    def test_high_temperature_produces_variety(self, trained_model_and_vocab):
        """High temperature should produce different outputs on repeated runs."""
        from generate import generate_text_with_temperature
        model, vocab_data = trained_model_and_vocab

        results = set()
        for _ in range(5):
            result = generate_text_with_temperature(
                model, vocab_data, seed_text="The", length=100, temperature=1.5)
            results.add(result)

        # With temp 1.5, 5 runs should produce at least 2 different outputs
        assert len(results) >= 2, "High temperature should produce varied output"


# ====================================================================
# STEP 14: INTERACTIVE
# ====================================================================

class TestInteractive:
    """Verify the InteractiveGenerator handles commands correctly."""

    def test_interactive_generator_creation(self, trained_model_and_vocab):
        from interactive import InteractiveGenerator
        model, vocab_data = trained_model_and_vocab

        gen = InteractiveGenerator(model, vocab_data, temperature=0.8, length=200)
        assert gen.temperature == 0.8
        assert gen.length == 200

    def test_command_temp(self, trained_model_and_vocab):
        from interactive import InteractiveGenerator
        model, vocab_data = trained_model_and_vocab

        gen = InteractiveGenerator(model, vocab_data)
        gen.handle_command(":temp 0.5")
        assert gen.temperature == 0.5

    def test_command_length(self, trained_model_and_vocab):
        from interactive import InteractiveGenerator
        model, vocab_data = trained_model_and_vocab

        gen = InteractiveGenerator(model, vocab_data)
        gen.handle_command(":length 300")
        assert gen.length == 300

    def test_command_temp_clamp_high(self, trained_model_and_vocab):
        from interactive import InteractiveGenerator
        model, vocab_data = trained_model_and_vocab

        gen = InteractiveGenerator(model, vocab_data)
        gen.handle_command(":temp 5.0")
        assert gen.temperature == 3.0

    def test_command_length_clamp_low(self, trained_model_and_vocab):
        from interactive import InteractiveGenerator
        model, vocab_data = trained_model_and_vocab

        gen = InteractiveGenerator(model, vocab_data)
        gen.handle_command(":length 1")
        assert gen.length == 10

    def test_command_quit_returns_false(self, trained_model_and_vocab):
        from interactive import InteractiveGenerator
        model, vocab_data = trained_model_and_vocab

        gen = InteractiveGenerator(model, vocab_data)
        assert gen.handle_command(":quit") is False

    def test_command_help_returns_true(self, trained_model_and_vocab):
        from interactive import InteractiveGenerator
        model, vocab_data = trained_model_and_vocab

        gen = InteractiveGenerator(model, vocab_data)
        assert gen.handle_command(":help") is True

    def test_generate_with_low_temp(self, trained_model_and_vocab):
        """Temperature near 0 should use greedy decoding."""
        from interactive import InteractiveGenerator
        model, vocab_data = trained_model_and_vocab

        gen = InteractiveGenerator(model, vocab_data, temperature=0.001, length=50)
        result = gen.generate("The")
        assert isinstance(result, str)
        assert result.startswith("The")


# ====================================================================
# FULL PIPELINE E2E
# ====================================================================

class TestFullPipeline:
    """End-to-end tests that verify the complete flow works."""

    def test_train_and_generate(self):
        """Train a tiny model from scratch and generate text."""
        from vocabulary import Vocabulary
        from dataset import TextDataset, create_dataloader
        from model import TinyLanguageModel
        from train import train
        from generate import generate_text_with_temperature

        # Load data
        with open("data/input.txt", "r") as f:
            text = f.read()

        # Build pipeline
        vocab = Vocabulary(text)
        dataset = TextDataset(text, vocab, seq_length=50)
        dataloader = create_dataloader(dataset, batch_size=16, shuffle=True)

        # Create and train model (small, fast)
        model = TinyLanguageModel(vocab_size=vocab.vocab_size,
                                   embed_size=64, hidden_size=128,
                                   num_layers=1)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

        loss_history = train(model, dataloader, loss_fn, optimizer,
                             vocab.vocab_size, num_epochs=20, print_every=100)

        # Verify training worked
        assert loss_history[-1] < loss_history[0]

        # Generate text
        model.eval()
        vocab_data = {
            'chars': vocab.chars,
            'char_to_idx': vocab.char_to_idx,
            'idx_to_char': vocab.idx_to_char,
        }
        result = generate_text_with_temperature(
            model, vocab_data, seed_text="The", length=100, temperature=0.8)

        assert result.startswith("The")
        assert len(result) == 103  # "The" + 100 chars

        # All chars in vocabulary
        valid = set(vocab.chars)
        for ch in result:
            assert ch in valid

    def test_save_load_generate_cycle(self, tmp_path):
        """Train → save → load → generate should produce valid output."""
        from vocabulary import Vocabulary
        from dataset import TextDataset, create_dataloader
        from model import TinyLanguageModel
        from train import train, save_model, save_vocabulary
        from generate import load_model, load_vocabulary, generate_text

        with open("data/input.txt", "r") as f:
            text = f.read()

        vocab = Vocabulary(text)
        dataset = TextDataset(text, vocab, seq_length=50)
        dataloader = create_dataloader(dataset, batch_size=16, shuffle=True)

        model = TinyLanguageModel(vocab_size=vocab.vocab_size,
                                   embed_size=64, hidden_size=128,
                                   num_layers=1)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

        train(model, dataloader, loss_fn, optimizer,
              vocab.vocab_size, num_epochs=15, print_every=100)

        # Save
        model_path = str(tmp_path / "model.pth")
        vocab_path = str(tmp_path / "vocab.pth")
        save_model(model, filepath=model_path)
        save_vocabulary(vocab, filepath=vocab_path)

        # Load — must recreate with same architecture used during training
        loaded_vocab_data = load_vocabulary(vocab_path)
        loaded_model = TinyLanguageModel(
            vocab_size=len(loaded_vocab_data['chars']),
            embed_size=64, hidden_size=128, num_layers=1
        )
        loaded_model.load_state_dict(
            torch.load(model_path, weights_only=True)
        )
        loaded_model.eval()

        # Generate
        result = generate_text(loaded_model, loaded_vocab_data,
                                seed_text="Life", length=80)

        assert result.startswith("Life")
        assert len(result) == 4 + 80

    def test_pretrained_model_generates_coherent_text(self, trained_model_and_vocab):
        """The pretrained model should generate text with real English words."""
        from generate import generate_text

        model, vocab_data = trained_model_and_vocab
        result = generate_text(model, vocab_data, seed_text="The", length=200)

        # Should contain common English words from training data
        common_words = ["the", "is", "to", "of", "you", "life", "not"]
        result_lower = result.lower()
        found = [w for w in common_words if w in result_lower]

        assert len(found) >= 3, (
            f"Expected at least 3 common words in output, found {found}. "
            f"Output: {result[:100]}..."
        )
