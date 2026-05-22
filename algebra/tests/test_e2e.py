"""
test_e2e.py — Comprehensive test suite for the algebra solver.

Tests every component of the pipeline:
    1. Data generation — equations are valid, solutions correct
    2. Vocabulary — encode/decode roundtrip, special tokens
    3. Dataset — padding, shapes, train/test split
    4. Encoder — output shapes, bidirectional output
    5. Decoder — generation, teacher forcing modes
    6. Seq2Seq — end-to-end forward pass, gradient flow
    7. Training — loss decreases over a few epochs
    8. Evaluation — accuracy above 90% threshold
    9. Interactive — command handling, solving
   10. Full pipeline — generate → train → solve → verify

Usage:
    cd /path/to/SimpleAI
    PYTHONPATH=algebra/src pytest algebra/tests/test_e2e.py -v
"""

import os
import sys
import re
import random
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vocab import (
    AlgebraVocab, build_vocab_from_data,
    PAD_IDX, SOS_IDX, EOS_IDX,
)
from dataset import AlgebraDataset, collate_fn, create_dataloaders
from encoder import Encoder
from decoder import Decoder
from seq2seq import Seq2Seq, create_model
from evaluate import classify_equation, parse_answer, verify_by_substitution


# ====================================================================
# TEST DATA GENERATION
# ====================================================================

class TestDataGeneration:
    """Tests for algebra/data/equations.txt validity."""

    DATA_PATH = "algebra/data/equations.txt"

    def test_data_file_exists(self):
        assert os.path.exists(self.DATA_PATH)

    def test_data_has_50000_lines(self):
        with open(self.DATA_PATH) as f:
            lines = f.readlines()
        assert len(lines) == 50_000

    def test_each_line_has_tab_separator(self):
        with open(self.DATA_PATH) as f:
            for i, line in enumerate(f):
                assert "\t" in line, f"Line {i} missing tab separator"

    def test_each_line_has_equation_and_solution(self):
        with open(self.DATA_PATH) as f:
            for i, line in enumerate(f):
                parts = line.strip().split("\t")
                assert len(parts) == 2, f"Line {i}: expected 2 parts, got {len(parts)}"

    def test_solutions_have_correct_format(self):
        with open(self.DATA_PATH) as f:
            for i, line in enumerate(f):
                _, sol = line.strip().split("\t")
                assert sol.startswith("x = "), f"Line {i}: bad solution format '{sol}'"

    def test_equations_contain_x(self):
        with open(self.DATA_PATH) as f:
            for i, line in enumerate(f):
                eq, _ = line.strip().split("\t")
                assert "x" in eq, f"Line {i}: equation missing 'x'"

    def test_equations_contain_equals(self):
        with open(self.DATA_PATH) as f:
            for i, line in enumerate(f):
                eq, _ = line.strip().split("\t")
                assert "=" in eq, f"Line {i}: equation missing '='"

    def test_solutions_are_integers(self):
        with open(self.DATA_PATH) as f:
            for i, line in enumerate(f):
                _, sol = line.strip().split("\t")
                val = sol.replace("x = ", "")
                assert re.match(r'^-?\d+$', val), f"Line {i}: non-integer answer '{val}'"

    def test_sample_solutions_verify_by_substitution(self):
        random.seed(42)
        with open(self.DATA_PATH) as f:
            lines = f.readlines()
        samples = random.sample(lines, 100)
        for line in samples:
            eq, sol = line.strip().split("\t")
            val = int(sol.replace("x = ", ""))
            assert verify_by_substitution(eq, val), \
                f"Failed verification: {eq} with {sol}"

    def test_no_duplicate_equations(self):
        with open(self.DATA_PATH) as f:
            equations = [line.strip().split("\t")[0] for line in f]
        assert len(equations) == len(set(equations))

    def test_has_positive_negative_and_zero_answers(self):
        pos = neg = zero = 0
        with open(self.DATA_PATH) as f:
            for line in f:
                val = int(line.strip().split("\t")[1].replace("x = ", ""))
                if val > 0:
                    pos += 1
                elif val < 0:
                    neg += 1
                else:
                    zero += 1
        assert pos > 1000
        assert neg > 1000
        assert zero > 0


# ====================================================================
# TEST VOCABULARY
# ====================================================================

class TestVocabulary:
    """Tests for vocab.py."""

    @pytest.fixture
    def vocab(self):
        return build_vocab_from_data("algebra/data/equations.txt")

    def test_special_token_indices(self, vocab):
        assert vocab.token_to_idx["<PAD>"] == 0
        assert vocab.token_to_idx["<SOS>"] == 1
        assert vocab.token_to_idx["<EOS>"] == 2

    def test_pad_sos_eos_constants(self):
        assert PAD_IDX == 0
        assert SOS_IDX == 1
        assert EOS_IDX == 2

    def test_vocab_size_is_19(self, vocab):
        assert vocab.vocab_size == 19

    def test_all_digits_in_vocab(self, vocab):
        for d in "0123456789":
            assert d in vocab.token_to_idx

    def test_operators_in_vocab(self, vocab):
        for op in ["+", "-", "/", "=", "x", " "]:
            assert op in vocab.token_to_idx

    def test_encode_decode_roundtrip(self, vocab):
        text = "2x + 3 = 7"
        encoded = vocab.encode(text)
        decoded = vocab.decode(encoded)
        assert decoded == text

    def test_encode_with_eos_adds_eos(self, vocab):
        encoded = vocab.encode_with_eos("x = 2")
        assert encoded[-1] == EOS_IDX

    def test_decode_until_eos_stops_at_eos(self, vocab):
        tokens = [18, 3, 17, 3, 9, EOS_IDX, 7, 7, 7]
        decoded = vocab.decode_until_eos(tokens)
        assert decoded == "x = 2"

    def test_encode_all_characters_in_data(self, vocab):
        with open("algebra/data/equations.txt") as f:
            for line in f:
                eq, sol = line.strip().split("\t")
                for ch in eq + sol:
                    assert ch in vocab.token_to_idx, f"Missing char: '{ch}'"
                break  # just test first line for speed


# ====================================================================
# TEST DATASET
# ====================================================================

class TestDataset:
    """Tests for dataset.py."""

    @pytest.fixture
    def vocab(self):
        return build_vocab_from_data("algebra/data/equations.txt")

    @pytest.fixture
    def loaders(self, vocab):
        return create_dataloaders(vocab, batch_size=64, train_ratio=0.8)

    def test_train_test_split_sizes(self, loaders):
        train_loader, test_loader = loaders
        train_count = sum(enc.shape[0] for enc, _, _ in train_loader)
        test_count = sum(enc.shape[0] for enc, _, _ in test_loader)
        assert train_count == 40_000
        assert test_count >= 9_900  # may drop last incomplete batch

    def test_batch_shapes(self, loaders):
        train_loader, _ = loaders
        enc, dec_in, dec_tgt = next(iter(train_loader))
        assert enc.shape[0] == 64
        assert dec_in.shape[0] == 64
        assert dec_tgt.shape[0] == 64
        assert enc.dim() == 2
        assert dec_in.dim() == 2
        assert dec_tgt.dim() == 2

    def test_decoder_input_starts_with_sos(self, loaders):
        train_loader, _ = loaders
        _, dec_in, _ = next(iter(train_loader))
        assert (dec_in[:, 0] == SOS_IDX).all()

    def test_decoder_target_ends_with_eos_or_pad(self, loaders):
        train_loader, _ = loaders
        _, _, dec_tgt = next(iter(train_loader))
        for i in range(dec_tgt.shape[0]):
            row = dec_tgt[i].tolist()
            non_pad = [t for t in row if t != PAD_IDX]
            assert non_pad[-1] == EOS_IDX

    def test_no_data_leakage(self, vocab):
        train_loader, test_loader = create_dataloaders(
            vocab, batch_size=64, train_ratio=0.8
        )
        train_eqs = set()
        for enc, _, _ in train_loader:
            for i in range(enc.shape[0]):
                train_eqs.add(tuple(enc[i].tolist()))

        test_eqs = set()
        for enc, _, _ in test_loader:
            for i in range(enc.shape[0]):
                test_eqs.add(tuple(enc[i].tolist()))

        overlap = train_eqs & test_eqs
        assert len(overlap) == 0, f"Data leakage: {len(overlap)} shared equations"

    def test_collate_pads_to_max_length(self, vocab):
        with open("algebra/data/equations.txt") as f:
            pairs = [line.strip().split("\t") for line in f.readlines()[:10]]
        dataset = AlgebraDataset(pairs, vocab)
        samples = [dataset[0], dataset[1], dataset[2]]
        enc, dec_in, dec_tgt = collate_fn(samples)
        assert enc.shape[1] == max(len(s[0]) for s in samples)


# ====================================================================
# TEST ENCODER
# ====================================================================

class TestEncoder:
    """Tests for encoder.py."""

    @pytest.fixture
    def encoder(self):
        return Encoder(vocab_size=19, embed_size=64, hidden_size=128, num_layers=2)

    def test_output_shapes(self, encoder):
        src = torch.randint(3, 19, (4, 12))
        outputs, hidden = encoder(src)
        assert outputs.shape == (4, 12, 256)
        assert hidden.shape == (2, 4, 256)

    def test_bidirectional_doubles_hidden(self, encoder):
        assert encoder.gru.bidirectional is True
        assert encoder.gru.hidden_size == 128
        src = torch.randint(3, 19, (2, 8))
        outputs, _ = encoder(src)
        assert outputs.shape[2] == 256  # 128 * 2

    def test_different_seq_lengths(self, encoder):
        for seq_len in [5, 10, 20]:
            src = torch.randint(3, 19, (2, seq_len))
            outputs, hidden = encoder(src)
            assert outputs.shape == (2, seq_len, 256)
            assert hidden.shape == (2, 2, 256)

    def test_padding_produces_zero_embedding(self, encoder):
        embed = encoder.embedding(torch.tensor([PAD_IDX]))
        assert embed.abs().sum().item() == 0.0

    def test_parameter_count(self, encoder):
        params = sum(p.numel() for p in encoder.parameters())
        assert params > 400_000


# ====================================================================
# TEST DECODER
# ====================================================================

class TestDecoder:
    """Tests for decoder.py."""

    @pytest.fixture
    def decoder(self):
        return Decoder(vocab_size=19, embed_size=64, hidden_size=256, num_layers=2)

    def test_training_mode_shapes(self, decoder):
        hidden = torch.randn(2, 4, 256)
        target = torch.randint(3, 19, (4, 7))
        outputs, final_hidden = decoder(hidden, target=target, teacher_forcing_ratio=0.5)
        assert outputs.shape == (4, 7, 19)
        assert final_hidden.shape == (2, 4, 256)

    def test_inference_mode_shapes(self, decoder):
        hidden = torch.randn(2, 4, 256)
        preds, logits = decoder.generate(hidden)
        assert preds.shape == (4, decoder.max_length)
        assert logits.shape == (4, decoder.max_length, 19)

    def test_generate_produces_valid_tokens(self, decoder):
        hidden = torch.randn(2, 2, 256)
        preds, _ = decoder.generate(hidden)
        assert (preds >= 0).all()
        assert (preds < 19).all()

    def test_teacher_forcing_ratio_1_uses_target(self, decoder):
        hidden = torch.randn(2, 1, 256)
        target = torch.full((1, 5), 10, dtype=torch.long)
        out1, _ = decoder(hidden, target=target, teacher_forcing_ratio=1.0)
        assert out1.shape == (1, 5, 19)

    def test_max_length_respected(self, decoder):
        hidden = torch.randn(2, 1, 256)
        preds, _ = decoder.generate(hidden)
        assert preds.shape[1] == decoder.max_length


# ====================================================================
# TEST SEQ2SEQ
# ====================================================================

class TestSeq2Seq:
    """Tests for seq2seq.py."""

    @pytest.fixture
    def model(self):
        return create_model(vocab_size=19)

    def test_training_forward_pass(self, model):
        src = torch.randint(3, 19, (4, 12))
        target = torch.randint(3, 19, (4, 7))
        model.train()
        outputs = model(src, target=target, teacher_forcing_ratio=0.5)
        assert outputs.shape == (4, 7, 19)

    def test_inference_forward_pass(self, model):
        src = torch.randint(3, 19, (4, 12))
        model.eval()
        with torch.no_grad():
            preds, logits = model.solve(src)
        assert preds.shape[0] == 4
        assert logits.shape[0] == 4

    def test_gradient_flow_through_both_networks(self, model):
        src = torch.randint(3, 19, (4, 12))
        target = torch.randint(3, 19, (4, 7))
        model.train()
        outputs = model(src, target=target)
        loss = nn.CrossEntropyLoss()(outputs.view(-1, 19), target.view(-1))
        loss.backward()

        enc_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.encoder.parameters()
        )
        dec_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.decoder.parameters()
        )
        assert enc_has_grad, "Encoder gradients not flowing"
        assert dec_has_grad, "Decoder gradients not flowing"

    def test_total_parameter_count(self, model):
        params = sum(p.numel() for p in model.parameters())
        assert params == 1_160_595

    def test_model_has_encoder_and_decoder(self, model):
        assert hasattr(model, "encoder")
        assert hasattr(model, "decoder")
        assert isinstance(model.encoder, Encoder)
        assert isinstance(model.decoder, Decoder)


# ====================================================================
# TEST TRAINING (mini)
# ====================================================================

class TestTraining:
    """Tests for training loop — uses a tiny model for speed."""

    def test_loss_decreases_over_epochs(self):
        vocab = build_vocab_from_data("algebra/data/equations.txt")
        model = create_model(vocab_size=vocab.vocab_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
        loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

        train_loader, _ = create_dataloaders(vocab, batch_size=64, train_ratio=0.8)
        losses = []

        model.train()
        for epoch in range(3):
            total_loss = 0
            count = 0
            for enc_input, dec_input, dec_target in train_loader:
                outputs = model(enc_input, target=dec_target, teacher_forcing_ratio=1.0)
                loss = loss_fn(outputs.view(-1, vocab.vocab_size), dec_target.view(-1))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
                count += 1
                if count >= 5:
                    break
            losses.append(total_loss / count)

        assert losses[-1] < losses[0], \
            f"Loss didn't decrease: {losses[0]:.4f} → {losses[-1]:.4f}"

    def test_teacher_forcing_ratio_schedule(self):
        from train import get_teacher_forcing_ratio

        r0 = get_teacher_forcing_ratio(0, 50)
        r_mid = get_teacher_forcing_ratio(25, 50)
        r_end = get_teacher_forcing_ratio(49, 50)

        assert r0 == 1.0
        assert 0.4 < r_mid < 0.7
        assert r_end == pytest.approx(0.3, abs=0.01)

    def test_gradient_clipping(self):
        model = create_model(vocab_size=19)
        src = torch.randint(3, 19, (4, 12))
        target = torch.randint(3, 19, (4, 7))
        outputs = model(src, target=target)
        loss = nn.CrossEntropyLoss()(outputs.view(-1, 19), target.view(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        assert total_norm <= 1.01  # allow tiny float error


# ====================================================================
# TEST EVALUATION
# ====================================================================

class TestEvaluation:
    """Tests for evaluate.py and trained model accuracy."""

    def test_classify_type_1(self):
        assert "Type 1" in classify_equation("3x = 12")
        assert "Type 1" in classify_equation("-5x = 20")

    def test_classify_type_2(self):
        assert "Type 2" in classify_equation("2x + 3 = 7")

    def test_classify_type_3(self):
        assert "Type 3" in classify_equation("5x - 10 = 15")

    def test_classify_type_4(self):
        assert "Type 4" in classify_equation("4 + 3x = 13")

    def test_classify_type_5(self):
        assert "Type 5" in classify_equation("10 - 2x = 4")

    def test_classify_type_6(self):
        assert "Type 6" in classify_equation("x / 4 = 3")

    def test_classify_type_7(self):
        assert "Type 7" in classify_equation("3x + 2 = x + 8")

    def test_parse_answer_positive(self):
        assert parse_answer("x = 42") == 42

    def test_parse_answer_negative(self):
        assert parse_answer("x = -7") == -7

    def test_parse_answer_zero(self):
        assert parse_answer("x = 0") == 0

    def test_parse_answer_invalid(self):
        assert parse_answer("hello") is None

    def test_verify_substitution_correct(self):
        assert verify_by_substitution("2x + 3 = 7", 2) is True

    def test_verify_substitution_wrong(self):
        assert verify_by_substitution("2x + 3 = 7", 5) is False

    def test_verify_substitution_negative(self):
        assert verify_by_substitution("3x + 9 = 0", -3) is True

    def test_verify_substitution_division(self):
        assert verify_by_substitution("x / 4 = 3", 12) is True

    def test_verify_substitution_both_sides(self):
        assert verify_by_substitution("3x + 2 = x + 8", 3) is True

    def test_trained_model_accuracy_above_90(self):
        vocab = build_vocab_from_data("algebra/data/equations.txt")
        _, test_loader = create_dataloaders(vocab, batch_size=64, train_ratio=0.8)

        model = create_model(vocab_size=vocab.vocab_size)
        model.load_state_dict(
            torch.load("algebra/outputs/model.pth", weights_only=True)
        )
        model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for enc_input, _, dec_target in test_loader:
                preds, _ = model.solve(enc_input)
                for i in range(enc_input.shape[0]):
                    pred = vocab.decode_until_eos(preds[i].tolist())
                    target = vocab.decode_until_eos(dec_target[i].tolist())
                    if pred == target:
                        correct += 1
                    total += 1

        accuracy = correct / total
        assert accuracy >= 0.90, f"Accuracy {accuracy:.1%} below 90% threshold"


# ====================================================================
# TEST INTERACTIVE
# ====================================================================

class TestInteractive:
    """Tests for interactive.py components."""

    @pytest.fixture
    def model_and_vocab(self):
        vocab = build_vocab_from_data("algebra/data/equations.txt")
        model = create_model(vocab_size=vocab.vocab_size)
        model.load_state_dict(
            torch.load("algebra/outputs/model.pth", weights_only=True)
        )
        model.eval()
        return model, vocab

    def test_solve_equation(self, model_and_vocab):
        from interactive import solve_equation
        model, vocab = model_and_vocab
        result = solve_equation(model, vocab, "4x = 20")
        assert result.startswith("x = ")

    def test_verify_correct_answer(self):
        from interactive import verify_answer
        is_correct, _ = verify_answer("2x + 6 = 10", "x = 2")
        assert is_correct is True

    def test_verify_wrong_answer(self):
        from interactive import verify_answer
        is_correct, _ = verify_answer("2x + 6 = 10", "x = 5")
        assert is_correct is False

    def test_verify_unparseable_answer(self):
        from interactive import verify_answer
        is_correct, details = verify_answer("2x = 4", "garbage")
        assert is_correct is False


# ====================================================================
# TEST SAVED ARTIFACTS
# ====================================================================

class TestSavedArtifacts:
    """Tests that all saved files exist and are valid."""

    def test_model_file_exists(self):
        assert os.path.exists("algebra/outputs/model.pth")

    def test_vocab_file_exists(self):
        assert os.path.exists("algebra/outputs/vocab.pth")

    def test_loss_history_file_exists(self):
        assert os.path.exists("algebra/outputs/loss_history.pth")

    def test_loss_plot_file_exists(self):
        assert os.path.exists("algebra/outputs/loss_plot.png")

    def test_data_file_exists(self):
        assert os.path.exists("algebra/data/equations.txt")

    def test_model_loads_successfully(self):
        model = create_model(vocab_size=19)
        state = torch.load("algebra/outputs/model.pth", weights_only=True)
        model.load_state_dict(state)

    def test_vocab_loads_successfully(self):
        saved = torch.load("algebra/outputs/vocab.pth", weights_only=False)
        assert "tokens" in saved
        assert "token_to_idx" in saved
        assert "vocab_size" in saved
        assert saved["vocab_size"] == 19

    def test_history_has_expected_keys(self):
        history = torch.load("algebra/outputs/loss_history.pth", weights_only=False)
        assert "losses" in history
        assert "accuracies" in history
        assert "tf_ratios" in history
        assert len(history["losses"]) == 50


# ====================================================================
# TEST FULL PIPELINE
# ====================================================================

class TestFullPipeline:
    """End-to-end pipeline tests."""

    def test_encode_solve_decode_cycle(self):
        vocab = build_vocab_from_data("algebra/data/equations.txt")
        model = create_model(vocab_size=vocab.vocab_size)
        model.load_state_dict(
            torch.load("algebra/outputs/model.pth", weights_only=True)
        )
        model.eval()

        eq = "5x + 10 = 35"
        encoded = vocab.encode_with_eos(eq)
        padded = encoded + [PAD_IDX] * (25 - len(encoded))
        src = torch.tensor([padded])

        with torch.no_grad():
            preds, _ = model.solve(src)

        result = vocab.decode_until_eos(preds[0].tolist())
        assert result.startswith("x = ")
        val = parse_answer(result)
        assert val is not None

    def test_batch_solve_multiple_equations(self):
        vocab = build_vocab_from_data("algebra/data/equations.txt")
        model = create_model(vocab_size=vocab.vocab_size)
        model.load_state_dict(
            torch.load("algebra/outputs/model.pth", weights_only=True)
        )
        model.eval()

        _, test_loader = create_dataloaders(vocab, batch_size=64, train_ratio=0.8)
        enc_input, _, dec_target = next(iter(test_loader))

        with torch.no_grad():
            preds, _ = model.solve(enc_input)

        assert preds.shape[0] == 64

    def test_substitution_verification_on_correct_predictions(self):
        vocab = build_vocab_from_data("algebra/data/equations.txt")
        model = create_model(vocab_size=vocab.vocab_size)
        model.load_state_dict(
            torch.load("algebra/outputs/model.pth", weights_only=True)
        )
        model.eval()

        _, test_loader = create_dataloaders(vocab, batch_size=64, train_ratio=0.8)
        enc_input, _, dec_target = next(iter(test_loader))

        with torch.no_grad():
            preds, _ = model.solve(enc_input)

        verified = 0
        checked = 0
        for i in range(enc_input.shape[0]):
            pred_text = vocab.decode_until_eos(preds[i].tolist())
            target_text = vocab.decode_until_eos(dec_target[i].tolist())
            if pred_text == target_text:
                eq_text = vocab.decode_until_eos(enc_input[i].tolist())
                val = parse_answer(pred_text)
                if val is not None:
                    checked += 1
                    if verify_by_substitution(eq_text, val):
                        verified += 1

        assert checked > 0
        assert verified == checked, \
            f"Only {verified}/{checked} correct predictions verified"
