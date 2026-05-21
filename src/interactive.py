"""
interactive.py — Chat-like interactive prompt for text generation.

PURPOSE:
    This wraps the trained model in a user-friendly loop where you can:
    - Type any seed text and see what the model generates
    - Adjust temperature on the fly
    - Change the generation length
    - Experiment freely without editing code

    It's the "front end" of your AI — the part a user interacts with.
    Everything behind it (model, vocabulary, generation) was built in
    previous steps.

HOW IT WORKS:
    ┌──────────────────────────────────────────────────────────┐
    │                                                          │
    │  User types seed text                                    │
    │       │                                                  │
    │       ▼                                                  │
    │  generate_text_with_temperature(seed, temp, length)      │
    │       │                                                  │
    │       ▼                                                  │
    │  Print generated text                                    │
    │       │                                                  │
    │       ▼                                                  │
    │  Wait for next input (or quit)                           │
    │                                                          │
    └──────────────────────────────────────────────────────────┘

COMMANDS:
    In addition to typing seed text, the user can type special commands:

    :temp 0.5    → change temperature to 0.5
    :length 300  → change generation length to 300 characters
    :settings    → show current temperature and length
    :help        → show available commands
    :quit        → exit the program

    Commands start with ':' to distinguish them from seed text.

WHAT THIS FILE PROVIDES:
    1. InteractiveGenerator class — manages state and the input loop
    2. main() — loads model + vocab and starts the interactive session

INPUT:  outputs/model.pth, outputs/vocab.pth, user keyboard input
OUTPUT: Generated text printed to console

Usage:
    PYTHONPATH=src python src/interactive.py
"""

import os
import sys
import torch

from model import TinyLanguageModel
from generate import (
    load_model,
    load_vocabulary,
    generate_text,
    generate_text_with_temperature,
)


class InteractiveGenerator:
    """
    Manages an interactive text generation session.

    This class holds the model, vocabulary, and current settings
    (temperature, length), and runs the input loop where the user
    types seed text and sees generated output.

    WHY A CLASS?
        We need to keep track of STATE between user inputs:
        - The model (loaded once, used every generation)
        - The vocabulary (loaded once, used for encode/decode)
        - Current temperature (can be changed with :temp command)
        - Current length (can be changed with :length command)

        A class bundles this state together cleanly. Each method
        can access self.model, self.temperature, etc. without
        passing them around as function arguments.

    ATTRIBUTES:
        model (TinyLanguageModel): The trained model in eval mode
        vocab_data (dict):         Vocabulary mappings
        temperature (float):       Current temperature (default 0.8)
        length (int):              Characters to generate (default 200)
    """

    def __init__(self, model, vocab_data, temperature=0.8, length=200):
        """
        Initialize the interactive generator.

        PARAMETERS:
            model (TinyLanguageModel): Trained model (already in eval mode)
            vocab_data (dict):         Vocabulary with char_to_idx, idx_to_char
            temperature (float):       Starting temperature (0.8 is a good default)
            length (int):              Starting generation length in characters
        """

        self.model = model
        self.vocab_data = vocab_data
        self.temperature = temperature
        self.length = length

    def generate(self, seed_text):
        """
        Generate text from a seed using current settings.

        If temperature is very low (< 0.01), we use greedy decoding
        instead — this avoids division-by-near-zero issues and gives
        truly deterministic output.

        PARAMETERS:
            seed_text (str): The text to continue from

        RETURNS:
            str: The seed + generated text
        """

        # ---- Choose generation method based on temperature ----
        # Near-zero temperature is effectively greedy — use the
        # dedicated greedy function to avoid numerical issues.
        if self.temperature < 0.01:
            return generate_text(
                self.model, self.vocab_data,
                seed_text=seed_text, length=self.length
            )
        else:
            return generate_text_with_temperature(
                self.model, self.vocab_data,
                seed_text=seed_text, length=self.length,
                temperature=self.temperature
            )

    def handle_command(self, command):
        """
        Process a ':' command from the user.

        Commands allow changing settings without restarting the program.
        This is much more convenient than editing code and re-running.

        SUPPORTED COMMANDS:
            :temp <value>    — change temperature (0.1 to 3.0)
            :length <value>  — change generation length (10 to 1000)
            :settings        — show current settings
            :help            — show available commands
            :quit / :exit    — exit the program

        PARAMETERS:
            command (str): The full command string (e.g., ":temp 0.5")

        RETURNS:
            bool: True if the program should continue, False to quit
        """

        # ---- Split command into parts ----
        # ":temp 0.5" → ["temp", "0.5"]
        # ":quit"     → ["quit"]
        parts = command[1:].strip().split()

        if not parts:
            print("  Unknown command. Type :help for available commands.")
            return True

        cmd = parts[0].lower()

        # ---- :quit or :exit ----
        if cmd in ('quit', 'exit', 'q'):
            return False

        # ---- :help ----
        elif cmd == 'help':
            self.print_help()

        # ---- :settings ----
        elif cmd == 'settings':
            self.print_settings()

        # ---- :temp <value> ----
        elif cmd == 'temp':
            if len(parts) < 2:
                print(f"  Current temperature: {self.temperature}")
                print(f"  Usage: :temp 0.8")
                return True

            try:
                new_temp = float(parts[1])
                if new_temp < 0.01:
                    new_temp = 0.0
                    print(f"  Temperature set to 0 (greedy mode)")
                elif new_temp > 3.0:
                    new_temp = 3.0
                    print(f"  Temperature clamped to 3.0 (maximum)")
                else:
                    print(f"  Temperature set to {new_temp}")

                self.temperature = new_temp

            except ValueError:
                print(f"  Invalid temperature: '{parts[1]}'. Use a number like 0.8")

        # ---- :length <value> ----
        elif cmd == 'length':
            if len(parts) < 2:
                print(f"  Current length: {self.length}")
                print(f"  Usage: :length 300")
                return True

            try:
                new_length = int(parts[1])
                if new_length < 10:
                    new_length = 10
                    print(f"  Length clamped to minimum: 10")
                elif new_length > 1000:
                    new_length = 1000
                    print(f"  Length clamped to maximum: 1000")
                else:
                    print(f"  Length set to {new_length}")

                self.length = new_length

            except ValueError:
                print(f"  Invalid length: '{parts[1]}'. Use a whole number like 300")

        # ---- Unknown command ----
        else:
            print(f"  Unknown command: ':{cmd}'. Type :help for available commands.")

        return True

    def print_help(self):
        """Print the list of available commands."""

        print("""
  Available commands:
    :temp <value>    Set temperature (0.1-3.0). Controls creativity.
                     Low (0.3) = safe, High (1.5) = creative
    :length <value>  Set generation length (10-1000 characters)
    :settings        Show current temperature and length
    :help            Show this help message
    :quit            Exit the program

  Or just type any text as a seed for generation!
  Examples: "The", "Life is", "In the beginning"
""")

    def print_settings(self):
        """Print the current generation settings."""

        # ---- Describe the temperature level ----
        if self.temperature < 0.01:
            temp_desc = "greedy (deterministic)"
        elif self.temperature <= 0.3:
            temp_desc = "very cautious"
        elif self.temperature <= 0.5:
            temp_desc = "conservative"
        elif self.temperature <= 0.8:
            temp_desc = "slightly creative"
        elif self.temperature <= 1.0:
            temp_desc = "balanced"
        elif self.temperature <= 1.5:
            temp_desc = "creative"
        else:
            temp_desc = "wild"

        print(f"""
  Current settings:
    Temperature : {self.temperature} ({temp_desc})
    Length      : {self.length} characters
""")

    def run(self):
        """
        Run the interactive generation loop.

        This is the main loop that:
        1. Shows a prompt (>>)
        2. Reads user input
        3. If it starts with ':', handle it as a command
        4. Otherwise, use it as seed text for generation
        5. Repeat until the user types :quit

        WHY input() WITH try/except?
            input() reads one line from the keyboard.
            If the user presses Ctrl+C (KeyboardInterrupt) or
            Ctrl+D (EOFError), we catch those and exit gracefully
            instead of showing an ugly error traceback.
        """

        self.print_welcome()

        while True:
            try:
                # ---- Show prompt and read input ----
                # The ">>" prompt mimics a chat interface.
                # .strip() removes leading/trailing whitespace.
                user_input = input("\n>> ").strip()

                # ---- Skip empty input ----
                if not user_input:
                    continue

                # ---- Handle commands ----
                if user_input.startswith(':'):
                    should_continue = self.handle_command(user_input)
                    if not should_continue:
                        print("\n  Goodbye! Happy generating.")
                        break
                    continue

                # ---- Validate seed text ----
                # Check that all characters in the seed exist in the vocabulary.
                # If the user types a character the model doesn't know (like '!'),
                # we'd get a KeyError. Better to catch it early with a nice message.
                char_to_idx = self.vocab_data['char_to_idx']
                unknown_chars = [ch for ch in user_input if ch not in char_to_idx]

                if unknown_chars:
                    print(f"  Unknown characters: {unknown_chars}")
                    print(f"  The model only knows these characters:")
                    print(f"  {self.vocab_data['chars']}")
                    continue

                # ---- Generate text ----
                print()
                generated = self.generate(user_input)
                print(generated)

            except KeyboardInterrupt:
                # ---- Handle Ctrl+C ----
                # User pressed Ctrl+C — exit gracefully.
                print("\n\n  Interrupted. Goodbye!")
                break

            except EOFError:
                # ---- Handle Ctrl+D ----
                # User pressed Ctrl+D (end of input) — exit gracefully.
                print("\n\n  Goodbye!")
                break

    def print_welcome(self):
        """Print the welcome message when the interactive session starts."""

        print("""
╔══════════════════════════════════════════════════════════╗
║           SimpleAI — Interactive Text Generator          ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  Type any text and the model will continue it.           ║
║  Type :help for commands, :quit to exit.                 ║
║                                                          ║
║  Current settings:                                       ║
║    Temperature: {temp:<5} ({desc})     ║
║    Length:      {length:<5} characters                        ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""".format(
            temp=self.temperature,
            desc=self._temp_description().ljust(22),
            length=self.length
        ))

    def _temp_description(self):
        """Return a short description of the current temperature."""

        if self.temperature < 0.01:
            return "greedy"
        elif self.temperature <= 0.3:
            return "very cautious"
        elif self.temperature <= 0.5:
            return "conservative"
        elif self.temperature <= 0.8:
            return "slightly creative"
        elif self.temperature <= 1.0:
            return "balanced"
        elif self.temperature <= 1.5:
            return "creative"
        else:
            return "wild"


def main():
    """
    Main function — load model and start interactive session.

    FLOW:
        1. Load model from outputs/model.pth
        2. Load vocabulary from outputs/vocab.pth
        3. Create InteractiveGenerator with default settings
        4. Run the interactive loop
    """

    # ==================================================================
    # SETUP
    # ==================================================================
    print("=" * 60)
    print("STEP 14: INTERACTIVE TEXT GENERATION")
    print("=" * 60)
    print()

    # ---- Check that required files exist ----
    model_path = "outputs/model.pth"
    vocab_path = "outputs/vocab.pth"

    if not os.path.exists(model_path):
        print(f"  ERROR: {model_path} not found!")
        print(f"  Run train.py first (Step 10) to train and save the model.")
        return

    if not os.path.exists(vocab_path):
        print(f"  ERROR: {vocab_path} not found!")
        print(f"  Run train.py first (Step 10) to save the vocabulary.")
        return

    # ---- Load the trained model ----
    print("Loading trained model...")
    vocab_data = load_vocabulary(vocab_path)
    vocab_size = len(vocab_data['chars'])
    model = load_model(model_path, vocab_size=vocab_size)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {total_params:,} parameters")
    print(f"  Vocabulary: {vocab_size} characters")

    # ==================================================================
    # START INTERACTIVE SESSION
    # ==================================================================
    # Create the interactive generator with sensible defaults:
    # - temperature=0.8: slightly creative, good balance
    # - length=200: enough to see a few sentences
    generator = InteractiveGenerator(
        model, vocab_data,
        temperature=0.8, length=200
    )

    generator.run()


# ======================================================================
# ENTRY POINT
# ======================================================================
if __name__ == "__main__":
    main()
