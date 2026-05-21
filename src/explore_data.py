"""
explore_data.py — Look at our training data before feeding it to the AI.

PURPOSE:
    Before training any model, it's critical to understand your data.
    "Garbage in, garbage out" is a core principle in AI — if your data
    has problems (weird characters, encoding issues, too little text),
    the model will learn those problems.

    This script is your "data microscope". It answers:
    - How much text do we have?
    - What characters appear in it? (This defines the model's vocabulary)
    - What does a sample of the text look like?
    - Which characters are most common?

    This is a habit used by every AI/ML engineer: ALWAYS inspect your
    data first. Professional ML engineers spend ~80% of their time on
    data preparation and only ~20% on the actual model.

WHAT THIS SCRIPT DOES (FLOW):
    1. Opens data/input.txt and reads all text into memory
    2. Counts characters, words, and lines (dataset size)
    3. Finds every unique character (the "vocabulary")
    4. Shows first/last few lines (sanity check)
    5. Counts how often each character appears (frequency analysis)
    6. Summarizes what this means for training

INPUT:  data/input.txt (a text file with training data)
OUTPUT: Printed analysis — sizes, vocabulary, frequency chart

Usage:
    python src/explore_data.py
"""


def main():
    """
    Main function — loads the text file and runs all analysis steps.

    This function takes no arguments. It reads directly from the
    hardcoded filepath 'data/input.txt'. In a larger project, you'd
    pass the filepath as a command-line argument, but for learning
    purposes we keep it simple.

    FLOW:
        load_file → count_stats → find_vocabulary → preview → frequency → summary
    """

    # ==================================================================
    # STEP 1: LOAD THE TEXT FILE
    # ==================================================================
    # open() is Python's built-in function for reading files.
    #
    # The 'with' statement ensures the file is properly closed after
    # reading, even if an error occurs. Without 'with', you'd need to
    # manually call f.close() — forgetting to do so can cause data
    # corruption or resource leaks.
    #
    # Parameters:
    #   filepath    — path to the file relative to where you run the script
    #   "r"         — read mode (as opposed to "w" for write, "a" for append)
    #   encoding    — "utf-8" handles international characters (accents, etc.)
    #
    # f.read() loads the ENTIRE file into memory as one big string.
    # For our small ~6KB file this is fine. For gigabyte files, you'd
    # read line by line to avoid running out of memory.
    #
    # INPUT:  file path string "data/input.txt"
    # OUTPUT: 'text' variable containing the entire file as a string
    #
    # EXAMPLE:
    #   If input.txt contains:
    #       Hello world
    #       Goodbye world
    #   Then text = "Hello world\nGoodbye world\n"
    #   (note: \n represents a newline character)

    filepath = "data/input.txt"

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    print("=" * 50)
    print("DATA EXPLORATION")
    print("=" * 50)
    print(f"File            : {filepath}")
    print()

    # ==================================================================
    # STEP 2: BASIC STATISTICS
    # ==================================================================
    # These numbers tell us how "big" our dataset is.
    # More data = more patterns for the model to learn.
    #
    # For reference, here's how our dataset compares:
    #   Our dataset  : ~6,000 characters  (tiny — just for learning)
    #   A short book : ~500,000 characters
    #   Wikipedia    : ~20 billion characters
    #   GPT-4 data   : trillions of characters
    #
    # len(text) counts every character including spaces and newlines.
    #
    # text.strip() removes leading/trailing whitespace.
    # .split("\n") breaks the string at every newline into a list.
    # EXAMPLE:
    #   "Hello\nWorld\n".strip().split("\n") = ["Hello", "World"]
    #
    # text.split() (no argument) splits on ANY whitespace (spaces,
    # tabs, newlines) and ignores consecutive whitespace.
    # EXAMPLE:
    #   "Hello   World\nFoo".split() = ["Hello", "World", "Foo"]

    total_chars = len(text)
    lines = text.strip().split("\n")
    total_lines = len(lines)
    words = text.split()
    total_words = len(words)

    print("--- Size ---")
    print(f"Total characters: {total_chars}")
    print(f"Total words     : {total_words}")
    print(f"Total lines     : {total_lines}")
    print(f"Avg chars/line  : {total_chars // total_lines}")
    # Average characters per line gives us a sense of how long each
    # quote is. The model will learn to produce outputs of similar length.
    print()

    # ==================================================================
    # STEP 3: CHARACTER ANALYSIS (VOCABULARY)
    # ==================================================================
    # Find every UNIQUE character in the text. This is our "vocabulary"
    # — the complete alphabet that our model will know about.
    #
    # KEY CONCEPT: The model can ONLY learn characters it has seen.
    # If the character 'z' never appears in the training data, the
    # model literally doesn't know it exists. It's like learning a
    # language but never seeing certain letters.
    #
    # set(text) creates a set of unique characters.
    # EXAMPLE:
    #   set("hello") = {'h', 'e', 'l', 'o'}  (no duplicate 'l')
    #
    # sorted() puts them in alphabetical/ASCII order for readability.
    # EXAMPLE:
    #   sorted({'o', 'h', 'e', 'l'}) = ['e', 'h', 'l', 'o']
    #
    # INPUT:  the full text string
    # OUTPUT: a sorted list of unique characters, and the count (vocab_size)

    unique_chars = sorted(set(text))
    vocab_size = len(unique_chars)

    print("--- Character vocabulary ---")
    print(f"Unique characters: {vocab_size}")
    print()

    # Display each character with its index number.
    # repr() makes invisible characters visible:
    #   repr('\n') = "'\\n'"  (shows the newline as \n)
    #   repr(' ')  = "' '"    (shows the space explicitly)
    #   repr('a')  = "'a'"    (regular characters shown as-is)
    #
    # The index numbers here preview what Step 04 will do —
    # each character gets a unique number. That's how we turn
    # text into numbers the model can process.
    print("All characters found:")
    for i, char in enumerate(unique_chars):
        display = repr(char)
        print(f"  [{i:2d}] {display}")
    print()

    # ==================================================================
    # STEP 4: PREVIEW THE DATA
    # ==================================================================
    # Always eyeball a sample of your data. This simple habit catches
    # problems that statistics alone can't reveal:
    #   - Weird characters or encoding issues (e.g., Ã© instead of é)
    #   - Unexpected formatting (extra blank lines, HTML tags, etc.)
    #   - Data quality issues (nonsense text, repeated lines, etc.)
    #
    # We look at both the beginning AND end to check consistency.
    #
    # lines[:5] = first 5 elements of the list (Python slicing)
    # lines[-3:] = last 3 elements of the list

    print("--- First 5 lines ---")
    for i, line in enumerate(lines[:5]):
        print(f"  {i + 1}: {line}")
    print()

    print("--- Last 3 lines ---")
    for line in lines[-3:]:
        print(f"  {line}")
    print()

    # ==================================================================
    # STEP 5: CHARACTER FREQUENCY ANALYSIS
    # ==================================================================
    # Count how often each character appears. This matters because:
    #
    # 1. COMMON CHARACTERS are easier for the model to learn.
    #    If 'e' appears 654 times, the model gets 654 examples of
    #    when 'e' is the right prediction. If 'z' appears 3 times,
    #    the model barely learns about 'z'.
    #
    # 2. FREQUENCY DISTRIBUTION tells us about the text's nature.
    #    English text always has space as most common, then 'e', 't',
    #    'a', 'o'... This is called "letter frequency" and it's the
    #    same pattern used to crack codes in cryptography!
    #
    # 3. RARE CHARACTERS might cause problems. If a character appears
    #    only once or twice, the model may never learn to use it
    #    correctly. We might need to remove or replace such characters.
    #
    # HOW THE COUNTING WORKS:
    #   char_counts = {}           ← start with empty dictionary
    #   for each character in text:
    #       if character in dict:  add 1 to its count
    #       else:                  set its count to 1
    #
    # .get(char, 0) returns the current count, or 0 if not seen yet.
    # EXAMPLE:
    #   char_counts = {}
    #   char_counts.get('a', 0) → 0  (not seen yet)
    #   char_counts['a'] = 0 + 1 = 1
    #   char_counts.get('a', 0) → 1  (seen once)
    #   char_counts['a'] = 1 + 1 = 2

    print("--- Top 15 most common characters ---")
    char_counts = {}
    for char in text:
        char_counts[char] = char_counts.get(char, 0) + 1

    # sorted() with key=lambda x: x[1] sorts by the count (second
    # element of each tuple). reverse=True puts highest counts first.
    #
    # .items() converts dict to list of (key, value) tuples:
    #   {'a': 5, 'b': 3}.items() → [('a', 5), ('b', 3)]
    #
    # lambda x: x[1] means "for each tuple x, use x[1] (the count)
    # as the sorting key"
    sorted_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)

    for char, count in sorted_chars[:15]:
        # Calculate what percentage of all characters this one represents.
        # EXAMPLE: if 'e' appears 654 times out of 6201 total:
        #   percentage = (654 / 6201) * 100 = 10.5%
        percentage = (count / total_chars) * 100

        # Create a simple visual bar chart using # characters.
        # Multiply percentage by 2 so the bars are a reasonable length.
        # EXAMPLE: 10.5% → int(10.5 * 2) = 21 → "#####################"
        bar = "#" * int(percentage * 2)
        print(f"  {repr(char):6s} : {count:4d} ({percentage:5.1f}%) {bar}")
    print()

    # ==================================================================
    # STEP 6: SUMMARY — WHAT THIS MEANS FOR OUR MODEL
    # ==================================================================
    print("=" * 50)
    print("WHAT THIS MEANS FOR OUR AI MODEL")
    print("=" * 50)
    print(f"""
Our model will:
  - Learn a vocabulary of {vocab_size} characters
  - Train on {total_chars} characters of text
  - Try to predict the next character given previous ones

The more text we have, the better the model can learn patterns.
With {total_lines} lines of quotes, the model should learn common
English patterns like "th", "the", "ing", "tion", etc.

Next step: convert these characters into numbers (Step 04).
""")


# ======================================================================
# ENTRY POINT
# ======================================================================
# Same pattern as hello_pytorch.py — only run main() when this file
# is executed directly (not when imported by another file).
#
# To run:  python src/explore_data.py
# This expects you to be in the SimpleAI/ root directory so that
# the relative path "data/input.txt" resolves correctly.
if __name__ == "__main__":
    main()
