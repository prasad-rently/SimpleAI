"""
plot_loss.py — Visualize the training loss curve.

PURPOSE:
    After training (Step 10), we have a list of 100 loss values stored in
    outputs/loss_history.pth. This script loads that list and plots it as
    a chart, showing how the model improved over time.

    The loss curve is the MOST IMPORTANT diagnostic tool in machine learning.
    It answers: "Is my model actually learning?"

WHAT THE LOSS CURVE TELLS YOU:
    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  Smooth downward curve  → model is learning normally        │
    │  Flat line              → model isn't learning (bad config) │
    │  Spiky/noisy            → learning rate too high            │
    │  Drops then flattens    → converged (done learning)         │
    │  Goes back UP           → overfitting or instability        │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

    Our model's curve should show a sharp drop early (learning common
    patterns like spaces and 'e'), then a gradual flattening as it
    runs out of easy patterns to learn.

WHY MATPLOTLIB?
    matplotlib is Python's standard charting library. It works like this:
    1. Create a figure (the canvas)
    2. Add data to it (the line)
    3. Add labels and styling (title, axes, grid)
    4. Save to a file (PNG image)

    It's the same tool used by data scientists, researchers, and engineers
    worldwide. Learning matplotlib basics is valuable for any Python work.

WHAT THIS FILE PROVIDES:
    1. plot_training_loss() — creates and saves the loss curve chart
    2. main() — loads loss_history.pth and calls the plotting function

INPUT:  outputs/loss_history.pth (list of 100 floats)
OUTPUT: outputs/loss_plot.png (chart image)

Usage:
    PYTHONPATH=src python src/plot_loss.py
"""

import os
import torch
import matplotlib
# ---- Use non-interactive backend ----
# By default, matplotlib tries to open a GUI window to display charts.
# On servers or headless machines (no monitor), this crashes.
# The 'Agg' backend tells matplotlib to render charts to image files
# without needing a display. "Agg" stands for "Anti-Grain Geometry",
# a C++ rendering engine that produces high-quality PNG images.
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_training_loss(loss_history, save_path="outputs/loss_plot.png"):
    """
    Plot the training loss curve and save it as a PNG image.

    This function takes the list of loss values from training and
    creates a line chart showing how loss decreased over epochs.

    HOW MATPLOTLIB WORKS (step by step):
        1. plt.figure()    → create a blank canvas (like opening a new document)
        2. plt.plot()      → draw a line on the canvas using our data
        3. plt.xlabel()    → label the x-axis
        4. plt.ylabel()    → label the y-axis
        5. plt.title()     → add a title at the top
        6. plt.savefig()   → save the canvas to a PNG file
        7. plt.close()     → free memory (important in scripts)

    WHAT IS figsize=(10, 6)?
        The figure size in inches: 10 inches wide, 6 inches tall.
        At 100 DPI (dots per inch), this creates a 1000×600 pixel image.
        This is a good size for viewing on screen or embedding in docs.

    WHAT IS plt.grid(True, alpha=0.3)?
        grid(True) adds light gridlines behind the chart.
        alpha=0.3 makes them 30% opaque (70% transparent).
        This helps you read values off the chart without the grid
        being distracting.

    WHAT IS plt.tight_layout()?
        Automatically adjusts margins so labels don't get cut off.
        Without it, long axis labels can extend beyond the image edge.

    PARAMETERS:
        loss_history (list[float]): Average loss per epoch from training.
                                    Example: [3.00, 2.71, 2.31, ..., 0.04]
        save_path (str):           Where to save the PNG image.
                                    Default: outputs/loss_plot.png

    RETURNS:
        None (saves image to disk and prints confirmation)

    EXAMPLE:
        >>> loss_history = [3.0, 2.5, 2.0, 1.5, 1.0, 0.5]
        >>> plot_training_loss(loss_history)
        Loss plot saved to outputs/loss_plot.png
    """

    # ---- Ensure output directory exists ----
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # ==================================================================
    # CREATE THE FIGURE
    # ==================================================================
    # plt.figure() creates a new blank canvas.
    # figsize=(10, 6) → 10 inches wide, 6 inches tall
    # This is the container that holds our chart.
    fig = plt.figure(figsize=(10, 6))

    # ==================================================================
    # PLOT THE LOSS CURVE
    # ==================================================================
    # plt.plot(x_values, y_values) draws a line connecting the points.
    #
    # x_values: range(len(loss_history)) → [0, 1, 2, ..., 99]
    #           These are the epoch numbers.
    #
    # y_values: loss_history → [3.00, 2.71, ..., 0.04]
    #           These are the average loss at each epoch.
    #
    # color='#2196F3' → Material Design blue. Looks professional.
    #   Colors in matplotlib can be:
    #   - Named: 'red', 'blue', 'green'
    #   - Hex codes: '#FF0000', '#2196F3'
    #   - RGB tuples: (0.13, 0.59, 0.95)
    #
    # linewidth=2 → makes the line thicker (default is 1).
    #   Thicker lines are easier to see, especially in small images.
    #
    # label='Training Loss' → text shown in the legend.
    #   The legend is the small box that identifies what each line means.
    epochs = range(len(loss_history))
    plt.plot(epochs, loss_history, color='#2196F3', linewidth=2,
             label='Training Loss')

    # ==================================================================
    # ADD REFERENCE LINE: RANDOM BASELINE
    # ==================================================================
    # Draw a horizontal dashed line at the random baseline loss.
    # This makes it easy to see that the model improved beyond random.
    #
    # plt.axhline() draws a HORIZONTAL line across the entire chart.
    #   y=3.8712     → the y-position (random baseline for 48 chars)
    #   color='red'  → red to stand out from the blue loss curve
    #   linestyle='--' → dashed line (-- = dashes, - = solid, : = dots)
    #   alpha=0.7    → slightly transparent so it doesn't dominate
    #   label=...    → text for the legend
    import math
    random_baseline = -math.log(1.0 / 48)
    plt.axhline(y=random_baseline, color='red', linestyle='--', alpha=0.7,
                label=f'Random Baseline ({random_baseline:.2f})')

    # ==================================================================
    # ADD LABELS AND STYLING
    # ==================================================================

    # ---- Axis labels ----
    # xlabel/ylabel tell the reader what each axis represents.
    # fontsize=12 makes the text slightly larger than default (10).
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)

    # ---- Title ----
    # The title goes at the top of the chart.
    # fontsize=14 and fontweight='bold' make it prominent.
    plt.title('Training Loss Over Time', fontsize=14, fontweight='bold')

    # ---- Legend ----
    # The legend shows what each line/color represents.
    # It automatically uses the 'label' text from plt.plot() and axhline().
    # loc='upper right' places it in the top-right corner.
    plt.legend(loc='upper right', fontsize=11)

    # ---- Grid ----
    # Light gridlines behind the chart help you read values.
    # alpha=0.3 makes them subtle — visible but not distracting.
    plt.grid(True, alpha=0.3)

    # ---- Layout ----
    # tight_layout() adjusts margins so nothing gets clipped.
    plt.tight_layout()

    # ==================================================================
    # ADD ANNOTATIONS
    # ==================================================================
    # Mark key points on the chart to highlight the learning journey.

    # ---- Annotate the starting loss ----
    # The first epoch's loss — where the model started.
    first_loss = loss_history[0]
    plt.annotate(f'Start: {first_loss:.2f}',
                 xy=(0, first_loss),
                 xytext=(15, first_loss + 0.3),
                 fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='gray'),
                 color='gray')

    # ---- Annotate the final loss ----
    # The last epoch's loss — where the model ended up.
    final_loss = loss_history[-1]
    final_epoch = len(loss_history) - 1
    plt.annotate(f'Final: {final_loss:.2f}',
                 xy=(final_epoch, final_loss),
                 xytext=(final_epoch - 25, final_loss + 0.5),
                 fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='gray'),
                 color='gray')

    # ==================================================================
    # SAVE THE CHART
    # ==================================================================
    # plt.savefig() renders the chart to a file.
    # dpi=150 → 150 dots per inch. Higher than default (100) for
    #           sharper text and lines. Good for viewing on screen.
    #           At 150 DPI, our 10×6 inch figure becomes 1500×900 pixels.
    # bbox_inches='tight' → crop whitespace around the chart.
    plt.savefig(save_path, dpi=150, bbox_inches='tight')

    # ---- Close the figure to free memory ----
    # Without plt.close(), each figure stays in memory.
    # In a script that creates many charts, this causes memory leaks.
    plt.close(fig)

    # ---- Report success ----
    file_size_kb = os.path.getsize(save_path) / 1024
    print(f"  Loss plot saved to {save_path} ({file_size_kb:.1f} KB)")


def print_loss_summary(loss_history):
    """
    Print a text-based summary of the training progress.

    This provides key statistics from the loss history without
    needing to look at the chart. Useful for quick checks in
    the terminal.

    WHAT WE REPORT:
        - Total epochs trained
        - First and final loss (to see total improvement)
        - Best loss and which epoch it occurred at
        - Loss at key milestones (25%, 50%, 75% through training)
        - Overall reduction percentage

    PARAMETERS:
        loss_history (list[float]): Average loss per epoch

    RETURNS:
        None (prints to console)
    """

    num_epochs = len(loss_history)
    first_loss = loss_history[0]
    final_loss = loss_history[-1]
    best_loss = min(loss_history)
    best_epoch = loss_history.index(best_loss)

    # ---- Calculate milestone losses ----
    # Show loss at 25%, 50%, 75% through training for a quick overview.
    # int() truncates to the nearest whole number.
    quarter = int(num_epochs * 0.25)
    half = int(num_epochs * 0.50)
    three_quarter = int(num_epochs * 0.75)

    print(f"""
  Loss Summary:
    ─────────────────────────────────────
    Total epochs  : {num_epochs}
    First loss    : {first_loss:.4f}  (epoch 0)
    Final loss    : {final_loss:.4f}  (epoch {num_epochs - 1})
    Best loss     : {best_loss:.4f}  (epoch {best_epoch})
    Improvement   : {((first_loss - final_loss) / first_loss * 100):.1f}% reduction
    ─────────────────────────────────────

  Loss at milestones:
    Epoch {0:3d} (  0%): {loss_history[0]:.4f}
    Epoch {quarter:3d} ( 25%): {loss_history[quarter]:.4f}
    Epoch {half:3d} ( 50%): {loss_history[half]:.4f}
    Epoch {three_quarter:3d} ( 75%): {loss_history[three_quarter]:.4f}
    Epoch {num_epochs-1:3d} (100%): {loss_history[-1]:.4f}
""")


def main():
    """
    Main function — load loss history and create the loss plot.

    FLOW:
        1. Load outputs/loss_history.pth (saved by train.py in Step 10)
        2. Print a text summary of the training progress
        3. Create and save the loss curve chart
    """

    # ==================================================================
    # SETUP
    # ==================================================================
    print("=" * 60)
    print("STEP 11: LOSS CURVE VISUALIZATION")
    print("=" * 60)
    print()

    # ---- Load loss history from disk ----
    # This was saved by train.py at the end of training.
    # torch.load() deserializes (converts from bytes back to Python objects)
    # the file we saved with torch.save().
    #
    # weights_only=False because we're loading a plain Python list,
    # not a model state_dict. PyTorch warns about this by default
    # because loading arbitrary Python objects can be a security risk
    # (pickle can execute code). Since we saved this file ourselves,
    # it's safe.
    loss_history_path = "outputs/loss_history.pth"

    if not os.path.exists(loss_history_path):
        print(f"  ERROR: {loss_history_path} not found!")
        print(f"  Run train.py first (Step 10) to generate loss history.")
        return

    loss_history = torch.load(loss_history_path, weights_only=False)
    print(f"Loaded loss history: {len(loss_history)} epochs from {loss_history_path}")

    # ==================================================================
    # TEXT SUMMARY
    # ==================================================================
    # Print key statistics before showing the chart.
    print_loss_summary(loss_history)

    # ==================================================================
    # CREATE THE PLOT
    # ==================================================================
    # Generate the loss curve chart and save as PNG.
    print("Creating loss plot...")
    plot_training_loss(loss_history, save_path="outputs/loss_plot.png")
    print()

    # ==================================================================
    # WHAT COMES NEXT
    # ==================================================================
    print("=" * 60)
    print("WHAT COMES NEXT (Step 12)")
    print("=" * 60)
    print("""
The model is trained and we can see it learned (loss curve goes down).
Now it's time to USE the model to generate text!

  Step 12: Load the saved model and generate new text
           The model predicts one character at a time,
           building up sentences character by character.

  To generate text, we'll need:
    - outputs/model.pth  (trained weights)
    - outputs/vocab.pth  (to decode numbers → text)
""")


# ======================================================================
# ENTRY POINT
# ======================================================================
if __name__ == "__main__":
    main()
