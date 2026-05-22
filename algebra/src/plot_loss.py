"""
plot_loss.py — Visualize training loss, accuracy, and teacher forcing.

PURPOSE:
    Generate plots showing how the model learned over 50 epochs.
    These visualizations help you understand:
    - Is the model learning? (loss should decrease)
    - Is it overfitting? (loss flattens but accuracy drops)
    - How does teacher forcing affect learning?

WHAT IT GENERATES:
    1. Training loss curve — shows loss decreasing over epochs
    2. Test accuracy curve — shows accuracy improving over epochs
    3. Teacher forcing schedule — shows TF ratio decreasing
    4. Combined plot — all three in one figure

OUTPUT: algebra/outputs/loss_plot.png

Usage:
    PYTHONPATH=algebra/src python algebra/src/plot_loss.py
"""

import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_training_history(history, output_path="algebra/outputs/loss_plot.png"):
    """
    Create a 3-panel figure showing training progress.

    Panel 1: Training loss over epochs (log scale)
    Panel 2: Test accuracy over epochs (with 90% target line)
    Panel 3: Teacher forcing ratio over epochs

    PARAMETERS:
        history (dict): Training history with 'losses', 'accuracies', 'tf_ratios'
        output_path (str): Where to save the plot
    """

    losses = history["losses"]
    accuracies = history["accuracies"]  # list of (epoch, accuracy) tuples
    tf_ratios = history["tf_ratios"]

    epochs = list(range(1, len(losses) + 1))
    acc_epochs = [e + 1 for e, _ in accuracies]
    acc_values = [a * 100 for _, a in accuracies]

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig.suptitle("Algebra Solver — Training Progress", fontsize=16, fontweight="bold")

    # ---- Panel 1: Loss ----
    ax1 = axes[0]
    ax1.plot(epochs, losses, color="#2196F3", linewidth=2, label="Training Loss")
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training Loss (lower is better)", fontsize=12)
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    # Annotate start and end
    ax1.annotate(f"{losses[0]:.3f}", xy=(1, losses[0]),
                 fontsize=9, color="#666666")
    ax1.annotate(f"{losses[-1]:.4f}", xy=(len(losses), losses[-1]),
                 fontsize=9, color="#666666", ha="right")

    # ---- Panel 2: Accuracy ----
    ax2 = axes[1]
    ax2.plot(acc_epochs, acc_values, color="#4CAF50", linewidth=2,
             marker="o", markersize=8, label="Test Accuracy")
    ax2.axhline(y=90, color="#FF5722", linestyle="--", linewidth=1.5,
                alpha=0.7, label="90% Target")
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_title("Test Accuracy (higher is better)", fontsize=12)
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower right")

    # Annotate each accuracy point
    for ep, acc in zip(acc_epochs, acc_values):
        ax2.annotate(f"{acc:.1f}%", xy=(ep, acc),
                     textcoords="offset points", xytext=(0, 12),
                     fontsize=9, ha="center", fontweight="bold")

    # ---- Panel 3: Teacher Forcing ----
    ax3 = axes[2]
    ax3.plot(epochs, tf_ratios, color="#FF9800", linewidth=2,
             label="Teacher Forcing Ratio")
    ax3.fill_between(epochs, tf_ratios, alpha=0.15, color="#FF9800")
    ax3.set_ylabel("TF Ratio", fontsize=12)
    ax3.set_xlabel("Epoch", fontsize=12)
    ax3.set_title("Teacher Forcing Schedule (decreasing over training)",
                  fontsize=12)
    ax3.set_ylim(0, 1.1)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper right")

    # Annotate start and end
    ax3.annotate(f"{tf_ratios[0]:.1f}", xy=(1, tf_ratios[0]),
                 fontsize=9, color="#666666")
    ax3.annotate(f"{tf_ratios[-1]:.1f}", xy=(len(tf_ratios), tf_ratios[-1]),
                 fontsize=9, color="#666666", ha="right")

    plt.tight_layout()

    # ---- Save ----
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    size_kb = os.path.getsize(output_path) / 1024
    print(f"  Plot saved to {output_path} ({size_kb:.1f} KB)")


def main():
    """Load training history and generate plots."""

    print("=" * 60)
    print("STEP 10: LOSS VISUALIZATION")
    print("=" * 60)
    print()

    # ---- Load history ----
    print("Loading training history...")
    history = torch.load("algebra/outputs/loss_history.pth", weights_only=False)

    print(f"  Epochs: {len(history['losses'])}")
    print(f"  Accuracy checkpoints: {len(history['accuracies'])}")
    print(f"  Loss range: {history['losses'][0]:.4f} → {history['losses'][-1]:.4f}")

    if history["accuracies"]:
        final_acc = history["accuracies"][-1][1]
        print(f"  Final accuracy: {final_acc:.1%}")
    print()

    # ---- Generate plot ----
    print("Generating plot...")
    plot_training_history(history)

    print()
    print("=" * 60)
    print("Visualization complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
