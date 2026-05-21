# SimpleAI

A tiny character-level text generator built from scratch to learn how AI models work.

## What is this?

This project builds a small neural network that:
1. Reads a text file (like Shakespeare quotes)
2. Learns the patterns in it (which characters tend to follow which)
3. Generates new text that mimics the style

It's the same core idea behind ChatGPT and Claude — just massively scaled down to run on any laptop.

## Project structure

```
SimpleAI/
├── data/           <- Training text goes here
├── src/            <- All Python source code
├── outputs/        <- Trained models and charts (generated, not committed)
├── requirements.txt <- Python dependencies
└── REQUIREMENTS.md  <- Detailed plan and AI concepts explained
```

## Getting started

```bash
# 1. Create a virtual environment (keeps dependencies isolated)
python3 -m venv venv

# 2. Activate it
source venv/bin/activate    # macOS/Linux
# venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

## Learning path

This project is built in 15 small steps, each on its own branch.
Check the [REQUIREMENTS.md](REQUIREMENTS.md) for the full plan.
