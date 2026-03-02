#!/bin/bash

# Speedrun for 2xA6000 GPUs training on custom Python dataset (58.4GB)
# This script trains nanochat on the Python code and documentation data we collected
# Expected runtime: ~16 hours for full training pipeline
#
# Key differences from original speedrun_2xa6000.sh:
# 1. Uses custom Python dataset instead of FineWeb-Edu
# 2. Data is pre-prepared, no download needed
# 3. Optimized for 2xA6000 (48GB VRAM each)

# Launch examples:
# 1) Simple: bash speedrun_2xa6000_python.sh
# 2) With screen: screen -L -Logfile speedrun_python.log -S speedrun_python bash speedrun_2xa6000_python.sh
# 3) With wandb: WANDB_RUN=python_2xa6000 bash speedrun_2xa6000_python.sh

# -----------------------------------------------------------------------------
# Environment setup

export OMP_NUM_THREADS=1

# IMPORTANT: Point to custom Python data directory
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export PYTHON_DATA_DIR="$NANOCHAT_BASE_DIR/custom_python_data"

# Verify custom data exists
if [ ! -d "$PYTHON_DATA_DIR" ]; then
    echo "ERROR: Python training data not found at $PYTHON_DATA_DIR"
    echo "Please run: python training-data-generator/scripts/prepare_for_nanochat.py"
    exit 1
fi

# Count shards
SHARD_COUNT=$(ls -1 "$PYTHON_DATA_DIR"/shard_*.parquet 2>/dev/null | wc -l)
if [ "$SHARD_COUNT" -eq 0 ]; then
    echo "ERROR: No shard files found in $PYTHON_DATA_DIR"
    exit 1
fi

echo "Found $SHARD_COUNT Python training shards in $PYTHON_DATA_DIR"

mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup

if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# Report setup

python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

# Install Rust / Cargo (if needed)
if ! command -v cargo &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

# Build the rustbpe Tokenizer
echo "Building tokenizer..."
uv run maturin develop --release --manifest-path ../rustbpe/Cargo.toml

# Train tokenizer on our Python data
# We'll use first 8 shards (~2B characters) for tokenizer training
echo "Training tokenizer on Python code..."
python -c "
import pyarrow.parquet as pq
import glob

# Read first 8 shards and extract text for tokenizer training
shard_files = sorted(glob.glob('$PYTHON_DATA_DIR/shard_*.parquet'))[:8]
texts = []
total_chars = 0
target_chars = 2_000_000_000  # 2B characters

print(f'Reading {len(shard_files)} shards for tokenizer training...')
for f in shard_files:
    table = pq.read_table(f)
    batch = table.column('text').to_pylist()
    for text in batch:
        texts.append(text)
        total_chars += len(text)
        if total_chars >= target_chars:
            break
    if total_chars >= target_chars:
        break

print(f'Collected {total_chars:,} characters from {len(texts):,} examples')

# Write to temporary file for tokenizer
import os
temp_file = os.path.join('$NANOCHAT_BASE_DIR', 'tokenizer_training_data.txt')
with open(temp_file, 'w', encoding='utf-8') as f:
    for text in texts[:100000]:  # Limit to 100k documents for speed
        f.write(text + '\\n\\n')

print(f'Wrote tokenizer training data to {temp_file}')
"

# Train the tokenizer
python -m scripts.tok_train --max-chars 2000000000

# Evaluate the tokenizer
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Prepare data for training

# Create symlink so nanochat's dataset.py finds our data
# nanochat expects data in base_data/, we have it in custom_python_data/
if [ ! -L "$NANOCHAT_BASE_DIR/base_data" ]; then
    echo "Creating symlink: base_data -> custom_python_data"
    ln -s "$PYTHON_DATA_DIR" "$NANOCHAT_BASE_DIR/base_data"
fi

# -----------------------------------------------------------------------------
# Base model (pretraining)

# Download the eval_bundle from s3 to evaluate CORE metric during training (~162MB)
EVAL_BUNDLE_URL=https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
if [ ! -d "$NANOCHAT_BASE_DIR/eval_bundle" ]; then
    echo "Downloading evaluation bundle..."
    curl -L -o eval_bundle.zip $EVAL_BUNDLE_URL
    unzip -q eval_bundle.zip
    rm eval_bundle.zip
    mv eval_bundle $NANOCHAT_BASE_DIR
fi

echo ""
echo "========================================================================"
echo "Starting base pretraining on Python dataset"
echo "========================================================================"
echo "Dataset: $SHARD_COUNT shards of Python code + documentation"
echo "Size: ~58.4GB (8.7M examples)"
echo "Model: d20 (561M parameters)"
echo "GPUs: 2xA6000 (48GB VRAM each)"
echo "Estimated time: ~12-16 hours"
echo "========================================================================"
echo ""

# pretrain the d20 model on Python data
# KEY SETTINGS for 2xA6000:
# - nproc_per_node=2 (2 GPUs)
# - device_batch_size=8 (fits in 48GB VRAM)
# - save_every=500 (checkpoint every 500 steps for resume capability)
# - Gradient accumulation automatically adjusted to maintain total_batch_size=524288
torchrun --standalone --nproc_per_node=2 -m scripts.base_train -- --depth=20 --device-batch-size=8 --save-every 500 --run=$WANDB_RUN

# evaluate the model on a larger chunk of train/val data and draw some samples
echo "Evaluating base model..."
torchrun --standalone --nproc_per_node=2 -m scripts.base_loss -- --device-batch-size=8

# evaluate the model on CORE tasks
torchrun --standalone --nproc_per_node=2 -m scripts.base_eval

# -----------------------------------------------------------------------------
# Midtraining (teach the model conversation special tokens, tool use, multiple choice)

echo ""
echo "========================================================================"
echo "Starting midtraining..."
echo "========================================================================"
echo ""

# run midtraining and eval the model
torchrun --standalone --nproc_per_node=2 -m scripts.mid_train -- --device-batch-size=8 --save-every 500 --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=2 -m scripts.chat_eval -- -i mid

# -----------------------------------------------------------------------------
# Supervised Finetuning (domain adaptation to each sequence all by itself per row)

echo ""
echo "========================================================================"
echo "Starting supervised finetuning..."
echo "========================================================================"
echo ""

# train sft and re-eval right away (should see a small bump)
torchrun --standalone --nproc_per_node=2 -m scripts.chat_sft -- --device-batch-size=2 --save-every 500 --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=2 -m scripts.chat_eval -- -i sft

# chat with the model over CLI! Leave out the -p to chat interactively
# python -m scripts.chat_cli -p "Write a Python function to reverse a list"

# even better, chat with your model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
# Reinforcement Learning (optional, commented out by default)

# run reinforcement learning
# torchrun --standalone --nproc_per_node=2 -m scripts.chat_rl -- --run=$WANDB_RUN
# eval the RL model only on GSM8K
# torchrun --standalone --nproc_per_node=2 -m scripts.chat_eval -- -i rl -a GSM8K

# -----------------------------------------------------------------------------
# Generate the full report

echo ""
echo "========================================================================"
echo "Generating final report..."
echo "========================================================================"
echo ""

python -m nanochat.report generate

echo ""
echo "========================================================================"
echo "Training complete! 🎉"
echo "========================================================================"
echo ""
echo "Model trained on: 58.4GB Python code + documentation (8.7M examples)"
echo ""
echo "To chat with your Python-specialized model:"
echo "  1. CLI: python -m scripts.chat_cli"
echo "  2. Web UI: python -m scripts.chat_web"
echo ""
echo "Example Python-specific prompts:"
echo "  • Write a Python function to calculate fibonacci numbers"
echo "  • Explain how decorators work in Python"
echo "  • Debug this code: [paste Python code]"
echo "  • What's the difference between list and tuple?"
echo ""
echo "View the training report: cat report.md"
echo ""
