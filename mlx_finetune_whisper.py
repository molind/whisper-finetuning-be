"""
Fine-tune OpenAI Whisper using Apple MLX framework.
Designed for Apple Silicon with zero-copy unified memory.

Usage:
    python mlx_finetune_whisper.py \
        --model mlx-community/whisper-small-mlx \
        --dataset-dir /Volumes/4tb/voice/cv-corpus-24.0-2025-12-05/be \
        --output-dir ./output_mlx_whisper \
        --batch-size 4 \
        --iters 10000 \
        --learning-rate 1e-5
"""

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

# Add mlx-whisper to path
import sys
sys.path.insert(0, os.path.expanduser("~/tmp/mlx-examples/whisper"))

from mlx_whisper.audio import load_audio, log_mel_spectrogram, pad_or_trim, N_SAMPLES
from mlx_whisper.load_models import load_model


def load_tokenizer(model_path):
    """Load the whisper tokenizer."""
    from mlx_whisper.tokenizer import get_tokenizer
    model_path = Path(model_path)
    if not model_path.exists():
        from huggingface_hub import snapshot_download
        model_path = Path(snapshot_download(repo_id=str(model_path)))
    with open(model_path / "config.json") as f:
        config = json.load(f)
    multilingual = config.get("n_vocab", 51865) >= 51865
    return get_tokenizer(multilingual=multilingual, language="be", task="transcribe")


def load_cv_dataset(dataset_dir, split="train"):
    """Load Common Voice TSV metadata. Returns list of (audio_path, sentence)."""
    split_to_file = {
        "train": "train.tsv",
        "validation": "dev.tsv",
        "dev": "dev.tsv",
        "test": "test.tsv",
    }
    tsv_path = os.path.join(dataset_dir, split_to_file[split])
    clips_dir = os.path.join(dataset_dir, "clips")

    samples = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            audio_path = os.path.join(clips_dir, row["path"])
            samples.append((audio_path, row["sentence"]))
    return samples


def prepare_batch(samples, tokenizer, n_mels=80):
    """Prepare a batch of (audio_path, sentence) pairs for training.

    Returns:
        mel: mx.array of shape (batch, n_frames, n_mels)
        tokens: mx.array of shape (batch, max_token_len)
        targets: mx.array of shape (batch, max_token_len)
    """
    mels = []
    all_tokens = []

    for audio_path, sentence in samples:
        # Load and preprocess audio
        try:
            audio = load_audio(audio_path)
        except Exception:
            continue
        audio = pad_or_trim(audio, N_SAMPLES)
        mel = log_mel_spectrogram(audio, n_mels=n_mels)
        mels.append(mel)

        # Tokenize text: prefix (SOT + language + task) + text tokens + EOT
        prefix = list(tokenizer.sot_sequence)
        text_tokens = tokenizer.encode(sentence)
        all_tokens.append(prefix + text_tokens + [tokenizer.eot])

    if not mels:
        return None, None, None, None

    # Pad mel spectrograms (should all be same size after pad_or_trim)
    mel_batch = mx.stack(mels)

    # Pad token sequences and track lengths
    max_len = max(len(t) for t in all_tokens)
    eot = tokenizer.eot
    padded_tokens = []
    # Length = number of valid target tokens (excluding padding)
    lengths = []
    for t in all_tokens:
        lengths.append(len(t) - 1)  # targets are shifted, so len-1 valid targets
        padded = list(t) + [eot] * (max_len - len(t))
        padded_tokens.append(padded)

    tokens_array = mx.array(padded_tokens)

    # Decoder input: all tokens except the last
    decoder_input = tokens_array[:, :-1]
    # Target: all tokens except the first (shifted by one)
    targets = tokens_array[:, 1:]
    lengths = mx.array(lengths)

    return mel_batch, decoder_input, targets, lengths


def loss_fn(model, mel, decoder_input, targets, lengths):
    """Compute cross-entropy loss for whisper fine-tuning."""
    logits = model(mel, decoder_input)

    # Mask: 1 for valid tokens (including first EOT), 0 for padding
    steps = mx.arange(targets.shape[1])[None, :]  # (1, seq_len)
    mask = (steps < lengths[:, None]).astype(mx.float32)

    ce = nn.losses.cross_entropy(logits, targets) * mask
    ntoks = mask.sum()
    loss = ce.astype(mx.float32).sum() / mx.maximum(ntoks, 1)

    return loss, ntoks


def evaluate(model, val_samples, tokenizer, n_mels, batch_size, max_batches=10):
    """Run evaluation and return average loss."""
    model.eval()
    total_loss = 0.0
    total_toks = 0
    n_batches = 0

    for i in range(0, min(len(val_samples), max_batches * batch_size), batch_size):
        batch = val_samples[i : i + batch_size]
        mel, decoder_input, targets, lengths = prepare_batch(batch, tokenizer, n_mels)
        if mel is None:
            continue

        loss, ntoks = loss_fn(model, mel, decoder_input, targets, lengths)
        mx.eval(loss, ntoks)
        total_loss += loss.item()  * ntoks.item()
        total_toks += ntoks.item()
        n_batches += 1

        if n_batches >= max_batches:
            break

    avg_loss = total_loss / max(total_toks, 1)
    return avg_loss


def save_checkpoint(model, optimizer, iteration, ckpt_path):
    """Save model weights, optimizer state, and training state to checkpoint."""
    os.makedirs(ckpt_path, exist_ok=True)
    # Model weights
    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(os.path.join(ckpt_path, "model.safetensors"), weights)
    # Optimizer state
    opt_state = dict(tree_flatten(optimizer.state))
    mx.save_safetensors(os.path.join(ckpt_path, "optimizer.safetensors"), opt_state)
    # Training state
    train_state = {"iter": iteration}
    with open(os.path.join(ckpt_path, "train_state.json"), "w") as f:
        json.dump(train_state, f)
    # Model config
    config = {
        "n_mels": model.dims.n_mels,
        "n_audio_ctx": model.dims.n_audio_ctx,
        "n_audio_state": model.dims.n_audio_state,
        "n_audio_head": model.dims.n_audio_head,
        "n_audio_layer": model.dims.n_audio_layer,
        "n_vocab": model.dims.n_vocab,
        "n_text_ctx": model.dims.n_text_ctx,
        "n_text_state": model.dims.n_text_state,
        "n_text_head": model.dims.n_text_head,
        "n_text_layer": model.dims.n_text_layer,
    }
    with open(os.path.join(ckpt_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


def train(args):
    # Limit MLX metal cache to prevent unbounded memory growth
    mx.set_cache_limit(10 * 1024**3)  # 10 GB

    print(f"Loading model: {args.model}")
    model = load_model(args.model, dtype=mx.float32)

    # Cast all float16 weights to float32 — load_model ignores dtype for safetensors
    params = tree_flatten(model.parameters())
    cast_params = [
        (k, v.astype(mx.float32) if v.dtype == mx.float16 else v) for k, v in params
    ]
    model.update(tree_unflatten(cast_params))
    mx.eval(model.parameters())

    tokenizer = load_tokenizer(args.model)

    n_mels = model.dims.n_mels
    eot_token = tokenizer.eot

    # Count parameters
    total_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"Total parameters: {total_params:,}")

    # Freeze encoder if requested
    if args.freeze_encoder:
        model.encoder.freeze()
        trainable_params = sum(
            p.size for _, p in tree_flatten(model.trainable_parameters())
        )
        print(f"Encoder frozen. Trainable parameters: {trainable_params:,}")

    # Load dataset
    print(f"Loading dataset from: {args.dataset_dir}")
    train_samples = load_cv_dataset(args.dataset_dir, "train")
    val_samples = load_cv_dataset(args.dataset_dir, "validation")
    print(f"Train: {len(train_samples)} samples, Val: {len(val_samples)} samples")

    # Shuffle training data
    np.random.seed(args.seed)
    np.random.shuffle(train_samples)

    # Setup optimizer
    if args.warmup_steps > 0:
        lr_schedule = optim.linear_schedule(
            init=1e-7,
            end=args.learning_rate,
            steps=args.warmup_steps,
        )
        decay_schedule = optim.cosine_decay(
            init=args.learning_rate,
            decay_steps=args.iters - args.warmup_steps,
        )
        lr_schedule = optim.join_schedules(
            [lr_schedule, decay_schedule],
            [args.warmup_steps],
        )
    else:
        lr_schedule = args.learning_rate

    optimizer = optim.AdamW(learning_rate=lr_schedule, weight_decay=args.weight_decay)

    # Resume from checkpoint if requested
    start_iter = 1
    if args.resume:
        ckpt_path = args.resume
        # Auto-detect latest checkpoint in output_dir
        if ckpt_path == "latest":
            ckpts = sorted(
                [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint_")],
            )
            if not ckpts:
                print("No checkpoints found, starting from scratch")
            else:
                ckpt_path = os.path.join(args.output_dir, ckpts[-1])
        if ckpt_path != "latest":
            print(f"Resuming from: {ckpt_path}")
            # Load model weights
            weights = mx.load(os.path.join(ckpt_path, "model.safetensors"))
            model.load_weights(list(weights.items()))
            if args.freeze_encoder:
                model.encoder.freeze()
            # Load optimizer state
            opt_state_path = os.path.join(ckpt_path, "optimizer.safetensors")
            if os.path.exists(opt_state_path):
                opt_weights = mx.load(opt_state_path)
                optimizer.state = tree_unflatten(list(opt_weights.items()))
            # Load training state
            train_state_path = os.path.join(ckpt_path, "train_state.json")
            if os.path.exists(train_state_path):
                with open(train_state_path) as f:
                    train_state = json.load(f)
                start_iter = train_state["iter"] + 1
            mx.eval(model.parameters(), optimizer.state)
            print(f"Resumed at iter {start_iter}")

    # Setup training
    loss_and_grad = nn.value_and_grad(model, lambda m, mel, di, tgt, lengths: loss_fn(m, mel, di, tgt, lengths))

    os.makedirs(args.output_dir, exist_ok=True)

    # Training loop
    model.train()
    sample_idx = 0
    losses = 0.0
    n_tokens = 0
    train_time = 0.0

    print(f"\nStarting training for {args.iters} iterations (from iter {start_iter})...")
    print(f"Batch size: {args.batch_size}, LR: {args.learning_rate}")

    for it in range(start_iter, args.iters + 1):
        tic = time.perf_counter()

        # Get batch
        if sample_idx + args.batch_size > len(train_samples):
            np.random.shuffle(train_samples)
            sample_idx = 0
        batch = train_samples[sample_idx : sample_idx + args.batch_size]
        sample_idx += args.batch_size

        mel, decoder_input, targets, lengths = prepare_batch(batch, tokenizer, n_mels)
        if mel is None:
            continue

        # Forward + backward + update
        (loss, ntoks), grad = loss_and_grad(model, mel, decoder_input, targets, lengths)
        optimizer.update(model, grad)
        mx.eval(model.parameters(), optimizer.state, loss, ntoks)

        losses += loss.item()
        n_tokens += ntoks.item()
        train_time += time.perf_counter() - tic

        # Log
        if it % args.log_every == 0:
            avg_loss = losses / args.log_every
            toks_sec = n_tokens / train_time
            peak_mem = mx.get_peak_memory() / 1e9
            active_mem = mx.get_active_memory() / 1e9
            lr = optimizer.learning_rate.item() if hasattr(optimizer.learning_rate, 'item') else args.learning_rate
            print(
                f"Iter {it}/{args.iters}: "
                f"loss={avg_loss:.4f}, "
                f"lr={lr:.2e}, "
                f"tok/s={toks_sec:.1f}, "
                f"mem={active_mem:.1f}/{peak_mem:.1f}GB, "
                f"it/s={args.log_every / train_time:.2f}"
            )
            losses = 0.0
            n_tokens = 0
            train_time = 0.0

        # Evaluate
        if it % args.eval_every == 0:
            val_loss = evaluate(
                model, val_samples, tokenizer, n_mels,
                args.batch_size, max_batches=args.val_batches,
            )
            print(f"Iter {it}: val_loss={val_loss:.4f}")
            model.train()

        # Save checkpoint
        if it % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_{it:07d}")
            save_checkpoint(model, optimizer, it, ckpt_path)
            print(f"Iter {it}: saved checkpoint to {ckpt_path}")

    # Save final model
    final_path = os.path.join(args.output_dir, "final")
    save_checkpoint(model, optimizer, args.iters, final_path)
    print(f"\nTraining complete. Final model saved to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Whisper with MLX")
    parser.add_argument("--model", type=str, default="mlx-community/whisper-small-mlx")
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./output_mlx_whisper")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--iters", type=int, default=10000)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--val-batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path, or 'latest' to auto-detect")
    args = parser.parse_args()
    train(args)
