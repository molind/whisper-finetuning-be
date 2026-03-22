"""
Fine-tune NVIDIA Conformer-CTC using Apple MLX framework.
Loads pre-trained weights from extracted NeMo checkpoint, fine-tunes with CTC loss.

Usage:
    python mlx_finetune_conformer.py \
        --model nemo_models/ctc_extracted \
        --dataset-dir /path/to/cv-corpus/be \
        --output-dir ./output_mlx_conformer \
        --batch-size 4 \
        --iters 10000 \
        --learning-rate 1e-4
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import sentencepiece as spm
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from belarusian_text_normalizer import BelarusianTextNormalizer
from mlx_conformer import (
    ConformerCTC,
    compute_mel_spectrogram,
    ctc_greedy_decode,
    load_audio_file,
    load_nemo_ctc,
    load_vocabulary,
)


# ─── Data Loading ────────────────────────────────────────────────────────────


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


def load_tokenizer(model_dir):
    """Load SentencePiece tokenizer from NeMo model directory."""
    model_dir = Path(model_dir)
    spe_files = list(model_dir.glob("*_tokenizer.model"))
    if not spe_files:
        raise FileNotFoundError(f"No SentencePiece tokenizer found in {model_dir}")
    sp = spm.SentencePieceProcessor()
    sp.Load(str(spe_files[0]))
    return sp


def subsample_length(T):
    """Compute time dimension after ConvSubsampling (2x stride-2 convs with pad=1, kernel=3)."""
    for _ in range(2):
        T = (T - 1) // 2 + 1
    return T


def prepare_batch(samples, tokenizer, normalizer):
    """Prepare a batch for CTC training.

    Returns:
        mel_batch: mx.array (B, T_max, 80) padded mel spectrograms
        targets: list of list of int — BPE token IDs per sample
        input_lengths: list of int — subsampled time lengths
        target_lengths: list of int — target sequence lengths
    """
    mels = []
    mel_lengths = []
    all_targets = []

    for audio_path, sentence in samples:
        try:
            audio = load_audio_file(audio_path)
        except Exception:
            continue

        if len(audio) < 320:  # skip audio shorter than 20ms
            continue

        mel = compute_mel_spectrogram(audio)  # (T, 80) mx.array
        T_mel = mel.shape[0]

        # Tokenize with SentencePiece
        norm_text = normalizer(sentence)
        tokens = tokenizer.EncodeAsIds(norm_text)
        if not tokens:
            continue

        # CTC requires T_sub >= S (input length >= target length)
        T_sub = subsample_length(T_mel)
        if T_sub < len(tokens):
            continue

        mels.append(mel)
        mel_lengths.append(T_mel)
        all_targets.append(tokens)

    if not mels:
        return None, None, None, None

    # Pad mels to max length in batch
    T_max = max(m.shape[0] for m in mels)
    padded_mels = []
    for m in mels:
        pad_len = T_max - m.shape[0]
        if pad_len > 0:
            m = mx.concatenate([m, mx.zeros((pad_len, m.shape[1]))], axis=0)
        padded_mels.append(m)
    mel_batch = mx.stack(padded_mels)

    input_lengths = [subsample_length(t) for t in mel_lengths]
    target_lengths = [len(t) for t in all_targets]

    return mel_batch, all_targets, input_lengths, target_lengths


# ─── CTC Loss ────────────────────────────────────────────────────────────────


def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank_id):
    """CTC loss via forward algorithm in log-space.

    Processes each sample independently (batch sizes are small on Apple Silicon).
    Gradients flow through log_probs back to the model.

    Args:
        log_probs: (B, T_max, C) log probabilities from model
        targets: list of list of int — per-sample token IDs (no blanks)
        input_lengths: list of int — valid time steps per sample (after subsampling)
        target_lengths: list of int — target sequence length per sample
        blank_id: int — blank class index (last)

    Returns:
        loss: scalar, mean CTC loss over batch
    """
    B = len(input_lengths)
    NEG_INF = -1e10
    neg_inf_scalar = mx.array(NEG_INF)
    total_loss = mx.array(0.0)
    valid = 0

    for b in range(B):
        T = input_lengths[b]
        S = target_lengths[b]

        if T <= 0 or S <= 0 or T < S:
            continue

        lp = log_probs[b]  # (T_max, C)
        tgt = targets[b]  # list of S ints

        # Extended labels: [blank, t0, blank, t1, ..., t_{S-1}, blank]
        L = 2 * S + 1
        ext_list = []
        for i in range(S):
            ext_list.append(blank_id)
            ext_list.append(tgt[i])
        ext_list.append(blank_id)
        ext_labels = mx.array(ext_list)

        # Initialize alpha in log-space
        alpha = mx.full((L,), NEG_INF)
        indices = mx.arange(L)
        alpha = mx.where(indices == 0, lp[0, blank_id], alpha)
        if S > 0:
            alpha = mx.where(indices == 1, lp[0, tgt[0]], alpha)

        # Precompute skip mask: allowed when ext[s] != blank AND ext[s] != ext[s-2]
        ext_s_minus_2 = mx.concatenate([mx.array([-1, -1]), ext_labels[:-2]])
        skip_mask = (ext_labels != blank_id) & (ext_labels != ext_s_minus_2)

        # Pre-create padding constants
        pad1 = mx.array([NEG_INF])
        pad2 = mx.array([NEG_INF, NEG_INF])

        # Forward algorithm
        for t in range(1, T):
            lp_t = lp[t][ext_labels]  # (L,) log probs for extended labels at time t

            stay = alpha  # (L,) same position
            from_prev = mx.concatenate([pad1, alpha[:-1]])  # (L,) from position s-1
            skip = mx.concatenate([pad2, alpha[:-2]])  # (L,) from position s-2

            skip_masked = mx.where(skip_mask, skip, neg_inf_scalar)

            alpha = mx.logaddexp(mx.logaddexp(stay, from_prev), skip_masked) + lp_t

        # Loss: -log P(targets | input)
        loss_b = -mx.logaddexp(alpha[L - 1], alpha[L - 2])
        total_loss = total_loss + loss_b
        valid += 1

    if valid == 0:
        return mx.array(0.0)
    return total_loss / valid


def loss_fn(model, mel_batch, targets, input_lengths, target_lengths, blank_id):
    """Compute CTC loss for a batch."""
    log_probs = model(mel_batch)
    return ctc_loss(log_probs, targets, input_lengths, target_lengths, blank_id)


# ─── Evaluation ──────────────────────────────────────────────────────────────


def evaluate_wer(model, val_samples, vocabulary, normalizer, max_samples=200):
    """Compute WER on validation set using greedy CTC decoding."""
    from jiwer import wer

    model.eval()
    refs, preds = [], []

    for i, (audio_path, ref) in enumerate(val_samples[:max_samples]):
        try:
            audio = load_audio_file(audio_path)
            mel = compute_mel_spectrogram(audio)
            mel = mx.expand_dims(mel, axis=0)
            log_probs = model(mel)
            mx.eval(log_probs)
            text = ctc_greedy_decode(log_probs[0], vocabulary)
        except Exception:
            continue

        norm_ref = normalizer(ref)
        norm_pred = normalizer(text)
        if norm_ref.strip():
            refs.append(norm_ref)
            preds.append(norm_pred)

    if not refs:
        return 1.0
    return wer(refs, preds)


# ─── Checkpointing ──────────────────────────────────────────────────────────


def save_checkpoint(model, optimizer, iteration, vocabulary, model_config, ckpt_path):
    """Save model weights, optimizer state, training state, config, and vocabulary."""
    os.makedirs(ckpt_path, exist_ok=True)

    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(os.path.join(ckpt_path, "model.safetensors"), weights)

    opt_state = dict(tree_flatten(optimizer.state))
    mx.save_safetensors(os.path.join(ckpt_path, "optimizer.safetensors"), opt_state)

    with open(os.path.join(ckpt_path, "train_state.json"), "w") as f:
        json.dump({"iter": iteration}, f)

    with open(os.path.join(ckpt_path, "config.json"), "w") as f:
        json.dump(model_config, f, indent=2)

    with open(os.path.join(ckpt_path, "vocabulary.json"), "w") as f:
        json.dump(vocabulary, f, ensure_ascii=False, indent=2)


# ─── Training ────────────────────────────────────────────────────────────────


def train(args):
    mx.set_cache_limit(10 * 1024**3)  # 10 GB

    # Load model
    print(f"Loading CTC model from: {args.model}")
    model = load_nemo_ctc(args.model)
    vocabulary = load_vocabulary(args.model)
    tokenizer = load_tokenizer(args.model)
    blank_id = model.num_classes - 1
    normalizer = BelarusianTextNormalizer()

    model_config = {
        "n_layers": len(model.layers),
        "d_model": model.d_model,
        "num_classes": model.num_classes,
    }

    # Freeze batch norm running statistics (not trainable — use pre-trained stats)
    for layer in model.layers:
        layer.conv.freeze(keys=["batch_norm_running_mean", "batch_norm_running_var"])

    total_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    trainable_params = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    print(f"Total: {total_params:,}, Trainable: {trainable_params:,}")
    print(f"Vocabulary: {len(vocabulary)} BPE tokens, Blank ID: {blank_id}")

    # Optionally freeze encoder (only train CTC head)
    if args.freeze_encoder:
        model.subsampling.freeze()
        for layer in model.layers:
            layer.freeze()
        trainable_params = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
        print(f"Encoder frozen. Trainable: {trainable_params:,}")

    # Load dataset
    print(f"Loading dataset from: {args.dataset_dir}")
    train_samples = load_cv_dataset(args.dataset_dir, "train")
    val_samples = load_cv_dataset(args.dataset_dir, "validation")
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    np.random.seed(args.seed)
    np.random.shuffle(train_samples)

    # LR schedule
    if args.warmup_steps > 0:
        lr_schedule = optim.join_schedules(
            [
                optim.linear_schedule(init=1e-7, end=args.learning_rate, steps=args.warmup_steps),
                optim.cosine_decay(
                    init=args.learning_rate, decay_steps=args.iters - args.warmup_steps
                ),
            ],
            [args.warmup_steps],
        )
    else:
        lr_schedule = args.learning_rate

    optimizer = optim.AdamW(learning_rate=lr_schedule, weight_decay=args.weight_decay)

    # Resume from checkpoint
    start_iter = 1
    if args.resume:
        ckpt_path = args.resume
        if ckpt_path == "latest":
            ckpts = sorted(
                [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint_")]
            )
            if not ckpts:
                print("No checkpoints found, starting from scratch")
                ckpt_path = None
            else:
                ckpt_path = os.path.join(args.output_dir, ckpts[-1])
        if ckpt_path:
            print(f"Resuming from: {ckpt_path}")
            weights = mx.load(os.path.join(ckpt_path, "model.safetensors"))
            model.load_weights(list(weights.items()))
            opt_state_path = os.path.join(ckpt_path, "optimizer.safetensors")
            if os.path.exists(opt_state_path):
                opt_weights = mx.load(opt_state_path)
                optimizer.state = tree_unflatten(list(opt_weights.items()))
            train_state_path = os.path.join(ckpt_path, "train_state.json")
            if os.path.exists(train_state_path):
                with open(train_state_path) as f:
                    train_state = json.load(f)
                start_iter = train_state["iter"] + 1
            # Re-freeze after loading weights
            for layer in model.layers:
                layer.conv.freeze(keys=["batch_norm_running_mean", "batch_norm_running_var"])
            if args.freeze_encoder:
                model.subsampling.freeze()
                for layer in model.layers:
                    layer.freeze()
            mx.eval(model.parameters(), optimizer.state)
            print(f"Resumed at iter {start_iter}")

    # Training setup
    loss_and_grad = nn.value_and_grad(
        model, lambda m, mel, tgt, il, tl, bid: loss_fn(m, mel, tgt, il, tl, bid)
    )

    os.makedirs(args.output_dir, exist_ok=True)

    model.train()
    sample_idx = 0
    losses = 0.0
    train_time = 0.0
    accum_steps = args.grad_accumulation
    micro_batch = args.batch_size
    effective_batch = micro_batch * accum_steps

    print(f"\nTraining for {args.iters} iters (from {start_iter})")
    print(f"Batch: {effective_batch} ({micro_batch}x{accum_steps} accum), LR: {args.learning_rate}")

    for it in range(start_iter, args.iters + 1):
        tic = time.perf_counter()

        accumulated_grad = None
        step_loss = 0.0
        accum_count = 0

        for _ in range(accum_steps):
            if sample_idx + micro_batch > len(train_samples):
                np.random.shuffle(train_samples)
                sample_idx = 0
            batch = train_samples[sample_idx : sample_idx + micro_batch]
            sample_idx += micro_batch

            mel_batch, targets, input_lengths, target_lengths = prepare_batch(
                batch, tokenizer, normalizer
            )
            if mel_batch is None:
                continue

            loss, grad = loss_and_grad(
                model, mel_batch, targets, input_lengths, target_lengths, blank_id
            )
            mx.eval(loss)
            step_loss += loss.item()
            accum_count += 1

            if accumulated_grad is None:
                accumulated_grad = grad
            else:
                accumulated_grad = tree_map(lambda a, b: a + b, accumulated_grad, grad)

        if accumulated_grad is None:
            continue

        # Average gradients and update
        if accum_count > 1:
            accumulated_grad = tree_map(lambda g: g * (1.0 / accum_count), accumulated_grad)
        optimizer.update(model, accumulated_grad)
        mx.eval(model.parameters(), optimizer.state)

        loss_val = step_loss / max(accum_count, 1)
        losses += loss_val
        train_time += time.perf_counter() - tic

        # Log
        if it % args.log_every == 0:
            avg_loss = losses / args.log_every
            lr = (
                optimizer.learning_rate.item()
                if hasattr(optimizer.learning_rate, 'item')
                else args.learning_rate
            )
            peak_mem = mx.get_peak_memory() / 1e9
            active_mem = mx.get_active_memory() / 1e9
            print(
                f"Iter {it}/{args.iters}: "
                f"loss={avg_loss:.4f}, lr={lr:.2e}, "
                f"mem={active_mem:.1f}/{peak_mem:.1f}GB, "
                f"it/s={args.log_every / train_time:.2f}"
            )
            losses = 0.0
            train_time = 0.0

        # Evaluate
        if it % args.eval_every == 0:
            wer_val = evaluate_wer(
                model, val_samples, vocabulary, normalizer, max_samples=args.val_samples
            )
            print(f"Iter {it}: val_wer={wer_val:.4f} ({wer_val * 100:.2f}%)")
            model.train()

        # Save checkpoint
        if it % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_{it:07d}")
            save_checkpoint(model, optimizer, it, vocabulary, model_config, ckpt_path)
            print(f"Iter {it}: saved to {ckpt_path}")

    # Save final model
    final_path = os.path.join(args.output_dir, "final")
    save_checkpoint(model, optimizer, args.iters, vocabulary, model_config, final_path)
    print(f"\nDone. Final model: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Conformer-CTC with MLX")
    parser.add_argument("--model", type=str, default="nemo_models/ctc_extracted",
                        help="Path to extracted NeMo CTC model directory")
    parser.add_argument("--dataset-dir", type=str, required=True,
                        help="Path to Common Voice dataset (with clips/ and train.tsv)")
    parser.add_argument("--output-dir", type=str, default="./output_mlx_conformer")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--iters", type=int, default=10000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="Freeze encoder, only train CTC head")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--val-samples", type=int, default=200)
    parser.add_argument("--grad-accumulation", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path, or 'latest' to auto-detect")
    args = parser.parse_args()
    train(args)
