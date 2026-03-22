"""
Fine-tune NVIDIA Conformer-Transducer (RNN-T) using Apple MLX framework.
Loads pre-trained weights from extracted NeMo checkpoint, fine-tunes with RNN-T loss.

Usage:
    python mlx_finetune_transducer.py \
        --model nemo_models/transducer_extracted \
        --dataset-dir /path/to/cv-corpus/be \
        --output-dir ./output_mlx_transducer \
        --batch-size 2 \
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
    ConformerBlock,
    ConformerTransducer,
    ConvSubsampling,
    RelPositionalEncoding,
    compute_mel_spectrogram,
    load_audio_file,
    load_nemo_transducer,
    load_vocabulary,
    transducer_greedy_decode,
)


# ─── Trainable LSTM Cell ────────────────────────────────────────────────────
# The LSTMCell in mlx_conformer.py is a plain class (not nn.Module),
# so its weights are invisible to model.parameters() and can't receive gradients.
# This nn.Module version fixes that for training.


class TrainableLSTMCell(nn.Module):
    """nn.Module LSTM cell — weights are part of the parameter tree."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = mx.zeros((4 * hidden_size, input_size))
        self.weight_hh = mx.zeros((4 * hidden_size, hidden_size))
        self.bias_ih = mx.zeros((4 * hidden_size,))
        self.bias_hh = mx.zeros((4 * hidden_size,))

    def __call__(self, x, state):
        """x: (batch, input_size), state: (h, c) each (batch, hidden_size)."""
        h, c = state
        gates = x @ self.weight_ih.T + self.bias_ih + h @ self.weight_hh.T + self.bias_hh
        i, f, g, o = mx.split(gates, 4, axis=-1)
        c_new = mx.sigmoid(f) * c + mx.sigmoid(i) * mx.tanh(g)
        h_new = mx.sigmoid(o) * mx.tanh(c_new)
        return h_new, c_new


def make_trainable(model):
    """Replace plain LSTMCell with nn.Module version so weights get gradients."""
    old = model.lstm
    trainable = TrainableLSTMCell(old.input_size, old.hidden_size)
    trainable.weight_ih = old.weight_ih
    trainable.weight_hh = old.weight_hh
    trainable.bias_ih = old.bias_ih
    trainable.bias_hh = old.bias_hh
    model.lstm = trainable
    return model


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
    spe_files = list(model_dir.glob("*_tokenizer*.model"))
    if not spe_files:
        raise FileNotFoundError(f"No SentencePiece tokenizer found in {model_dir}")
    sp = spm.SentencePieceProcessor()
    sp.Load(str(spe_files[0]))
    return sp


def subsample_length(T):
    """Compute time dimension after ConvSubsampling (2x stride-2 convs)."""
    for _ in range(2):
        T = (T - 1) // 2 + 1
    return T


def prepare_batch(samples, tokenizer, normalizer):
    """Prepare a batch for RNN-T training.

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

        if len(audio) < 320:
            continue

        mel = compute_mel_spectrogram(audio)
        T_mel = mel.shape[0]

        norm_text = normalizer(sentence)
        tokens = tokenizer.EncodeAsIds(norm_text)
        if not tokens:
            continue

        # RNN-T requires T_sub > 0
        T_sub = subsample_length(T_mel)
        if T_sub <= 0:
            continue

        mels.append(mel)
        mel_lengths.append(T_mel)
        all_targets.append(tokens)

    if not mels:
        return None, None, None, None

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


# ─── RNN-T Loss ──────────────────────────────────────────────────────────────


def loss_fn(model, mel_batch, all_targets, input_lengths, target_lengths, blank_id):
    """Compute RNN-T loss for a batch.

    For each sample:
    1. Encoder forward (batched across samples)
    2. Prediction network forward (sequential over target tokens)
    3. Joint network: broadcast enc × pred → log_probs (T, U+1, V)
    4. Forward algorithm on the (T, U+1) lattice

    The forward variable alpha[t][u] = log prob of reaching state (t, u),
    where t = encoder frames consumed, u = labels emitted.

    Transitions from (t, u):
      - emit blank → (t+1, u): log P(blank | enc[t], pred[u])
      - emit label[u] → (t, u+1): log P(label[u] | enc[t], pred[u])

    Terminal: loss = -(alpha[T-1][U] + log P(blank | enc[T-1], pred[U]))
    """
    # Batched encoder
    enc_all = model.encode(mel_batch)  # (B, T_max_sub, d_model)

    B = len(input_lengths)
    total_loss = mx.array(0.0)
    valid = 0

    for b in range(B):
        T = input_lengths[b]
        U = target_lengths[b]
        targets = all_targets[b]

        if T <= 0 or U <= 0:
            continue

        enc = enc_all[b, :T, :]  # (T, d_model)

        # Prediction network: process [blank, y0, y1, ..., y_{U-1}]
        pred_tokens = [blank_id] + targets
        h = mx.zeros((1, model.pred_hidden))
        c = mx.zeros((1, model.pred_hidden))
        pred_outputs = []
        for tok in pred_tokens:
            emb = model.embed[tok][None, :]  # (1, pred_hidden)
            h, c = model.lstm(emb, (h, c))
            pred_outputs.append(h[0])  # (pred_hidden,)
        pred_out = mx.stack(pred_outputs)  # (U+1, pred_hidden)

        # Joint network: (T, pred_hidden) + (U+1, pred_hidden) → (T, U+1, V)
        enc_proj = model.joint_enc(enc)  # (T, pred_hidden)
        pred_proj = model.joint_pred(pred_out)  # (U+1, pred_hidden)
        joint = nn.relu(enc_proj[:, None, :] + pred_proj[None, :, :])
        logits = model.joint_out(joint)  # (T, U+1, vocab_size)
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

        # Forward algorithm
        # Initialize column t=0
        alpha_col = [mx.array(0.0)]  # alpha[0][0] = 0
        for u in range(1, U + 1):
            alpha_col.append(alpha_col[-1] + log_probs[0, u - 1, targets[u - 1]])

        # Fill columns t=1..T-1
        for t in range(1, T):
            new_col = [alpha_col[0] + log_probs[t - 1, 0, blank_id]]
            for u in range(1, U + 1):
                from_blank = alpha_col[u] + log_probs[t - 1, u, blank_id]
                from_label = new_col[-1] + log_probs[t, u - 1, targets[u - 1]]
                new_col.append(mx.logaddexp(from_blank, from_label))
            alpha_col = new_col

        # Terminal: emit blank at last frame to consume it
        loss_b = -(alpha_col[U] + log_probs[T - 1, U, blank_id])
        total_loss = total_loss + loss_b
        valid += 1

    if valid == 0:
        return mx.array(0.0)
    return total_loss / valid


# ─── Evaluation ──────────────────────────────────────────────────────────────


def evaluate_wer(model, val_samples, vocabulary, normalizer, max_samples=200):
    """Compute WER on validation set using greedy RNN-T decoding."""
    from jiwer import wer

    model.eval()
    refs, preds = [], []

    for i, (audio_path, ref) in enumerate(val_samples[:max_samples]):
        try:
            audio = load_audio_file(audio_path)
            mel = compute_mel_spectrogram(audio)
            mel = mx.expand_dims(mel, axis=0)
            tokens = model.greedy_decode(mel)
            text = transducer_greedy_decode(tokens, vocabulary)
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
    print(f"Loading Transducer model from: {args.model}")
    model = load_nemo_transducer(args.model)
    model = make_trainable(model)  # Replace LSTMCell with nn.Module version
    vocabulary = load_vocabulary(args.model)
    tokenizer = load_tokenizer(args.model)
    blank_id = model.vocab_size - 1
    normalizer = BelarusianTextNormalizer()

    model_config = {
        "n_layers": len(model.layers),
        "d_model": model.d_model,
        "pred_hidden": model.pred_hidden,
        "vocab_size": model.vocab_size,
    }

    # Freeze batch norm running statistics
    for layer in model.layers:
        layer.conv.freeze(keys=["batch_norm_running_mean", "batch_norm_running_var"])

    total_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    trainable_params = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    print(f"Total: {total_params:,}, Trainable: {trainable_params:,}")
    print(f"Vocabulary: {len(vocabulary)} BPE tokens, Blank ID: {blank_id}")

    # Optionally freeze encoder
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
    parser = argparse.ArgumentParser(description="Fine-tune Conformer-Transducer (RNN-T) with MLX")
    parser.add_argument("--model", type=str, default="nemo_models/transducer_extracted",
                        help="Path to extracted NeMo Transducer model directory")
    parser.add_argument("--dataset-dir", type=str, required=True,
                        help="Path to Common Voice dataset (with clips/ and train.tsv)")
    parser.add_argument("--output-dir", type=str, default="./output_mlx_transducer")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Micro-batch size (RNN-T uses more memory than CTC)")
    parser.add_argument("--iters", type=int, default=10000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="Freeze encoder, only train prediction + joint networks")
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
