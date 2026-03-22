"""
MLX implementation of NVIDIA's Conformer-CTC model.
Loads weights from NeMo .nemo checkpoint (extracted).

Usage:
    python mlx_conformer.py --model nemo_models/ctc_extracted --audio test.mp3
    python mlx_conformer.py --model nemo_models/ctc_extracted --eval --dataset-dir /path/to/cv/be
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
import numpy as np


# ─── Mel Spectrogram (matches NeMo AudioToMelSpectrogramPreprocessor) ───────

def compute_mel_spectrogram(audio, sample_rate=16000, n_fft=512, win_length=400,
                            hop_length=160, n_mels=80, dither=1e-5):
    """Compute log-mel spectrogram matching NeMo's preprocessor.

    NeMo config: window_size=0.025 (400 samples), window_stride=0.01 (160 samples),
    n_fft=512, features=80, normalize=per_feature, log=true, dither=1e-5.
    """
    audio = np.array(audio, dtype=np.float32)

    # Dither
    if dither > 0:
        audio = audio + dither * np.random.randn(*audio.shape).astype(np.float32)

    # STFT
    window = np.hanning(win_length + 1)[:-1].astype(np.float32)
    # Pad audio
    pad_length = (n_fft - hop_length) // 2
    audio = np.pad(audio, (pad_length, pad_length), mode="reflect")

    n_frames = 1 + (len(audio) - n_fft) // hop_length
    frames = np.stack([audio[i * hop_length : i * hop_length + n_fft] for i in range(n_frames)])
    frames = frames * np.pad(window, (0, n_fft - win_length))

    spectrum = np.fft.rfft(frames, n=n_fft)
    power = np.abs(spectrum) ** 2

    # Mel filterbank
    mel_fb = _mel_filterbank(n_mels, n_fft, sample_rate)
    mel_spec = power @ mel_fb.T

    # Log
    mel_spec = np.log(np.maximum(mel_spec, 1e-10))

    # Per-feature normalization (zero mean, unit variance per mel bin)
    mean = mel_spec.mean(axis=0, keepdims=True)
    std = mel_spec.std(axis=0, keepdims=True)
    mel_spec = (mel_spec - mean) / np.maximum(std, 1e-5)

    return mx.array(mel_spec, dtype=mx.float32)


def _mel_filterbank(n_mels, n_fft, sample_rate):
    """Create mel filterbank matrix."""
    low_freq = 0.0
    high_freq = sample_rate / 2.0

    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_low = hz_to_mel(low_freq)
    mel_high = hz_to_mel(high_freq)
    mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)

    for i in range(n_mels):
        left, center, right = bin_points[i], bin_points[i + 1], bin_points[i + 2]
        for j in range(left, center):
            fb[i, j] = (j - left) / max(center - left, 1)
        for j in range(center, right):
            fb[i, j] = (right - j) / max(right - center, 1)

    return fb


# ─── Conformer Modules ─────────────────────────────────────────────────────

class ConvSubsampling(nn.Module):
    """Striding conv subsampling (factor 4): 2 × Conv2d(stride=2) + Linear."""

    def __init__(self, d_model, feat_in=80):
        super().__init__()
        self.conv1_weight = None  # Loaded from checkpoint
        self.conv1_bias = None
        self.conv2_weight = None
        self.conv2_bias = None
        self.out = nn.Linear(d_model * (feat_in // 4), d_model)

    def __call__(self, x):
        # x: (batch, time, feat=80)
        b, t, f = x.shape
        # MLX conv2d: input (B, H, W, C) — treat time as H, freq as W, 1 channel
        x = x.reshape(b, t, f, 1)

        # Conv2d with stride 2
        x = self._conv2d(x, self.conv1_weight, self.conv1_bias, stride=2)
        x = nn.relu(x)
        x = self._conv2d(x, self.conv2_weight, self.conv2_bias, stride=2)
        x = nn.relu(x)

        # NeMo order: (B, C, T', F') → transpose(1,2) → (B, T', C, F') → reshape (B, T', C*F')
        # MLX output is (B, T', F', C) — need to swap F' and C to match NeMo's flatten order
        b, t2, f2, c = x.shape
        x = x.transpose(0, 1, 3, 2).reshape(b, t2, c * f2)  # (B, T', C*F')

        # Linear projection to d_model
        x = self.out(x)
        return x

    def _conv2d(self, x, weight, bias, stride=2):
        """Conv2d. x: (B,H,W,Cin), weight: (Cout,Cin,kH,kW) from PyTorch."""
        # MLX wants weight as (Cout, kH, kW, Cin)
        w = weight.transpose(0, 2, 3, 1)
        out = mx.conv2d(x, w, stride=stride, padding=1)
        if bias is not None:
            out = out + bias
        return out


class FeedForward(nn.Module):
    """FFN with SiLU activation and dropout."""

    def __init__(self, d_model, expansion=4):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model * expansion)
        self.linear2 = nn.Linear(d_model * expansion, d_model)

    def __call__(self, x):
        x = self.linear1(x)
        x = nn.silu(x)
        x = self.linear2(x)
        return x


class RelPositionalEncoding:
    """Relative sinusoidal positional encoding (NeMo-style, descending)."""

    @staticmethod
    def get_encoding(length, d_model):
        """Generate 2*length-1 position encodings from (length-1) down to -(length-1)."""
        positions = np.arange(length - 1, -length, -1, dtype=np.float32)[:, None]
        div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * -(math.log(10000.0) / d_model))
        pe = np.zeros((2 * length - 1, d_model), dtype=np.float32)
        pe[:, 0::2] = np.sin(positions * div_term)
        pe[:, 1::2] = np.cos(positions * div_term)
        return mx.array(pe)


class RelPosMultiHeadAttention(nn.Module):
    """Multi-head attention with relative positional encoding (Shaw-style)."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.linear_pos = nn.Linear(d_model, d_model, bias=False)

        # Untied biases
        self.pos_bias_u = None  # (n_heads, d_head)
        self.pos_bias_v = None

    def __call__(self, x, pos_emb):
        B, T, _ = x.shape
        H, D = self.n_heads, self.d_head

        # Q, K, V projections
        q = self.linear_q(x).reshape(B, T, H, D)
        k = self.linear_k(x).reshape(B, T, H, D).transpose(0, 2, 1, 3)  # (B, H, T, D)
        v = self.linear_v(x).reshape(B, T, H, D).transpose(0, 2, 1, 3)

        # Positional encoding projection: (2T-1, d_model) → (2T-1, H, D)
        p = self.linear_pos(pos_emb).reshape(-1, H, D).transpose(1, 0, 2)  # (H, 2T-1, D)

        # Content attention: (q + bias_u) @ k^T → (B, H, T, T)
        q_u = (q + self.pos_bias_u[None, None, :, :]).transpose(0, 2, 1, 3)  # (B, H, T, D)
        matrix_ac = q_u @ k.transpose(0, 1, 3, 2)  # (B, H, T, T)

        # Position attention: (q + bias_v) @ p^T → (B, H, T, 2T-1)
        q_v = (q + self.pos_bias_v[None, None, :, :]).transpose(0, 2, 1, 3)  # (B, H, T, D)
        matrix_bd = q_v @ p.transpose(0, 2, 1)[None, :, :, :]  # (B, H, T, 2T-1)

        # Relative shift: align position scores to correct relative positions
        matrix_bd = self._rel_shift(matrix_bd)
        # Trim to match matrix_ac size
        matrix_bd = matrix_bd[:, :, :, :T]

        scores = (matrix_ac + matrix_bd) * self.scale
        attn = mx.softmax(scores, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, self.d_model)
        return self.linear_out(out)

    def _rel_shift(self, x):
        """NeMo-style relative shift for positional attention scores.
        Input: (B, H, T, 2T-1) → Output: (B, H, T, 2T-1) with shifted alignment.
        """
        B, H, T, pos_len = x.shape
        # Pad zero column on the left
        x = mx.pad(x, [(0, 0), (0, 0), (0, 0), (1, 0)])  # (B, H, T, 2T)
        # Reshape to interleave
        x = x.reshape(B, H, pos_len + 1, T)  # (B, H, 2T, T)
        # Drop first row and reshape back
        x = x[:, :, 1:, :].reshape(B, H, T, pos_len)  # (B, H, T, 2T-1)
        return x


class ConvModule(nn.Module):
    """Conformer convolution module: pointwise → GLU → depthwise → BN → pointwise."""

    def __init__(self, d_model, kernel_size=31):
        super().__init__()
        self.pointwise_conv1_weight = None  # (2*d, d, 1)
        self.pointwise_conv1_bias = None
        self.depthwise_conv_weight = None  # (d, 1, k)
        self.depthwise_conv_bias = None
        self.batch_norm_weight = None
        self.batch_norm_bias = None
        self.batch_norm_running_mean = None
        self.batch_norm_running_var = None
        self.pointwise_conv2_weight = None  # (d, d, 1)
        self.pointwise_conv2_bias = None
        self.kernel_size = kernel_size
        self.d_model = d_model

    def __call__(self, x):
        # x: (B, T, D)
        B, T, D = x.shape

        # Pointwise conv1 (expansion with GLU)
        # Implemented as linear: (B, T, D) @ (D, 2D) → (B, T, 2D)
        w1 = self.pointwise_conv1_weight.squeeze(-1)  # (2D, D)
        out = x @ w1.T + self.pointwise_conv1_bias
        # GLU: split in half, gate with sigmoid
        out, gate = mx.split(out, 2, axis=-1)
        out = out * mx.sigmoid(gate)

        # Depthwise conv1d: (B, T, D) with groups=D — each channel independently
        pad = self.kernel_size // 2
        out = mx.pad(out, [(0, 0), (pad, pad), (0, 0)])
        # MLX conv1d: input (B, T, C), weight (C_out, kernel, C_in/groups)
        dw = self.depthwise_conv_weight  # (D, 1, K) from NeMo
        # Reshape to MLX conv1d format: (D, K, 1) for groups=D
        dw = dw.transpose(0, 2, 1)  # (D, K, 1)
        out = mx.conv1d(out, dw, groups=self.d_model)
        out = out[:, :T, :]  # trim to original length

        out = out + self.depthwise_conv_bias

        # Batch norm (inference mode — use running stats)
        out = (out - self.batch_norm_running_mean) / mx.sqrt(self.batch_norm_running_var + 1e-5)
        out = out * self.batch_norm_weight + self.batch_norm_bias

        out = nn.silu(out)

        # Pointwise conv2
        w2 = self.pointwise_conv2_weight.squeeze(-1)  # (D, D)
        out = out @ w2.T + self.pointwise_conv2_bias

        return out


class ConformerBlock(nn.Module):
    """Single Conformer block: FFN½ → Attn → Conv → FFN½ → LayerNorm."""

    def __init__(self, d_model, n_heads, conv_kernel_size=31, ff_expansion=4):
        super().__init__()
        self.norm_feed_forward1 = nn.LayerNorm(d_model)
        self.feed_forward1 = FeedForward(d_model, ff_expansion)
        self.norm_self_att = nn.LayerNorm(d_model)
        self.self_attn = RelPosMultiHeadAttention(d_model, n_heads)
        self.norm_conv = nn.LayerNorm(d_model)
        self.conv = ConvModule(d_model, conv_kernel_size)
        self.norm_feed_forward2 = nn.LayerNorm(d_model)
        self.feed_forward2 = FeedForward(d_model, ff_expansion)
        self.norm_out = nn.LayerNorm(d_model)

    def __call__(self, x, pos_emb):
        # Macaron FFN (half-step)
        x = x + 0.5 * self.feed_forward1(self.norm_feed_forward1(x))
        # Self-attention with relative positional encoding
        x = x + self.self_attn(self.norm_self_att(x), pos_emb)
        # Convolution
        x = x + self.conv(self.norm_conv(x))
        # Macaron FFN (half-step)
        x = x + 0.5 * self.feed_forward2(self.norm_feed_forward2(x))
        # Final layer norm
        x = self.norm_out(x)
        return x


class ConformerCTC(nn.Module):
    """Full Conformer-CTC model."""

    def __init__(self, n_layers=18, d_model=512, n_heads=8, feat_in=80,
                 conv_kernel_size=31, ff_expansion=4, num_classes=129):
        super().__init__()
        self.d_model = d_model
        self.subsampling = ConvSubsampling(d_model, feat_in)
        self.layers = [
            ConformerBlock(d_model, n_heads, conv_kernel_size, ff_expansion)
            for _ in range(n_layers)
        ]
        # CTC decoder: 1D conv with kernel=1
        self.decoder_weight = None  # (num_classes, d_model, 1)
        self.decoder_bias = None  # (num_classes,)
        self.num_classes = num_classes

    def __call__(self, mel):
        """Forward pass.

        Args:
            mel: (batch, time, n_mels) log-mel spectrogram

        Returns:
            log_probs: (batch, time', num_classes) log probabilities
        """
        # Subsampling + scaling (NeMo: scaling is done in pos_enc layer)
        x = self.subsampling(mel)
        x = x * (self.d_model ** 0.5)

        # Generate positional encoding once for all layers
        T = x.shape[1]
        pos_emb = RelPositionalEncoding.get_encoding(T, self.d_model)

        for layer in self.layers:
            x = layer(x, pos_emb)

        # CTC head: linear projection (equiv to 1D conv with kernel=1)
        w = self.decoder_weight.squeeze(-1)  # (num_classes, d_model)
        logits = x @ w.T + self.decoder_bias

        return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


# ─── Weight Loading ─────────────────────────────────────────────────────────

def load_nemo_ctc(model_dir):
    """Load NeMo CTC model weights into MLX ConformerCTC."""
    import torch

    model_dir = Path(model_dir)
    weights = torch.load(model_dir / "model_weights.ckpt", map_location="cpu", weights_only=False)

    model = ConformerCTC()

    # Conv subsampling
    model.subsampling.conv1_weight = mx.array(weights["encoder.pre_encode.conv.0.weight"].numpy())
    model.subsampling.conv1_bias = mx.array(weights["encoder.pre_encode.conv.0.bias"].numpy())
    model.subsampling.conv2_weight = mx.array(weights["encoder.pre_encode.conv.2.weight"].numpy())
    model.subsampling.conv2_bias = mx.array(weights["encoder.pre_encode.conv.2.bias"].numpy())
    model.subsampling.out.weight = mx.array(weights["encoder.pre_encode.out.weight"].numpy())
    model.subsampling.out.bias = mx.array(weights["encoder.pre_encode.out.bias"].numpy())

    # Conformer layers
    for i, layer in enumerate(model.layers):
        prefix = f"encoder.layers.{i}"

        # FFN 1
        layer.norm_feed_forward1.weight = mx.array(weights[f"{prefix}.norm_feed_forward1.weight"].numpy())
        layer.norm_feed_forward1.bias = mx.array(weights[f"{prefix}.norm_feed_forward1.bias"].numpy())
        layer.feed_forward1.linear1.weight = mx.array(weights[f"{prefix}.feed_forward1.linear1.weight"].numpy())
        layer.feed_forward1.linear1.bias = mx.array(weights[f"{prefix}.feed_forward1.linear1.bias"].numpy())
        layer.feed_forward1.linear2.weight = mx.array(weights[f"{prefix}.feed_forward1.linear2.weight"].numpy())
        layer.feed_forward1.linear2.bias = mx.array(weights[f"{prefix}.feed_forward1.linear2.bias"].numpy())

        # Self-attention
        layer.norm_self_att.weight = mx.array(weights[f"{prefix}.norm_self_att.weight"].numpy())
        layer.norm_self_att.bias = mx.array(weights[f"{prefix}.norm_self_att.bias"].numpy())
        layer.self_attn.linear_q.weight = mx.array(weights[f"{prefix}.self_attn.linear_q.weight"].numpy())
        layer.self_attn.linear_q.bias = mx.array(weights[f"{prefix}.self_attn.linear_q.bias"].numpy())
        layer.self_attn.linear_k.weight = mx.array(weights[f"{prefix}.self_attn.linear_k.weight"].numpy())
        layer.self_attn.linear_k.bias = mx.array(weights[f"{prefix}.self_attn.linear_k.bias"].numpy())
        layer.self_attn.linear_v.weight = mx.array(weights[f"{prefix}.self_attn.linear_v.weight"].numpy())
        layer.self_attn.linear_v.bias = mx.array(weights[f"{prefix}.self_attn.linear_v.bias"].numpy())
        layer.self_attn.linear_out.weight = mx.array(weights[f"{prefix}.self_attn.linear_out.weight"].numpy())
        layer.self_attn.linear_out.bias = mx.array(weights[f"{prefix}.self_attn.linear_out.bias"].numpy())
        layer.self_attn.linear_pos.weight = mx.array(weights[f"{prefix}.self_attn.linear_pos.weight"].numpy())
        layer.self_attn.pos_bias_u = mx.array(weights[f"{prefix}.self_attn.pos_bias_u"].numpy())
        layer.self_attn.pos_bias_v = mx.array(weights[f"{prefix}.self_attn.pos_bias_v"].numpy())

        # Conv module
        layer.norm_conv.weight = mx.array(weights[f"{prefix}.norm_conv.weight"].numpy())
        layer.norm_conv.bias = mx.array(weights[f"{prefix}.norm_conv.bias"].numpy())
        layer.conv.pointwise_conv1_weight = mx.array(weights[f"{prefix}.conv.pointwise_conv1.weight"].numpy())
        layer.conv.pointwise_conv1_bias = mx.array(weights[f"{prefix}.conv.pointwise_conv1.bias"].numpy())
        layer.conv.depthwise_conv_weight = mx.array(weights[f"{prefix}.conv.depthwise_conv.weight"].numpy())
        layer.conv.depthwise_conv_bias = mx.array(weights[f"{prefix}.conv.depthwise_conv.bias"].numpy())
        layer.conv.batch_norm_weight = mx.array(weights[f"{prefix}.conv.batch_norm.weight"].numpy())
        layer.conv.batch_norm_bias = mx.array(weights[f"{prefix}.conv.batch_norm.bias"].numpy())
        layer.conv.batch_norm_running_mean = mx.array(weights[f"{prefix}.conv.batch_norm.running_mean"].numpy())
        layer.conv.batch_norm_running_var = mx.array(weights[f"{prefix}.conv.batch_norm.running_var"].numpy())
        layer.conv.pointwise_conv2_weight = mx.array(weights[f"{prefix}.conv.pointwise_conv2.weight"].numpy())
        layer.conv.pointwise_conv2_bias = mx.array(weights[f"{prefix}.conv.pointwise_conv2.bias"].numpy())

        # FFN 2
        layer.norm_feed_forward2.weight = mx.array(weights[f"{prefix}.norm_feed_forward2.weight"].numpy())
        layer.norm_feed_forward2.bias = mx.array(weights[f"{prefix}.norm_feed_forward2.bias"].numpy())
        layer.feed_forward2.linear1.weight = mx.array(weights[f"{prefix}.feed_forward2.linear1.weight"].numpy())
        layer.feed_forward2.linear1.bias = mx.array(weights[f"{prefix}.feed_forward2.linear1.bias"].numpy())
        layer.feed_forward2.linear2.weight = mx.array(weights[f"{prefix}.feed_forward2.linear2.weight"].numpy())
        layer.feed_forward2.linear2.bias = mx.array(weights[f"{prefix}.feed_forward2.linear2.bias"].numpy())

        # Output norm
        layer.norm_out.weight = mx.array(weights[f"{prefix}.norm_out.weight"].numpy())
        layer.norm_out.bias = mx.array(weights[f"{prefix}.norm_out.bias"].numpy())

    # CTC decoder
    model.decoder_weight = mx.array(weights["decoder.decoder_layers.0.weight"].numpy())
    model.decoder_bias = mx.array(weights["decoder.decoder_layers.0.bias"].numpy())

    mx.eval(model.parameters())
    return model


# ─── CTC Greedy Decode ──────────────────────────────────────────────────────

def ctc_greedy_decode(log_probs, vocabulary):
    """Greedy CTC decoding: take argmax, collapse repeats, remove blanks."""
    # log_probs: (time, num_classes), blank = last index
    blank_id = log_probs.shape[-1] - 1
    indices = mx.argmax(log_probs, axis=-1)
    indices = np.array(indices)

    tokens = []
    prev = -1
    for idx in indices:
        if idx != prev and idx != blank_id:
            tokens.append(idx)
        prev = idx

    # Join BPE tokens
    text = ""
    for t in tokens:
        piece = vocabulary[t]
        piece = piece.replace("▁", " ")
        text += piece

    return text.strip()


# ─── Main ───────────────────────────────────────────────────────────────────

def load_vocabulary(model_dir):
    """Load vocabulary from NeMo config."""
    import yaml
    with open(Path(model_dir) / "model_config.yaml") as f:
        config = yaml.safe_load(f)
    return config["decoder"]["vocabulary"]


def load_audio_file(path):
    """Load audio file to 16kHz mono numpy array."""
    import subprocess
    cmd = [
        "ffmpeg", "-i", str(path), "-ar", "16000", "-ac", "1",
        "-f", "f32le", "-hide_banner", "-loglevel", "error", "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True)
    return np.frombuffer(result.stdout, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="MLX Conformer-CTC inference")
    parser.add_argument("--model", type=str, default="nemo_models/ctc_extracted")
    parser.add_argument("--audio", type=str, help="Audio file to transcribe")
    parser.add_argument("--eval", action="store_true", help="Run WER evaluation")
    parser.add_argument("--dataset-dir", type=str)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    print(f"Loading model from: {args.model}")
    model = load_nemo_ctc(args.model)
    vocabulary = load_vocabulary(args.model)
    print(f"Model loaded. {len(vocabulary)} vocab tokens, 18 Conformer layers")

    if args.audio:
        audio = load_audio_file(args.audio)
        mel = compute_mel_spectrogram(audio)
        mel = mx.expand_dims(mel, axis=0)  # add batch dim

        log_probs = model(mel)
        mx.eval(log_probs)
        text = ctc_greedy_decode(log_probs[0], vocabulary)
        print(f"\nTranscription: {text}")

    elif args.eval:
        from jiwer import wer
        import sys
        sys.path.insert(0, ".")
        from belarusian_text_normalizer import BelarusianTextNormalizer

        normalizer = BelarusianTextNormalizer()

        tsv_path = os.path.join(args.dataset_dir, "test.tsv")
        clips_dir = os.path.join(args.dataset_dir, "clips")
        samples = []
        with open(tsv_path, "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                samples.append((os.path.join(clips_dir, row["path"]), row["sentence"]))

        if args.max_samples > 0:
            samples = samples[:args.max_samples]

        print(f"Evaluating {len(samples)} samples...")
        refs, preds = [], []
        tic = time.perf_counter()

        for i, (audio_path, ref) in enumerate(samples):
            try:
                audio = load_audio_file(audio_path)
                mel = compute_mel_spectrogram(audio)
                mel = mx.expand_dims(mel, axis=0)
                log_probs = model(mel)
                mx.eval(log_probs)
                text = ctc_greedy_decode(log_probs[0], vocabulary)
            except Exception as e:
                continue

            norm_ref = normalizer(ref)
            norm_pred = normalizer(text)
            if norm_ref.strip():
                refs.append(norm_ref)
                preds.append(norm_pred)

            if (i + 1) % 50 == 0:
                w = wer(refs, preds)
                speed = (i + 1) / (time.perf_counter() - tic)
                print(f"  [{i+1}/{len(samples)}] WER={w:.4f} ({speed:.1f} samples/s)")

        elapsed = time.perf_counter() - tic
        final_wer = wer(refs, preds)
        print(f"\n{'=' * 50}")
        print(f"Model: {args.model}")
        print(f"WER: {final_wer:.4f} ({final_wer*100:.2f}%)")
        print(f"Samples: {len(refs)}, Time: {elapsed:.1f}s ({len(refs)/elapsed:.1f} samples/s)")
        print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
