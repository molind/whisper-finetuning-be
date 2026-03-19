"""
Evaluate MLX Whisper model on CommonVoice test set.
Computes WER using BelarusianTextNormalizer.

Usage:
    python mlx_eval_whisper.py --model ./output_mlx_small/final --dataset-dir /path/to/cv/be
    python mlx_eval_whisper.py --model mlx-community/whisper-small-mlx --dataset-dir /path/to/cv/be
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from jiwer import wer
from mlx.utils import tree_flatten, tree_unflatten

sys.path.insert(0, os.path.expanduser("~/tmp/mlx-examples/whisper"))

from mlx_whisper.audio import load_audio, log_mel_spectrogram, pad_or_trim, N_SAMPLES
from mlx_whisper.load_models import load_model
from mlx_whisper.tokenizer import get_tokenizer
from mlx_whisper.decoding import DecodingOptions, decode

from belarusian_text_normalizer import BelarusianTextNormalizer


def load_tokenizer_from_model(model_path):
    model_path = Path(model_path)
    if not model_path.exists():
        from huggingface_hub import snapshot_download
        model_path = Path(snapshot_download(repo_id=str(model_path)))
    with open(model_path / "config.json") as f:
        config = json.load(f)
    multilingual = config.get("n_vocab", 51865) >= 51865
    return get_tokenizer(multilingual=multilingual, language="be", task="transcribe")


def load_cv_dataset(dataset_dir, split="test"):
    split_to_file = {"train": "train.tsv", "validation": "dev.tsv", "dev": "dev.tsv", "test": "test.tsv"}
    tsv_path = os.path.join(dataset_dir, split_to_file[split])
    clips_dir = os.path.join(dataset_dir, "clips")
    samples = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            audio_path = os.path.join(clips_dir, row["path"])
            samples.append((audio_path, row["sentence"]))
    return samples


def transcribe(model, audio_path, n_mels):
    try:
        audio = load_audio(audio_path)
    except Exception as e:
        return None
    audio = pad_or_trim(audio, N_SAMPLES)
    mel = log_mel_spectrogram(audio, n_mels=n_mels)
    mel = mx.expand_dims(mel, axis=0)
    options = DecodingOptions(language="be", task="transcribe", without_timestamps=True, fp16=False)
    result = decode(model, mel, options)
    return result[0].text.strip()


def main():
    parser = argparse.ArgumentParser(description="Evaluate MLX Whisper on CommonVoice")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit samples (0 = all)")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = load_model(args.model, dtype=mx.float32)

    # Cast float16 weights to float32
    params = tree_flatten(model.parameters())
    cast_params = [(k, v.astype(mx.float32) if v.dtype == mx.float16 else v) for k, v in params]
    model.update(tree_unflatten(cast_params))
    mx.eval(model.parameters())

    tokenizer = load_tokenizer_from_model(args.model)
    normalizer = BelarusianTextNormalizer()
    n_mels = model.dims.n_mels

    print(f"Loading dataset: {args.dataset_dir} ({args.split})")
    samples = load_cv_dataset(args.dataset_dir, args.split)
    if args.max_samples > 0:
        samples = samples[:args.max_samples]
    print(f"Evaluating {len(samples)} samples...")

    references = []
    predictions = []
    errors = 0
    tic = time.perf_counter()

    for i, (audio_path, reference) in enumerate(samples):
        pred = transcribe(model, audio_path, n_mels)
        if pred is None:
            errors += 1
            continue

        norm_ref = normalizer(reference)
        norm_pred = normalizer(pred)

        # Skip empty references
        if not norm_ref.strip():
            continue

        references.append(norm_ref)
        predictions.append(norm_pred)

        if (i + 1) % 100 == 0:
            partial_wer = wer(references, predictions)
            elapsed = time.perf_counter() - tic
            speed = (i + 1) / elapsed
            print(f"  [{i+1}/{len(samples)}] WER={partial_wer:.4f} ({speed:.1f} samples/s)")

    elapsed = time.perf_counter() - tic
    final_wer = wer(references, predictions)

    print(f"\n{'=' * 50}")
    print(f"Model: {args.model}")
    print(f"Split: {args.split} ({len(references)} samples, {errors} errors)")
    print(f"WER: {final_wer:.4f} ({final_wer*100:.2f}%)")
    print(f"Time: {elapsed:.1f}s ({len(references)/elapsed:.1f} samples/s)")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
