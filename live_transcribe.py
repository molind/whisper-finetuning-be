"""
Live microphone transcription using fine-tuned MLX Whisper.

Detects speech using energy-based VAD, then transcribes complete
utterances when silence is detected.

Usage:
    python live_transcribe.py --model ./output_mlx_small/final
    python live_transcribe.py --model mlx-community/whisper-small-mlx
"""

import argparse
import json
import sys
import os
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import sounddevice as sd

sys.path.insert(0, os.path.expanduser("~/tmp/mlx-examples/whisper"))

from mlx_whisper.audio import log_mel_spectrogram, pad_or_trim, N_SAMPLES
from mlx_whisper.load_models import load_model
from mlx_whisper.tokenizer import get_tokenizer
from mlx_whisper.decoding import DecodingOptions, decode


SAMPLE_RATE = 16000


def load_tokenizer_from_model(model_path):
    model_path = Path(model_path)
    if not model_path.exists():
        from huggingface_hub import snapshot_download
        model_path = Path(snapshot_download(repo_id=str(model_path)))
    with open(model_path / "config.json") as f:
        config = json.load(f)
    multilingual = config.get("n_vocab", 51865) >= 51865
    return get_tokenizer(multilingual=multilingual, language="be", task="transcribe")


def transcribe_audio(model, audio_np):
    """Transcribe a numpy audio array using MLX whisper."""
    audio = mx.array(audio_np, dtype=mx.float32)
    audio = pad_or_trim(audio, N_SAMPLES)
    mel = log_mel_spectrogram(audio, n_mels=model.dims.n_mels)
    mel = mx.expand_dims(mel, axis=0)

    options = DecodingOptions(
        language="be", task="transcribe", without_timestamps=True, fp16=False,
    )
    result = decode(model, mel, options)
    return result[0].text.strip()


def get_rms(audio):
    """Root mean square energy of audio chunk."""
    return np.sqrt(np.mean(audio ** 2))


def main():
    parser = argparse.ArgumentParser(description="Live Whisper transcription")
    parser.add_argument("--model", type=str, default="./output_mlx_small/final")
    parser.add_argument("--energy-threshold", type=float, default=0.01,
                        help="RMS energy threshold for speech detection")
    parser.add_argument("--silence-duration", type=float, default=0.8,
                        help="Seconds of silence to end an utterance")
    parser.add_argument("--min-speech-duration", type=float, default=0.3,
                        help="Minimum speech duration to transcribe (seconds)")
    parser.add_argument("--max-speech-duration", type=float, default=25.0,
                        help="Maximum speech duration before forced transcription (seconds)")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = load_model(args.model, dtype=mx.float32)

    # Cast float16 weights to float32
    from mlx.utils import tree_flatten, tree_unflatten
    params = tree_flatten(model.parameters())
    cast_params = [
        (k, v.astype(mx.float32) if v.dtype == mx.float16 else v) for k, v in params
    ]
    model.update(tree_unflatten(cast_params))
    mx.eval(model.parameters())

    tokenizer = load_tokenizer_from_model(args.model)
    print(f"Model loaded. n_mels={model.dims.n_mels}")

    # Warm up model with a silent decode
    dummy = np.zeros(SAMPLE_RATE, dtype=np.float32)
    transcribe_audio(model, dummy)
    print("Model warmed up.")

    # Recording parameters
    chunk_duration = 0.1  # 100ms chunks
    chunk_samples = int(SAMPLE_RATE * chunk_duration)

    print(f"\nListening... (Ctrl+C to stop)")
    print(f"Speak naturally. Transcription appears after you pause.")
    print(f"Energy threshold: {args.energy_threshold}")
    print("=" * 60)

    # Calibrate noise floor
    print("Calibrating noise floor (stay quiet for 1 second)...")
    calibration = sd.rec(SAMPLE_RATE, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    noise_floor = get_rms(calibration)
    threshold = max(args.energy_threshold, noise_floor * 3)
    print(f"Noise floor: {noise_floor:.4f}, threshold: {threshold:.4f}")
    print("=" * 60 + "\n")

    speech_buffer = []
    is_speaking = False
    silence_chunks = 0
    silence_chunks_needed = int(args.silence_duration / chunk_duration)
    min_speech_chunks = int(args.min_speech_duration / chunk_duration)
    max_speech_chunks = int(args.max_speech_duration / chunk_duration)

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=chunk_samples,
    )

    with stream:
        try:
            while True:
                audio_chunk, overflowed = stream.read(chunk_samples)
                mono = audio_chunk[:, 0]
                rms = get_rms(mono)

                if rms > threshold:
                    # Speech detected
                    if not is_speaking:
                        is_speaking = True
                        sys.stdout.write("  [...]")
                        sys.stdout.flush()
                    silence_chunks = 0
                    speech_buffer.append(mono.copy())
                elif is_speaking:
                    # Silence while we were speaking
                    silence_chunks += 1
                    speech_buffer.append(mono.copy())

                    # End of utterance or max duration reached
                    if silence_chunks >= silence_chunks_needed or len(speech_buffer) >= max_speech_chunks:
                        if len(speech_buffer) >= min_speech_chunks:
                            # Trim trailing silence
                            trim = max(0, len(speech_buffer) - silence_chunks)
                            audio = np.concatenate(speech_buffer[:trim])
                            duration = len(audio) / SAMPLE_RATE

                            tic = time.perf_counter()
                            text = transcribe_audio(model, audio)
                            elapsed = time.perf_counter() - tic

                            sys.stdout.write(f"\r\033[K")
                            if text:
                                print(f"  {text}  ({duration:.1f}s, {elapsed:.1f}s)")
                            else:
                                print(f"\r\033[K", end="")

                        else:
                            sys.stdout.write(f"\r\033[K")
                            sys.stdout.flush()

                        speech_buffer = []
                        is_speaking = False
                        silence_chunks = 0

        except KeyboardInterrupt:
            print("\n\nStopped.")


if __name__ == "__main__":
    main()
