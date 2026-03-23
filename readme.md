## Whisper Fine-Tuning for Belarusian

Fine-tuning [OpenAI Whisper](https://github.com/openai/whisper) for Belarusian speech recognition using HuggingFace Transformers and Apple MLX.

Forked from [navalnica/whisper-finetuning-be](https://github.com/navalnica/whisper-finetuning-be). This fork adds MLX training/eval/inference on Apple Silicon and a HuggingFace-to-MLX model converter.

### Results (CommonVoice 24.0 test, 500 samples)

| Model | WER | Speed | Params | Framework |
|-------|-----|-------|--------|-----------|
| **Conformer-Transducer** | **6.29%** | 7.6 s/s | ~120M | MLX ([molind/mlx-conformer](https://github.com/molind/mlx-conformer)) |
| Conformer-CTC | 7.58% | 8.2 s/s | ~120M | MLX ([molind/mlx-conformer](https://github.com/molind/mlx-conformer)) |
| Whisper Small (ales, unfrozen enc) | 8.25% | 2.4 s/s | 244M | MLX |
| Whisper Small (ales, original) | 8.46% | 1.8 s/s | 244M | MLX |
| Whisper Large-v3-turbo (no fine-tune) | 61.71% | 1.1 s/s | 809M | MLX |
| Whisper Small (no fine-tune) | 79.53% | 2.4 s/s | 244M | MLX |

### Published Models

| Model | Link |
|-------|------|
| Whisper Small Belarusian | [ales/whisper-small-belarusian](https://huggingface.co/ales/whisper-small-belarusian) |
| Whisper Base Belarusian | [ales/whisper-base-belarusian](https://huggingface.co/ales/whisper-base-belarusian) |
| Conformer-CTC Belarusian (MLX) | [molind/conformer-ctc-be-mlx](https://huggingface.co/molind/conformer-ctc-be-mlx) |
| Conformer-Transducer Belarusian (MLX) | [molind/conformer-transducer-be-mlx](https://huggingface.co/molind/conformer-transducer-be-mlx) |

## Quick Start

```bash
# Setup
make install-uv && make venv-create && make venv-install

# Training (PyTorch — HuggingFace Trainer)
bash bash_runners/run_small.sh

# Training (MLX — Apple Silicon, ~5x faster)
python mlx_finetune_whisper.py \
    --model mlx-community/whisper-small-mlx \
    --dataset-dir /path/to/cv-corpus/be \
    --freeze-encoder --batch-size 4 --iters 10000

# Resume training
python mlx_finetune_whisper.py \
    --model mlx-community/whisper-small-mlx \
    --dataset-dir /path/to/cv-corpus/be \
    --output-dir ./output_mlx_small \
    --resume latest

# Evaluate (MLX)
python mlx_eval_whisper.py \
    --model ./output_mlx_small/final \
    --dataset-dir /path/to/cv-corpus/be

# Live microphone transcription
python live_transcribe.py --model ./output_mlx_small/final

# Convert HuggingFace model to MLX
python convert_hf_to_mlx.py --model ales/whisper-small-belarusian --output ./ales_mlx
```

## What's in This Fork

**New scripts (MLX, Apple Silicon):**
- `mlx_finetune_whisper.py` — MLX Whisper fine-tuning. Float16→float32 weight casting, checkpoint resume, gradient accumulation, memory-bounded training.
- `mlx_eval_whisper.py` — WER evaluation for MLX Whisper models on CommonVoice.
- `live_transcribe.py` — Real-time microphone transcription with VAD-based speech detection.
- `convert_hf_to_mlx.py` — Convert HuggingFace Whisper models to MLX format.

**Original scripts (PyTorch):**
- `run_speech_recognition_seq2seq_streaming.py` — HuggingFace Seq2Seq training with streaming dataset support.
- `run_eval_whisper_streaming.py` — PyTorch WER evaluation.
- `custom_trainer.py` — Custom LR scheduler with `learning_rate_end` support.
- `belarusian_text_normalizer.py` — Text normalizer preserving Belarusian apostrophe.

## Key Findings

**MLX float32 casting required:** MLX whisper models from `mlx-community` store weights as float16. `load_model(dtype=mx.float32)` does not cast them. Explicit casting after load prevents AdamW overflow → NaN loss.

**Memory management:** `mx.set_cache_limit()` is essential for long training runs. Without it, MLX uses all available memory and swaps heavily.

**Unfreezing encoder helps:** Fine-tuning with unfrozen encoder (all 244M params) improves WER from 8.46% to 8.25%, but overfits quickly. Weight decay 0.1 stabilizes training.

**Conformer beats Whisper for single-language ASR:** NVIDIA's Conformer-Transducer achieves 6.29% WER with half the parameters and 3x faster inference. See [molind/mlx-conformer](https://github.com/molind/mlx-conformer).

## Requirements

- Python 3.13+, `uv` package manager
- Apple Silicon Mac (for MLX scripts) or CUDA GPU (for PyTorch scripts)
- ffmpeg
- CommonVoice dataset (Belarusian)
