# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fine-tuning OpenAI's Whisper model for Belarusian speech recognition using HuggingFace Transformers and Apple MLX. Published models: `ales/whisper-small-belarusian`, `ales/whisper-base-belarusian`. Dataset: CommonVoice 24.0 (Belarusian).

## Commands

```bash
# Setup
make install-uv          # Install uv package manager
make venv-create          # Create Python 3.13+ venv
make venv-install         # Install dependencies (uv sync)
make server-install-system-deps  # Install ffmpeg, git-lfs, tmux

# Training (PyTorch — HuggingFace Trainer)
python run_speech_recognition_seq2seq_streaming.py <args>  # or use bash runners:
bash bash_runners/run_tiny_debug.sh   # Quick debug run (tiny model, 500 steps)
bash bash_runners/run_small.sh        # Fine-tune small model
bash bash_runners/run_base.sh         # Fine-tune base model

# Training (MLX — Apple Silicon)
python mlx_finetune_whisper.py <args>  # Requires mlx-whisper from ~/tmp/mlx-examples/whisper
python mlx_finetune_whisper.py \
    --model mlx-community/whisper-small-mlx \
    --dataset-dir /path/to/cv-corpus/be \
    --freeze-encoder --batch-size 4 --iters 10000

# Inference (NVIDIA Conformer on MLX — best WER)
python mlx_conformer.py --model nemo_models/ctc_extracted --audio test.mp3          # CTC (7.58% WER)
python mlx_conformer.py --model nemo_models/transducer_extracted --type transducer --audio test.mp3  # RNN-T (6.29% WER)

# Evaluation
python run_eval_whisper_streaming.py <args>  # PyTorch/HF models
python mlx_eval_whisper.py --model ./output_mlx_small/final --dataset-dir /path/to/cv/be  # MLX Whisper
python mlx_conformer.py --model nemo_models/ctc_extracted --eval --dataset-dir /path/to/cv/be  # Conformer
bash bash_runners/eval_cv11_test.sh   # Eval on CommonVoice11 test
bash bash_runners/eval_fleurs_test.sh # Eval on Fleurs test

# Live transcription (MLX)
python live_transcribe.py --model ./output_mlx_small/final

# Model conversion
python convert_hf_to_mlx.py --model ales/whisper-small-belarusian --output ./ales_mlx
```

No formal test suite exists. Validation is done via debug training runs and evaluation scripts.

## Architecture

**Entry points:**
- `run_speech_recognition_seq2seq_streaming.py` — PyTorch training script. HuggingFace Seq2Seq pipeline with streaming dataset support. Configurable via CLI args or JSON config.
- `mlx_finetune_whisper.py` — MLX training script for Apple Silicon. Reads CommonVoice data directly from local TSV/clips. ~5x faster than PyTorch on M-series Macs.
- `mlx_conformer.py` — MLX port of NVIDIA's Conformer-CTC and Conformer-Transducer models. Loads NeMo checkpoints directly. Best WER: 6.29% (Transducer), 7.58% (CTC) on CV24 test.
- `run_eval_whisper_streaming.py` — PyTorch evaluation script. Computes WER, optionally saves predictions to Excel, can push results to HuggingFace Hub.
- `mlx_eval_whisper.py` — MLX evaluation script for Whisper models. Computes WER on CommonVoice test set.
- `convert_hf_to_mlx.py` — Convert HuggingFace Whisper models to MLX whisper format.
- `live_transcribe.py` — Live microphone transcription using MLX Whisper. VAD-based: detects speech segments, transcribes complete utterances.

**Key modules:**
- `custom_trainer.py` — `Seq2SeqTrainerCustomLinearScheduler`: extends HuggingFace's Seq2SeqTrainer with a custom linear LR scheduler that supports `learning_rate_end` and proper resume-from-checkpoint behavior.
- `belarusian_text_normalizer.py` — `BelarusianTextNormalizer`: extends BasicTextNormalizer to preserve Belarusian-specific characters (apostrophe) during WER computation.

**Data pipeline (PyTorch):** Raw audio → 16kHz resampling → Whisper feature extractor → tokenizer → `DataCollatorSpeechSeq2SeqWithPadding` (dynamic padding). Both training and evaluation support streaming mode for memory efficiency (`streaming_train`/`streaming_eval` flags).

**Data pipeline (MLX):** Local MP3 clips → `load_audio` (16kHz) → `pad_or_trim` (30s) → `log_mel_spectrogram` → tokenizer (SOT + language + task + text + EOT). Reads directly from CommonVoice TSV files, no HuggingFace datasets dependency.

**Logging:** TensorBoard (migration to MLflow planned). Experiment tracking via HuggingFace Hub.

## Key Design Decisions

- **Streaming mode** is default for PyTorch training to handle large datasets without full download.
- **Checkpoint resume** is automatic — the PyTorch trainer detects existing checkpoints in output_dir. MLX script supports `--resume latest` or `--resume <path>` to continue from a checkpoint (saves model weights, optimizer state, and iteration number).
- **WER metric** is the primary evaluation metric; best model is saved based on it.
- All audio is resampled to 16kHz (Whisper requirement).
- Whisper's 30-second window limitation requires special handling for longer utterances (see README for details on chunking/striding and hallucination mitigation).
- **MLX float32 casting** — MLX whisper models from `mlx-community` store weights as float16 in safetensors. `load_model(dtype=mx.float32)` does not cast them, so explicit casting after load is required to prevent AdamW overflow → NaN loss.

## Package Management

Uses `uv` (not pip). Dependencies defined in `pyproject.toml`, locked in `uv.lock`. Python ≥3.13 required.

## Formatting

Code formatted with `black` and `isort`.
