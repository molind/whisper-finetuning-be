# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fine-tuning OpenAI's Whisper model for Belarusian speech recognition using HuggingFace Transformers. Published models: `ales/whisper-small-belarusian`, `ales/whisper-base-belarusian`. Dataset: CommonVoice 11.0 (Belarusian).

## Commands

```bash
# Setup
make install-uv          # Install uv package manager
make venv-create          # Create Python 3.13+ venv
make venv-install         # Install dependencies (uv sync)
make server-install-system-deps  # Install ffmpeg, git-lfs, tmux

# Training
python run_speech_recognition_seq2seq_streaming.py <args>  # or use bash runners:
bash bash_runners/run_tiny_debug.sh   # Quick debug run (tiny model, 500 steps)
bash bash_runners/run_small.sh        # Fine-tune small model
bash bash_runners/run_base.sh         # Fine-tune base model

# Evaluation
python run_eval_whisper_streaming.py <args>  # or use bash runners:
bash bash_runners/eval_cv11_test.sh   # Eval on CommonVoice11 test
bash bash_runners/eval_fleurs_test.sh # Eval on Fleurs test
```

No formal test suite exists. Validation is done via debug training runs and evaluation scripts.

## Architecture

**Entry points:**
- `run_speech_recognition_seq2seq_streaming.py` — Main training script. HuggingFace Seq2Seq pipeline with streaming dataset support. Configurable via CLI args or JSON config.
- `run_eval_whisper_streaming.py` — Evaluation script. Computes WER, optionally saves predictions to Excel, can push results to HuggingFace Hub.

**Key modules:**
- `custom_trainer.py` — `Seq2SeqTrainerCustomLinearScheduler`: extends HuggingFace's Seq2SeqTrainer with a custom linear LR scheduler that supports `learning_rate_end` and proper resume-from-checkpoint behavior.
- `belarusian_text_normalizer.py` — `BelarusianTextNormalizer`: extends BasicTextNormalizer to preserve Belarusian-specific characters (apostrophe) during WER computation.

**Data pipeline:** Raw audio → 16kHz resampling → Whisper feature extractor → tokenizer → `DataCollatorSpeechSeq2SeqWithPadding` (dynamic padding). Both training and evaluation support streaming mode for memory efficiency (`streaming_train`/`streaming_eval` flags).

**Logging:** TensorBoard (migration to MLflow planned). Experiment tracking via HuggingFace Hub.

## Key Design Decisions

- **Streaming mode** is default for training to handle large datasets without full download.
- **Checkpoint resume** is automatic — the trainer detects existing checkpoints in output_dir.
- **WER metric** is the primary evaluation metric; best model is saved based on it.
- All audio is resampled to 16kHz (Whisper requirement).
- Whisper's 30-second window limitation requires special handling for longer utterances (see README for details on chunking/striding and hallucination mitigation).

## Package Management

Uses `uv` (not pip). Dependencies defined in `pyproject.toml`, locked in `uv.lock`. Python ≥3.13 required.

## Formatting

Code formatted with `black` and `isort`.
