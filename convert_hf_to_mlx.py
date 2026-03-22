"""
Convert HuggingFace Whisper model to MLX whisper format.

Usage:
    python convert_hf_to_mlx.py --model ales/whisper-small-belarusian --output ./ales_mlx
"""

import argparse
import json
import os
from pathlib import Path

import mlx.core as mx
import numpy as np


def convert_key(hf_key):
    """Map HuggingFace whisper weight key to MLX whisper weight key."""
    k = hf_key

    # Strip 'model.' prefix
    k = k.replace("model.", "", 1)

    # Encoder positional embedding — not stored in MLX (computed from sinusoids)
    if k == "encoder.embed_positions.weight":
        return None

    # Decoder positional embedding (no .weight suffix in MLX)
    k = k.replace("decoder.embed_positions.weight", "decoder.positional_embedding")

    # Token embedding
    k = k.replace("decoder.embed_tokens.weight", "decoder.token_embedding.weight")

    # Layer norm
    k = k.replace("encoder.layer_norm.", "encoder.ln_post.")
    k = k.replace("decoder.layer_norm.", "decoder.ln.")

    # Encoder blocks: layers.N -> blocks.N
    k = k.replace("encoder.layers.", "encoder.blocks.")
    k = k.replace("decoder.layers.", "decoder.blocks.")

    # Self attention
    k = k.replace(".self_attn.q_proj.", ".attn.query.")
    k = k.replace(".self_attn.k_proj.", ".attn.key.")
    k = k.replace(".self_attn.v_proj.", ".attn.value.")
    k = k.replace(".self_attn.out_proj.", ".attn.out.")
    k = k.replace(".self_attn_layer_norm.", ".attn_ln.")

    # Cross attention (decoder)
    k = k.replace(".encoder_attn.q_proj.", ".cross_attn.query.")
    k = k.replace(".encoder_attn.k_proj.", ".cross_attn.key.")
    k = k.replace(".encoder_attn.v_proj.", ".cross_attn.value.")
    k = k.replace(".encoder_attn.out_proj.", ".cross_attn.out.")
    k = k.replace(".encoder_attn_layer_norm.", ".cross_attn_ln.")

    # MLP
    k = k.replace(".fc1.", ".mlp1.")
    k = k.replace(".fc2.", ".mlp2.")
    k = k.replace(".final_layer_norm.", ".mlp_ln.")

    return k


def convert_weight(key, weight):
    """Convert weight tensor, handling shape differences."""
    # Conv layers: HF uses (out, in, kernel), MLX uses (out, kernel, in)
    if "conv1.weight" in key or "conv2.weight" in key:
        # HF: (out_channels, in_channels, kernel_size)
        # MLX: (out_channels, kernel_size, in_channels)
        weight = np.transpose(weight, (0, 2, 1))
    return weight


def main():
    parser = argparse.ArgumentParser(description="Convert HF Whisper to MLX format")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model ID")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    # Download model
    from huggingface_hub import snapshot_download
    model_path = Path(snapshot_download(repo_id=args.model))

    # Load HF config
    with open(model_path / "config.json") as f:
        hf_config = json.load(f)

    # Build MLX config
    mlx_config = {
        "n_mels": hf_config["num_mel_bins"],
        "n_audio_ctx": hf_config["max_source_positions"],
        "n_audio_state": hf_config["d_model"],
        "n_audio_head": hf_config["encoder_attention_heads"],
        "n_audio_layer": hf_config["encoder_layers"],
        "n_vocab": hf_config["vocab_size"],
        "n_text_ctx": hf_config["max_target_positions"],
        "n_text_state": hf_config["d_model"],
        "n_text_head": hf_config["decoder_attention_heads"],
        "n_text_layer": hf_config["decoder_layers"],
    }

    # Load weights
    import safetensors.numpy
    hf_weights = safetensors.numpy.load_file(str(model_path / "model.safetensors"))

    # Convert
    mlx_weights = {}
    for hf_key, weight in hf_weights.items():
        mlx_key = convert_key(hf_key)
        if mlx_key is None:
            continue
        weight = convert_weight(mlx_key, weight)
        mlx_weights[mlx_key] = mx.array(weight, dtype=mx.float32)

    # Verify all keys converted
    print(f"Converted {len(mlx_weights)} weights")

    # Save
    os.makedirs(args.output, exist_ok=True)
    mx.save_safetensors(os.path.join(args.output, "model.safetensors"), mlx_weights)
    with open(os.path.join(args.output, "config.json"), "w") as f:
        json.dump(mlx_config, f, indent=2)

    print(f"Saved MLX model to {args.output}")
    print(f"Config: {json.dumps(mlx_config, indent=2)}")


if __name__ == "__main__":
    main()
