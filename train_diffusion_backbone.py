#!/usr/bin/env python3
"""
ARC Diffusion Backbone Training with JSON Config Support

Trains the diffusion model with integrated size prediction head.

Usage:
    uv run python train_diffusion_backbone.py --config configs/test_config.json
    uv run python train_diffusion_backbone.py --config configs/test_config.json --no-wandb

Config file structure:
    {
        "model": {d_model, nhead, num_layers, max_size, etc.},
        "training": {batch_size, learning_rate, optimizer_steps, etc.},
        "data": {data_dir, datasets, etc.},
        "auxiliary_loss": {include_size_head, size_head_hidden_dim, etc.},
        "output": {output_dir, use_wandb, etc.}
    }
"""
import json
import argparse
import sys
from pathlib import Path

from src.training import train_arc_diffusion


def load_and_flatten_config(config_path: str) -> dict:
    """
    Load configuration from JSON file and flatten nested structure.

    The training function expects a flat dictionary, so we flatten the nested
    config structure while preserving the auxiliary_loss section.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Flatten nested config sections
    flat_config = {}

    # Copy top-level fields (model_version, tag, pretrained_model_path, etc.)
    for key in ['model_version', 'tag', 'pretrained_model_path']:
        if key in config:
            flat_config[key] = config[key]

    # Flatten model, training, data, and output sections
    for section in ['model', 'training', 'data', 'output']:
        if section in config:
            flat_config.update(config[section])

    # Keep auxiliary_loss and lora as nested dicts (expected by training.py)
    if 'auxiliary_loss' in config:
        flat_config['auxiliary_loss'] = config['auxiliary_loss']
    if 'lora' in config:
        flat_config['lora'] = config['lora']

    return flat_config


def main():
    parser = argparse.ArgumentParser(
        description="Train ARC Diffusion Model with JSON Config",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to JSON config file (see configs/ for examples)"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging (overrides config)"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run profiling mode: execute 20 steps with PyTorch profiler and output timing breakdown"
    )

    args = parser.parse_args()

    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)

    # Load and flatten config
    try:
        config = load_and_flatten_config(args.config)
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        sys.exit(1)

    # Override wandb setting if requested
    if args.no_wandb:
        config['use_wandb'] = False

    # Add profiling flag to config
    config['profile_mode'] = args.profile

    print(f"✨ Training ARC Diffusion Model")
    print(f"📁 Config: {args.config}")
    print(f"📊 Model: {config['d_model']}d, {config['num_layers']} layers")
    print(f"🎯 Training: {config['optimizer_steps']} steps, batch_size={config['batch_size']}")
    print(f"💾 Output: {config['output_dir']}")
    if args.profile:
        print(f"🔬 Profiling mode: Will run 20 steps with PyTorch profiler")

    # Train model
    try:
        model = train_arc_diffusion(config=config)
        print("✅ Training completed successfully!")
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()