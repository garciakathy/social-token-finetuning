#!/usr/bin/env python3
"""Check if DINO checkpoint contains trained weights or random initialization."""
import torch
import sys
from pathlib import Path
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python check_dino_checkpoint.py <checkpoint_path>")
    sys.exit(1)

ckpt_path = sys.argv[1]

if not Path(ckpt_path).exists():
    print(f"ERROR: Checkpoint not found: {ckpt_path}")
    sys.exit(1)

print("="*80)
print("DINO CHECKPOINT ANALYSIS")
print("="*80)
print(f"Checkpoint: {ckpt_path}")
print(f"Size: {Path(ckpt_path).stat().st_size / 1024 / 1024:.2f} MB")
print()

try:
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    print("Checkpoint structure:")
    if isinstance(checkpoint, dict):
        print(f"  Type: Dictionary with {len(checkpoint)} keys")
        print(f"  Keys: {list(checkpoint.keys())}")

        # Check for state dict
        state_dict = None
        if "student_backbone" in checkpoint:
            print("\n  Found 'student_backbone' key (DINO teacher-student training)")
            state_dict = checkpoint["student_backbone"]
        elif "model" in checkpoint:
            print("\n  Found 'model' key")
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            print("\n  Found 'state_dict' key")
            state_dict = checkpoint["state_dict"]
        else:
            # Assume entire checkpoint is state dict
            print("\n  Assuming entire checkpoint is state_dict")
            state_dict = checkpoint

        if state_dict is not None:
            print(f"\nState dict contains {len(state_dict)} parameters")

            # Sample some weights
            print("\nSampling parameter statistics:")
            param_stats = []

            for name, param in list(state_dict.items())[:10]:
                if isinstance(param, torch.Tensor):
                    param_np = param.cpu().numpy()
                    stats = {
                        'name': name,
                        'shape': param.shape,
                        'mean': param_np.mean(),
                        'std': param_np.std(),
                        'min': param_np.min(),
                        'max': param_np.max(),
                    }
                    param_stats.append(stats)
                    print(f"\n  {name}:")
                    print(f"    Shape: {stats['shape']}")
                    print(f"    Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")
                    print(f"    Min: {stats['min']:.6f}, Max: {stats['max']:.6f}")

            # Check if weights look random
            print("\n" + "="*80)
            print("ANALYSIS:")
            print("="*80)

            if param_stats:
                avg_mean = np.mean([s['mean'] for s in param_stats])
                avg_std = np.mean([s['std'] for s in param_stats])

                print(f"Average mean across parameters: {avg_mean:.6f}")
                print(f"Average std across parameters: {avg_std:.6f}")

                # Check for random initialization patterns
                issues = []

                # Check if all means are near zero
                if abs(avg_mean) < 0.01:
                    issues.append("All parameter means near zero (suspicious)")

                # Check for uniform std (sign of random init)
                stds = [s['std'] for s in param_stats]
                std_variance = np.std(stds)
                if std_variance < 0.001:
                    issues.append(f"Very uniform std across layers (variance={std_variance:.6f})")

                # Check training metadata
                if "epoch" in checkpoint:
                    epoch = checkpoint["epoch"]
                    print(f"\nTraining metadata found:")
                    print(f"  Epoch: {epoch}")
                    if epoch == 0:
                        issues.append("Checkpoint saved at epoch 0 (untrained)")
                else:
                    issues.append("No training metadata (epoch, step, etc.)")

                if "optimizer" in checkpoint:
                    print(f"  Optimizer state: Present")
                else:
                    print(f"  Optimizer state: Missing")

                if issues:
                    print("\n⚠️  POTENTIAL ISSUES:")
                    for issue in issues:
                        print(f"  - {issue}")
                    print("\n❌ CONCLUSION: Checkpoint may be UNTRAINED or RANDOM INITIALIZATION")
                    print("\nThis would explain why all embeddings have identical statistics.")
                else:
                    print("\n✅ Checkpoint appears to contain trained weights")

        # Check for additional info
        if "args" in checkpoint:
            print(f"\nTraining args found: {checkpoint['args']}")

        if "loss" in checkpoint:
            print(f"Validation loss: {checkpoint['loss']}")

    else:
        print(f"  Type: {type(checkpoint)}")
        print(f"  Checkpoint is not a dictionary - unusual format")

except Exception as e:
    print(f"\n❌ ERROR loading checkpoint: {e}")
    import traceback
    traceback.print_exc()

print("="*80)
