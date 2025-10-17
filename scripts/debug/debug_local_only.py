#!/usr/bin/env python3
"""
Debug script for local-only ablation training failures.
Investigates why local tokens produce NaN during training.
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def check_embeddings_direct(parent_dir, num_samples=10):
    """Check embeddings by directly scanning directories (no manifest)."""
    print("Scanning directories directly...")

    parent_path = Path(parent_dir)
    subdirs = [d for d in parent_path.iterdir() if d.is_dir()]
    print(f"Found {len(subdirs)} subdirectories")

    issues = []
    stats = defaultdict(list)
    successful_loads = 0

    for idx, scene_dir in enumerate(subdirs[:num_samples]):
        locals_path = scene_dir / "locals_npz"

        if not locals_path.exists():
            issues.append(f"{scene_dir.name}: No locals_npz directory")
            continue

        caption_files = list(locals_path.glob("*.npz"))
        if not caption_files:
            issues.append(f"{scene_dir.name}: No .npz files found")
            continue

        print(f"\nScene: {scene_dir.name}")
        print(f"  Found {len(caption_files)} caption files")

        for cap_file in caption_files[:3]:
            try:
                data = np.load(cap_file)
                embeddings = data['embeddings']

                print(f"  Caption: {cap_file.name}")
                print(f"    Shape: {embeddings.shape}")
                print(f"    Min: {embeddings.min():.4f}, Max: {embeddings.max():.4f}")

                if np.isnan(embeddings).any():
                    issues.append(f"{scene_dir.name}/{cap_file.name}: Contains NaN!")
                    print(f"    ⚠️  WARNING: Contains NaN!")

                if embeddings.shape[0] > 1:
                    pairwise_diffs = np.abs(embeddings[:-1] - embeddings[1:]).max()
                    if pairwise_diffs < 1e-6:
                        issues.append(f"{scene_dir.name}/{cap_file.name}: Identical embeddings!")
                        print(f"    ⚠️  WARNING: All embeddings identical!")

                stats['min'].append(embeddings.min())
                stats['max'].append(embeddings.max())
                stats['mean'].append(embeddings.mean())
                stats['std'].append(embeddings.std())

                successful_loads += 1
            except Exception as e:
                issues.append(f"{scene_dir.name}/{cap_file.name}: Error - {e}")

    print(f"\nProcessed {successful_loads} caption files")

    if issues:
        print("\n⚠️  Issues found:")
        for issue in issues[:20]:
            print(f"  - {issue}")
        return False
    else:
        print("\n✅ No issues found")
        return True


def check_embeddings_integrity(parent_dir, num_samples=10):
    """Check if local embeddings contain NaN/Inf or are all identical."""
    print("="*80)
    print("1. CHECKING LOCAL EMBEDDINGS INTEGRITY")
    print("="*80)

    # Try multiple manifest filenames
    manifest_names = [
        "scene_packs_manifest_recovered.csv",
        "scene_packs_manifest.csv",
        "manifest.csv"
    ]

    manifest_path = None
    for name in manifest_names:
        path = Path(parent_dir) / name
        if path.exists():
            manifest_path = path
            break

    if manifest_path is None:
        print(f"ERROR: No manifest found. Tried:")
        for name in manifest_names:
            print(f"  - {Path(parent_dir) / name}")
        print("\nAttempting direct directory scan...")
        return check_embeddings_direct(parent_dir, num_samples)

    df = pd.read_csv(manifest_path)
    print(f"Loaded manifest: {manifest_path.name}")
    print(f"Columns: {list(df.columns)}")
    print(f"Total entries: {len(df)}")

    # Detect column name for scene pack directory
    scene_col = None
    for col in ['scene_pack', 'scene_id', 'clip_id', 'video_id', 'id']:
        if col in df.columns:
            scene_col = col
            break

    if scene_col is None:
        print(f"ERROR: Could not find scene identifier column")
        print(f"Available columns: {list(df.columns)}")
        return False

    print(f"Using '{scene_col}' as scene identifier\n")

    issues = []
    stats = defaultdict(list)

    for idx, row in df.head(num_samples).iterrows():
        scene_pack_dir = Path(parent_dir) / str(row[scene_col])
        locals_path = scene_pack_dir / "locals_npz"

        if not locals_path.exists():
            issues.append(f"Row {idx}: locals_npz directory not found at {locals_path}")
            continue

        # Check local embeddings for each caption
        caption_files = list(locals_path.glob("*.npz"))
        if not caption_files:
            issues.append(f"Row {idx}: No .npz files found in {locals_path}")
            continue

        print(f"\nSample {idx} - Scene: {row[scene_col]}")
        print(f"  Found {len(caption_files)} caption files")

        for cap_file in caption_files[:3]:  # Check first 3 captions
            try:
                data = np.load(cap_file)
                embeddings = data['embeddings']  # Shape: (num_tokens, embed_dim)

                print(f"  Caption: {cap_file.name}")
                print(f"    Shape: {embeddings.shape}")
                print(f"    Dtype: {embeddings.dtype}")
                print(f"    Min: {embeddings.min():.4f}, Max: {embeddings.max():.4f}")
                print(f"    Mean: {embeddings.mean():.4f}, Std: {embeddings.std():.4f}")

                # Check for NaN/Inf
                if np.isnan(embeddings).any():
                    issues.append(f"Row {idx}, {cap_file.name}: Contains NaN values!")
                    print(f"    ⚠️  WARNING: Contains NaN!")

                if np.isinf(embeddings).any():
                    issues.append(f"Row {idx}, {cap_file.name}: Contains Inf values!")
                    print(f"    ⚠️  WARNING: Contains Inf!")

                # Check if all embeddings are identical
                if embeddings.shape[0] > 1:
                    pairwise_diffs = np.abs(embeddings[:-1] - embeddings[1:]).max()
                    print(f"    Max pairwise diff: {pairwise_diffs:.6f}")
                    if pairwise_diffs < 1e-6:
                        issues.append(f"Row {idx}, {cap_file.name}: All embeddings identical!")
                        print(f"    ⚠️  WARNING: All embeddings are identical!")

                # Collect statistics
                stats['min'].append(embeddings.min())
                stats['max'].append(embeddings.max())
                stats['mean'].append(embeddings.mean())
                stats['std'].append(embeddings.std())
                stats['num_tokens'].append(embeddings.shape[0])
                stats['embed_dim'].append(embeddings.shape[1])

            except Exception as e:
                issues.append(f"Row {idx}, {cap_file.name}: Error loading - {e}")
                print(f"    ❌ ERROR: {e}")

    # Print summary statistics
    print("\n" + "="*80)
    print("EMBEDDING STATISTICS SUMMARY")
    print("="*80)
    for key, values in stats.items():
        if values:
            print(f"{key:15s}: mean={np.mean(values):.4f}, std={np.std(values):.4f}, "
                  f"min={np.min(values):.4f}, max={np.max(values):.4f}")

    # Print issues
    if issues:
        print("\n" + "="*80)
        print("⚠️  ISSUES FOUND:")
        print("="*80)
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("\n✅ No issues found in local embeddings")
        return True


def check_global_embeddings(parent_dir, num_samples=10):
    """Check global embeddings for comparison."""
    print("\n" + "="*80)
    print("2. CHECKING GLOBAL EMBEDDINGS (for comparison)")
    print("="*80)

    # Try multiple manifest filenames
    manifest_names = [
        "scene_packs_manifest_recovered.csv",
        "scene_packs_manifest.csv",
        "manifest.csv"
    ]

    manifest_path = None
    for name in manifest_names:
        path = Path(parent_dir) / name
        if path.exists():
            manifest_path = path
            break

    if manifest_path is None:
        print(f"WARNING: No manifest found, skipping global check")
        return True

    df = pd.read_csv(manifest_path)

    # Detect column name
    scene_col = None
    for col in ['scene_pack', 'scene_id', 'clip_id', 'video_id', 'id']:
        if col in df.columns:
            scene_col = col
            break

    if scene_col is None:
        print(f"WARNING: Could not find scene column")
        return True

    issues = []

    for idx, row in df.head(num_samples).iterrows():
        scene_pack_dir = Path(parent_dir) / str(row[scene_col])
        global_path = scene_pack_dir / "global_vec.npy"

        if not global_path.exists():
            issues.append(f"Row {idx}: global_vec.npy not found")
            continue

        try:
            global_emb = np.load(global_path)
            print(f"\nSample {idx} - Scene: {row[scene_col]}")
            print(f"  Shape: {global_emb.shape}")
            print(f"  Min: {global_emb.min():.4f}, Max: {global_emb.max():.4f}")
            print(f"  Mean: {global_emb.mean():.4f}, Std: {global_emb.std():.4f}")

            if np.isnan(global_emb).any():
                issues.append(f"Row {idx}: Global embedding contains NaN!")
            if np.isinf(global_emb).any():
                issues.append(f"Row {idx}: Global embedding contains Inf!")

        except Exception as e:
            issues.append(f"Row {idx}: Error loading global - {e}")
            print(f"  ❌ ERROR: {e}")

    if issues:
        print("\n⚠️  Issues in global embeddings:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("\n✅ Global embeddings look good")
        return True


def test_forward_pass(parent_dir):
    """Test a forward pass with local-only mode."""
    print("\n" + "="*80)
    print("3. TESTING FORWARD PASS WITH LOCAL-ONLY MODE")
    print("="*80)

    try:
        # Import the training script modules
        from scripts.llm_finetuning.next_utt_social_ooo_fix import (
            GemmaWithInjection, load_lm_and_tokenizer
        )

        # Load model (small version for testing)
        lm_name = "google/gemma-2-2b-it"
        print(f"Loading model: {lm_name}")
        lm, tokenizer = load_lm_and_tokenizer(lm_name)

        # Create model wrapper
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        model = GemmaWithInjection(lm, tokenizer)
        model = model.to(device)
        model.eval()

        # Create dummy inputs
        batch_size = 2
        seq_len = 20
        embed_dim = 768  # DINO embedding dimension

        input_ids = torch.randint(100, 1000, (batch_size, seq_len)).to(device)
        attention_mask = torch.ones(batch_size, seq_len).to(device)
        labels = input_ids.clone()

        # Create dummy visual inputs
        proj_global = torch.randn(batch_size, embed_dim).to(device) * 0.1

        # Local tokens: list of tensors per batch item
        proj_locals = [
            torch.randn(5, embed_dim).to(device) * 0.1  # 5 local tokens
            for _ in range(batch_size)
        ]

        print("\nTest 1: Both tokens mode")
        try:
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    proj_global=proj_global,
                    proj_locals=proj_locals,
                    inject_visuals=True,
                    ablation_mode="both"
                )
            loss = outputs.loss
            print(f"  Loss: {loss.item():.4f}")
            if torch.isnan(loss):
                print("  ❌ Loss is NaN!")
            else:
                print("  ✅ Forward pass successful")
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()

        print("\nTest 2: Local-only mode")
        try:
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    proj_global=proj_global,
                    proj_locals=proj_locals,
                    inject_visuals=True,
                    ablation_mode="local_only"
                )
            loss = outputs.loss
            print(f"  Loss: {loss.item():.4f}")
            if torch.isnan(loss):
                print("  ❌ Loss is NaN!")
            else:
                print("  ✅ Forward pass successful")
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()

        print("\nTest 3: Global-only mode")
        try:
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    proj_global=proj_global,
                    proj_locals=proj_locals,
                    inject_visuals=True,
                    ablation_mode="global_only"
                )
            loss = outputs.loss
            print(f"  Loss: {loss.item():.4f}")
            if torch.isnan(loss):
                print("  ❌ Loss is NaN!")
            else:
                print("  ✅ Forward pass successful")
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()

        print("\nTest 4: Gradient check with local-only")
        model.train()
        try:
            # Reset inputs with requires_grad
            proj_locals_grad = [
                torch.randn(5, embed_dim, requires_grad=True).to(device) * 0.1
                for _ in range(batch_size)
            ]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                proj_global=proj_global,
                proj_locals=proj_locals_grad,
                inject_visuals=True,
                ablation_mode="local_only"
            )
            loss = outputs.loss
            print(f"  Loss: {loss.item():.4f}")

            if torch.isnan(loss):
                print("  ❌ Loss is NaN before backward!")
                return False

            loss.backward()
            print("  ✅ Backward pass successful")

            # Check gradients
            for i, loc in enumerate(proj_locals_grad):
                if loc.grad is not None:
                    grad_norm = loc.grad.norm().item()
                    print(f"    Batch {i} gradient norm: {grad_norm:.6f}")
                    if torch.isnan(loc.grad).any():
                        print(f"    ❌ Gradient contains NaN!")
                    if grad_norm < 1e-8:
                        print(f"    ⚠️  Gradient is very small (vanishing)")
                else:
                    print(f"    ⚠️  No gradient for batch {i}")

        except Exception as e:
            print(f"  ❌ ERROR during gradient check: {e}")
            import traceback
            traceback.print_exc()
            return False

        return True

    except Exception as e:
        print(f"❌ ERROR: Failed to import or initialize model: {e}")
        import traceback
        traceback.print_exc()
        return False


def analyze_failed_metrics(metrics_path):
    """Analyze the failed training metrics."""
    print("\n" + "="*80)
    print("4. ANALYZING FAILED TRAINING METRICS")
    print("="*80)

    try:
        df = pd.read_csv(metrics_path)
        print(f"Loaded metrics with {len(df)} rows")

        # Check training rows
        train_rows = df[df['split'] == 'train']
        print(f"\nTraining rows: {len(train_rows)}")
        print(f"  Loss values: {train_rows['loss'].describe()}")
        print(f"  NaN count: {train_rows['loss'].isna().sum()}")
        print(f"  Inf count: {np.isinf(train_rows['loss']).sum()}")

        # Check first few rows
        print("\nFirst 10 training steps:")
        print(train_rows[['epoch', 'step', 'loss', 'ppl', 'lr_proj', 'lr_dino']].head(10))

        # Check if NaN appears from the start
        first_loss = train_rows.iloc[0]['loss']
        if np.isnan(first_loss):
            print("\n❌ CRITICAL: Loss is NaN from the very first step!")
            print("   This suggests an initialization or data loading issue.")
        else:
            print(f"\n✅ First loss is valid: {first_loss:.4f}")
            # Find when NaN first appears
            nan_mask = train_rows['loss'].isna() | np.isinf(train_rows['loss'])
            if nan_mask.any():
                first_nan_idx = nan_mask.idxmax()
                first_nan_row = train_rows.loc[first_nan_idx]
                print(f"⚠️  NaN first appeared at epoch {first_nan_row['epoch']}, "
                      f"step {first_nan_row['step']}")

        # Check if any embeddings are being used
        print(f"\nEmbedding usage statistics:")
        print(f"  g_has (global tokens present): {train_rows['g_has'].mean():.2f} avg")
        print(f"  l_has (local tokens present): {train_rows['l_has'].mean():.2f} avg")

        if train_rows['l_has'].mean() < 1:
            print("  ⚠️  WARNING: Very few or no local tokens are being used!")

    except Exception as e:
        print(f"❌ ERROR analyzing metrics: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("="*80)
    print("LOCAL-ONLY ABLATION DEBUGGING TOOL")
    print("="*80)
    print()

    # Get parent directory from command line or use default
    if len(sys.argv) > 1:
        parent_dir = sys.argv[1]
    else:
        # Default path from slurm scripts
        parent_dir = "/home/kgarci18/data_lisik3/kgarci18/ooo/train/social_tokens"
        print(f"No parent directory provided, using default: {parent_dir}")
        print("Usage: python debug_local_only.py <parent_dir>")
        print()

    # Check if directory exists
    if not Path(parent_dir).exists():
        print(f"❌ ERROR: Directory does not exist: {parent_dir}")
        print("Please provide a valid parent directory containing scene packs.")
        return

    print(f"Parent directory: {parent_dir}\n")

    # Run all checks
    results = {}

    results['embeddings'] = check_embeddings_integrity(parent_dir, num_samples=5)
    results['global'] = check_global_embeddings(parent_dir, num_samples=5)
    results['forward'] = test_forward_pass(parent_dir)

    # Analyze failed metrics if available
    metrics_path = Path("data/results/metrics_local_only.csv")
    if metrics_path.exists():
        analyze_failed_metrics(metrics_path)
    else:
        print(f"\n⚠️  Metrics file not found at {metrics_path}")

    # Final summary
    print("\n" + "="*80)
    print("DEBUGGING SUMMARY")
    print("="*80)

    all_passed = all(results.values())

    for check, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{check.upper():20s}: {status}")

    print()
    if all_passed:
        print("✅ All checks passed! The issue may be in training dynamics or data loading.")
        print("\nRecommendations:")
        print("  1. Check learning rate - may need to be different for local-only")
        print("  2. Check batch size - local-only may need larger batches for stability")
        print("  3. Add gradient clipping to prevent explosions")
        print("  4. Check if local tokens have sufficient diversity across frames")
    else:
        print("❌ Found issues that need to be fixed!")
        print("\nPriority actions:")
        if not results.get('embeddings', True):
            print("  1. Fix local embedding generation in preprocessing")
        if not results.get('forward', True):
            print("  2. Debug model forward pass with local-only mode")
        print("  3. Review training logs for more detailed error messages")

    print("="*80)


if __name__ == "__main__":
    main()
