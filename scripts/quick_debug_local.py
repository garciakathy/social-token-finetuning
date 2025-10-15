#!/usr/bin/env python3
"""
Quick debug script for local embeddings (no model loading required).
Can run on local machine without GPU.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import sys


def analyze_local_embeddings(parent_dir, num_samples=20):
    """Deep analysis of local token embeddings."""
    print("="*80)
    print("LOCAL EMBEDDINGS ANALYSIS")
    print("="*80)

    manifest_path = Path(parent_dir) / "scene_packs_manifest_recovered.csv"
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found at {manifest_path}")
        return

    df = pd.read_csv(manifest_path)
    print(f"Total scenes in manifest: {len(df)}")

    all_stats = {
        'min_vals': [],
        'max_vals': [],
        'mean_vals': [],
        'std_vals': [],
        'l2_norms': [],
        'num_tokens_per_caption': [],
        'embed_dims': [],
        'pairwise_sims': [],
        'identical_count': 0,
        'nan_count': 0,
        'inf_count': 0,
        'zero_count': 0,
    }

    successful_loads = 0
    failed_loads = []

    for idx, row in df.head(num_samples).iterrows():
        scene_pack_dir = Path(parent_dir) / row['scene_pack']
        locals_path = scene_pack_dir / "locals_npz"

        if not locals_path.exists():
            failed_loads.append((idx, "locals_npz directory not found"))
            continue

        caption_files = sorted(locals_path.glob("*.npz"))
        if not caption_files:
            failed_loads.append((idx, "No .npz files in locals_npz"))
            continue

        print(f"\nScene {idx}: {row['scene_pack']}")
        print(f"  Captions: {len(caption_files)}")

        for cap_idx, cap_file in enumerate(caption_files):
            try:
                data = np.load(cap_file)
                embeddings = data['embeddings']

                # Basic stats
                all_stats['min_vals'].append(embeddings.min())
                all_stats['max_vals'].append(embeddings.max())
                all_stats['mean_vals'].append(embeddings.mean())
                all_stats['std_vals'].append(embeddings.std())
                all_stats['num_tokens_per_caption'].append(embeddings.shape[0])
                all_stats['embed_dims'].append(embeddings.shape[1])

                # L2 norms per token
                l2_norms = np.linalg.norm(embeddings, axis=1)
                all_stats['l2_norms'].extend(l2_norms.tolist())

                # Check for issues
                if np.isnan(embeddings).any():
                    all_stats['nan_count'] += 1
                    print(f"    ⚠️  Caption {cap_idx}: Contains NaN!")

                if np.isinf(embeddings).any():
                    all_stats['inf_count'] += 1
                    print(f"    ⚠️  Caption {cap_idx}: Contains Inf!")

                if np.allclose(embeddings, 0):
                    all_stats['zero_count'] += 1
                    print(f"    ⚠️  Caption {cap_idx}: All zeros!")

                # Check if tokens are identical
                if embeddings.shape[0] > 1:
                    pairwise_diff = np.abs(embeddings[:-1] - embeddings[1:]).max()
                    if pairwise_diff < 1e-6:
                        all_stats['identical_count'] += 1
                        print(f"    ⚠️  Caption {cap_idx}: All {embeddings.shape[0]} tokens identical!")

                    # Cosine similarity between consecutive tokens
                    for i in range(len(embeddings) - 1):
                        v1 = embeddings[i]
                        v2 = embeddings[i + 1]
                        cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                        all_stats['pairwise_sims'].append(cos_sim)

                if cap_idx == 0:  # Show details for first caption
                    print(f"    Shape: {embeddings.shape}, Mean: {embeddings.mean():.4f}, "
                          f"Std: {embeddings.std():.4f}")

                successful_loads += 1

            except Exception as e:
                failed_loads.append((idx, f"Error loading {cap_file.name}: {e}"))

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    print(f"Successfully loaded: {successful_loads} caption files")
    print(f"Failed loads: {len(failed_loads)}")

    if failed_loads:
        print("\nFailure details:")
        for idx, reason in failed_loads[:10]:
            print(f"  Scene {idx}: {reason}")

    print(f"\nData quality issues:")
    print(f"  Captions with NaN: {all_stats['nan_count']}")
    print(f"  Captions with Inf: {all_stats['inf_count']}")
    print(f"  Captions with all zeros: {all_stats['zero_count']}")
    print(f"  Captions with identical tokens: {all_stats['identical_count']}")

    if all_stats['mean_vals']:
        print(f"\nValue ranges:")
        print(f"  Min value across all: {min(all_stats['min_vals']):.6f}")
        print(f"  Max value across all: {max(all_stats['max_vals']):.6f}")
        print(f"  Mean of means: {np.mean(all_stats['mean_vals']):.6f}")
        print(f"  Mean of stds: {np.mean(all_stats['std_vals']):.6f}")

    if all_stats['l2_norms']:
        print(f"\nL2 norms:")
        print(f"  Mean: {np.mean(all_stats['l2_norms']):.4f}")
        print(f"  Std: {np.std(all_stats['l2_norms']):.4f}")
        print(f"  Min: {np.min(all_stats['l2_norms']):.4f}")
        print(f"  Max: {np.max(all_stats['l2_norms']):.4f}")

    if all_stats['num_tokens_per_caption']:
        print(f"\nTokens per caption:")
        print(f"  Mean: {np.mean(all_stats['num_tokens_per_caption']):.2f}")
        print(f"  Std: {np.std(all_stats['num_tokens_per_caption']):.2f}")
        print(f"  Min: {np.min(all_stats['num_tokens_per_caption'])}")
        print(f"  Max: {np.max(all_stats['num_tokens_per_caption'])}")

    if all_stats['pairwise_sims']:
        print(f"\nPairwise cosine similarities (consecutive tokens):")
        print(f"  Mean: {np.mean(all_stats['pairwise_sims']):.4f}")
        print(f"  Std: {np.std(all_stats['pairwise_sims']):.4f}")
        print(f"  Min: {np.min(all_stats['pairwise_sims']):.4f}")
        print(f"  Max: {np.max(all_stats['pairwise_sims']):.4f}")

        # High similarity indicates tokens are too similar
        high_sim_ratio = np.mean(np.array(all_stats['pairwise_sims']) > 0.99)
        print(f"  Ratio with similarity > 0.99: {high_sim_ratio:.2%}")

        if high_sim_ratio > 0.5:
            print(f"    ⚠️  WARNING: {high_sim_ratio:.0%} of consecutive tokens are nearly identical!")
            print(f"               This suggests insufficient frame diversity.")

    # Plot distributions
    plot_distributions(all_stats)

    return all_stats


def plot_distributions(stats):
    """Plot distributions of embedding statistics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # L2 norms
    if stats['l2_norms']:
        axes[0, 0].hist(stats['l2_norms'], bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('L2 Norm')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Distribution of Token L2 Norms')
        axes[0, 0].axvline(np.mean(stats['l2_norms']), color='r', linestyle='--',
                           label=f'Mean: {np.mean(stats["l2_norms"]):.2f}')
        axes[0, 0].legend()

    # Pairwise similarities
    if stats['pairwise_sims']:
        axes[0, 1].hist(stats['pairwise_sims'], bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Cosine Similarity')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Pairwise Cosine Similarities (Consecutive Tokens)')
        axes[0, 1].axvline(np.mean(stats['pairwise_sims']), color='r', linestyle='--',
                           label=f'Mean: {np.mean(stats["pairwise_sims"]):.2f}')
        axes[0, 1].legend()

    # Tokens per caption
    if stats['num_tokens_per_caption']:
        axes[1, 0].hist(stats['num_tokens_per_caption'], bins=20, edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Number of Tokens')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Tokens per Caption')
        axes[1, 0].axvline(np.mean(stats['num_tokens_per_caption']), color='r', linestyle='--',
                           label=f'Mean: {np.mean(stats["num_tokens_per_caption"]):.1f}')
        axes[1, 0].legend()

    # Standard deviations
    if stats['std_vals']:
        axes[1, 1].hist(stats['std_vals'], bins=50, edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('Standard Deviation')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Embedding Standard Deviations per Caption')
        axes[1, 1].axvline(np.mean(stats['std_vals']), color='r', linestyle='--',
                           label=f'Mean: {np.mean(stats["std_vals"]):.3f}')
        axes[1, 1].legend()

    plt.tight_layout()
    output_path = Path("data/local_embeddings_debug.png")
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Plots saved to: {output_path}")


def compare_with_global(parent_dir, num_samples=20):
    """Compare local embeddings with global embeddings."""
    print("\n" + "="*80)
    print("COMPARISON WITH GLOBAL EMBEDDINGS")
    print("="*80)

    manifest_path = Path(parent_dir) / "scene_packs_manifest_recovered.csv"
    df = pd.read_csv(manifest_path)

    local_norms = []
    global_norms = []

    for idx, row in df.head(num_samples).iterrows():
        scene_pack_dir = Path(parent_dir) / row['scene_pack']

        # Load global
        global_path = scene_pack_dir / "global_vec.npy"
        if global_path.exists():
            global_emb = np.load(global_path)
            global_norms.append(np.linalg.norm(global_emb))

        # Load locals
        locals_path = scene_pack_dir / "locals_npz"
        if locals_path.exists():
            for cap_file in locals_path.glob("*.npz"):
                try:
                    data = np.load(cap_file)
                    embeddings = data['embeddings']
                    local_norms.extend(np.linalg.norm(embeddings, axis=1).tolist())
                except:
                    pass

    if local_norms and global_norms:
        print(f"\nGlobal embeddings (per scene):")
        print(f"  Mean L2 norm: {np.mean(global_norms):.4f}")
        print(f"  Std: {np.std(global_norms):.4f}")

        print(f"\nLocal embeddings (per token):")
        print(f"  Mean L2 norm: {np.mean(local_norms):.4f}")
        print(f"  Std: {np.std(local_norms):.4f}")

        ratio = np.mean(local_norms) / np.mean(global_norms)
        print(f"\nLocal/Global norm ratio: {ratio:.4f}")

        if ratio < 0.1 or ratio > 10:
            print(f"  ⚠️  WARNING: Large magnitude difference between local and global!")
            print(f"             This may cause training instability.")


def main():
    print("="*80)
    print("QUICK LOCAL EMBEDDINGS DEBUG")
    print("="*80)
    print()

    if len(sys.argv) > 1:
        parent_dir = sys.argv[1]
    else:
        # Try to find data directory
        possible_paths = [
            "/home/kgarci18/data_lisik3/kgarci18/ooo/train/social_tokens",
            "/Users/aarondaly/kg/code/social-token-finetuning/data/social_tokens",
            "data/social_tokens"
        ]

        parent_dir = None
        for path in possible_paths:
            if Path(path).exists():
                parent_dir = path
                break

        if parent_dir is None:
            print("ERROR: Could not find data directory.")
            print("Usage: python quick_debug_local.py <parent_dir>")
            print("\nSearched:")
            for path in possible_paths:
                print(f"  - {path}")
            return

    print(f"Parent directory: {parent_dir}\n")

    if not Path(parent_dir).exists():
        print(f"ERROR: Directory does not exist: {parent_dir}")
        return

    # Run analysis
    stats = analyze_local_embeddings(parent_dir, num_samples=30)
    compare_with_global(parent_dir, num_samples=30)

    # Diagnose issues
    print("\n" + "="*80)
    print("DIAGNOSIS")
    print("="*80)

    if stats:
        issues = []

        if stats['nan_count'] > 0:
            issues.append(f"NaN values detected in {stats['nan_count']} captions")

        if stats['identical_count'] > stats['identical_count'] * 0.3:
            issues.append(f"High ratio of identical tokens ({stats['identical_count']} captions)")

        if stats['pairwise_sims'] and np.mean(stats['pairwise_sims']) > 0.95:
            issues.append(f"Very high token similarity (mean={np.mean(stats['pairwise_sims']):.3f})")

        if stats['std_vals'] and np.mean(stats['std_vals']) < 0.01:
            issues.append(f"Very low embedding variance (mean std={np.mean(stats['std_vals']):.4f})")

        if issues:
            print("⚠️  ISSUES FOUND:")
            for issue in issues:
                print(f"  - {issue}")

            print("\nLIKELY ROOT CAUSE:")
            if stats['identical_count'] > 5 or (stats['pairwise_sims'] and np.mean(stats['pairwise_sims']) > 0.95):
                print("  → Tokens are too similar due to:")
                print("    1. Insufficient frame diversity (same frame used for multiple tokens)")
                print("    2. Static/unchanging video content")
                print("    3. Small window size in uniform distribution (pad_frames too small)")

            print("\nRECOMMENDED FIXES:")
            print("  1. Increase frame diversity in preprocessing")
            print("  2. Use different frames for each token instead of averaging nearby frames")
            print("  3. Add noise augmentation to local embeddings during training")
            print("  4. Increase --max-locals to use more diverse tokens")
        else:
            print("✅ No obvious data issues detected")
            print("\nIf training still fails, the issue is likely in:")
            print("  1. Model architecture (projector initialization)")
            print("  2. Training dynamics (learning rate, gradient scaling)")
            print("  3. Batch size (may need larger batches for local-only)")

    print("="*80)


if __name__ == "__main__":
    main()
