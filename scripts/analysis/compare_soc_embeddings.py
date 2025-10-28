#!/usr/bin/env python3
"""
Sanity check: Compare embeddings with and without social token injection.

Usage:
    python compare_soc_embeddings.py \
        --baseline-dir ~/path/to/inject_none_features \
        --soc-dir ~/path/to/inject_full_features \
        --output-dir ~/path/to/output
"""

import argparse
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
import pandas as pd


def load_features(feature_dir, pattern="best.npz"):
    """Load all feature files matching pattern in directory."""
    feature_dir = Path(feature_dir)
    files = list(feature_dir.glob(f"*{pattern}"))

    if not files:
        raise FileNotFoundError(f"No files matching *{pattern} in {feature_dir}")

    # Load the first (or best) matching file
    npz_file = files[0]
    print(f"Loading features from: {npz_file}")

    data = np.load(npz_file, allow_pickle=True)
    X = data['X']  # [N_items, feature_dim]
    uids = data['uids'] if 'uids' in data else None

    print(f"  Shape: {X.shape}")
    return X, uids, npz_file.name


def compute_statistics(baseline, soc):
    """Compute various statistics comparing baseline and SOC embeddings."""
    assert baseline.shape == soc.shape, f"Shape mismatch: {baseline.shape} vs {soc.shape}"

    stats = {}

    # 1. Absolute difference
    delta = soc - baseline
    stats['mean_abs_delta'] = np.mean(np.abs(delta))
    stats['std_abs_delta'] = np.std(np.abs(delta))
    stats['max_abs_delta'] = np.max(np.abs(delta))
    stats['min_abs_delta'] = np.min(np.abs(delta))

    # 2. Relative change
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_change = np.abs(delta) / (np.abs(baseline) + 1e-8)
    stats['mean_rel_change'] = np.nanmean(rel_change)
    stats['median_rel_change'] = np.nanmedian(rel_change)

    # 3. Cosine similarity per item
    cosine_sims = []
    for i in range(baseline.shape[0]):
        sim = 1 - cosine(baseline[i], soc[i])
        cosine_sims.append(sim)
    stats['mean_cosine_sim'] = np.mean(cosine_sims)
    stats['std_cosine_sim'] = np.std(cosine_sims)

    # 4. Correlation between baseline and SOC features
    flat_baseline = baseline.flatten()
    flat_soc = soc.flatten()
    rho, p = spearmanr(flat_baseline, flat_soc)
    stats['spearman_rho'] = rho
    stats['spearman_p'] = p

    # 5. Frobenius norm of difference
    stats['frobenius_norm'] = np.linalg.norm(delta, 'fro')

    # 6. Per-dimension statistics
    stats['dims_changed_gt_1pct'] = np.sum(np.abs(delta).mean(axis=0) > 0.01)
    stats['dims_changed_gt_5pct'] = np.sum(np.abs(delta).mean(axis=0) > 0.05)
    stats['dims_changed_gt_10pct'] = np.sum(np.abs(delta).mean(axis=0) > 0.10)

    return stats, delta, cosine_sims


def plot_comparison(baseline, soc, delta, cosine_sims, output_dir):
    """Generate visualization plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Delta histogram
    ax = axes[0, 0]
    ax.hist(delta.flatten(), bins=100, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Delta (SOC - Baseline)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Embedding Deltas')
    ax.axvline(0, color='red', linestyle='--', linewidth=1)

    # 2. Absolute delta histogram
    ax = axes[0, 1]
    ax.hist(np.abs(delta).flatten(), bins=100, alpha=0.7, edgecolor='black', color='orange')
    ax.set_xlabel('|Delta|')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Absolute Deltas')

    # 3. Cosine similarity distribution
    ax = axes[0, 2]
    ax.hist(cosine_sims, bins=50, alpha=0.7, edgecolor='black', color='green')
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Frequency')
    ax.set_title('Per-Item Cosine Similarity\n(Baseline vs SOC)')

    # 4. Mean absolute delta per dimension
    ax = axes[1, 0]
    mean_abs_delta_per_dim = np.abs(delta).mean(axis=0)
    ax.plot(mean_abs_delta_per_dim, linewidth=0.5)
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Mean |Delta|')
    ax.set_title('Mean Absolute Delta per Dimension')

    # 5. Baseline vs SOC scatter (sample)
    ax = axes[1, 1]
    sample_size = min(10000, baseline.size)
    indices = np.random.choice(baseline.size, size=sample_size, replace=False)
    ax.scatter(baseline.flatten()[indices], soc.flatten()[indices],
               alpha=0.1, s=1)
    ax.set_xlabel('Baseline Embedding Values')
    ax.set_ylabel('SOC Embedding Values')
    ax.set_title('Baseline vs SOC (sampled)')
    # Add diagonal line
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'r--', alpha=0.5, linewidth=1)

    # 6. Per-item L2 norm of delta
    ax = axes[1, 2]
    l2_norms = np.linalg.norm(delta, axis=1)
    ax.hist(l2_norms, bins=50, alpha=0.7, edgecolor='black', color='purple')
    ax.set_xlabel('L2 Norm of Delta')
    ax.set_ylabel('Frequency')
    ax.set_title('Per-Item L2 Norm of Delta')

    plt.tight_layout()
    plot_path = output_dir / "embedding_comparison.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot to: {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare baseline vs SOC embeddings")
    parser.add_argument("--baseline-dir", required=True,
                        help="Directory with features from --inject none")
    parser.add_argument("--soc-dir", required=True,
                        help="Directory with features from --inject full/global/local")
    parser.add_argument("--output-dir", default="./embedding_comparison",
                        help="Output directory for results")
    parser.add_argument("--pattern", default="best.npz",
                        help="File pattern to match (default: best.npz)")
    args = parser.parse_args()

    # Load features
    print("\n=== Loading Features ===")
    baseline, baseline_uids, baseline_file = load_features(args.baseline_dir, args.pattern)
    soc, soc_uids, soc_file = load_features(args.soc_dir, args.pattern)

    # Verify alignment
    if baseline_uids is not None and soc_uids is not None:
        if not np.array_equal(baseline_uids, soc_uids):
            print("WARNING: UIDs don't match! Results may be incorrect.")

    # Compute statistics
    print("\n=== Computing Statistics ===")
    stats, delta, cosine_sims = compute_statistics(baseline, soc)

    # Print results
    print("\n=== Comparison Statistics ===")
    print(f"Baseline file: {baseline_file}")
    print(f"SOC file:      {soc_file}")
    print(f"Shape:         {baseline.shape}")
    print()
    for key, value in stats.items():
        if isinstance(value, float):
            if 'p' in key:
                print(f"  {key:25s}: {value:.2e}")
            else:
                print(f"  {key:25s}: {value:.6f}")
        else:
            print(f"  {key:25s}: {value}")

    # Save statistics
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_df = pd.DataFrame([stats])
    stats_df['baseline_file'] = baseline_file
    stats_df['soc_file'] = soc_file
    stats_df['n_items'] = baseline.shape[0]
    stats_df['feature_dim'] = baseline.shape[1]

    stats_path = output_dir / "comparison_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"\nSaved statistics to: {stats_path}")

    # Generate plots
    print("\n=== Generating Plots ===")
    plot_comparison(baseline, soc, delta, cosine_sims, args.output_dir)

    # Save delta for further analysis
    delta_path = output_dir / "embedding_delta.npz"
    np.savez(delta_path,
             delta=delta,
             baseline=baseline,
             soc=soc,
             baseline_uids=baseline_uids,
             soc_uids=soc_uids)
    print(f"Saved delta embeddings to: {delta_path}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
