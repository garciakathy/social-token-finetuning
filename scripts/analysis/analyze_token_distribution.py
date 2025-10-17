#!/usr/bin/env python3
"""
Analyze the distribution of social tokens in training sequences.
Computes the fraction of SOC_G and SOC_L tokens over total sequence length.
"""

import os
import sys
import pickle
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def load_manifest(parent_dir):
    """Load the social packs manifest."""
    manifest_candidates = [
        "social_packs_manifest.csv",
        "scene_packs_manifest_recovered.csv",
        "scene_packs_manifest.csv"
    ]

    for candidate in manifest_candidates:
        manifest_path = os.path.join(parent_dir, candidate)
        if os.path.exists(manifest_path):
            print(f"Loading manifest: {manifest_path}")
            return pd.read_csv(manifest_path)

    raise FileNotFoundError(f"No manifest found in {parent_dir}")


def analyze_token_distribution(parent_dir, col_transcript="caption", max_samples=None):
    """
    Analyze social token distribution in sequences.

    Args:
        parent_dir: Parent directory with social_packs_manifest.csv
        col_transcript: Column name for transcripts
        max_samples: Max number of samples to analyze (None = all)
    """

    # Load manifest
    manifest = load_manifest(parent_dir)
    print(f"Loaded {len(manifest)} clips from manifest")

    # Check what status values exist
    if 'status' in manifest.columns:
        print(f"\nStatus value counts:")
        print(manifest['status'].value_counts())

        # Filter to successful clips only if status column exists and has 'success' values
        if 'success' in manifest['status'].values:
            manifest = manifest[manifest['status'] == 'success'].copy()
            print(f"Filtered to {len(manifest)} clips with status='success'")
        else:
            print("Warning: No clips with status='success', using all clips")
    else:
        print("No 'status' column found, using all clips")

    if len(manifest) == 0:
        print("\nERROR: No clips to analyze after filtering!")
        print("Check your manifest file and ensure clips were processed successfully.")
        sys.exit(1)

    if max_samples:
        manifest = manifest.head(max_samples)
        print(f"Limited to {max_samples} samples for analysis")

    # Statistics collectors
    stats = {
        'total_sequences': 0,
        'total_words': 0,
        'total_soc_g': 0,
        'total_soc_l': 0,
        'total_soc': 0,
        'fractions': [],
        'soc_g_fractions': [],
        'soc_l_fractions': [],
        'seq_lengths': [],
        'num_soc_g_per_seq': [],
        'num_soc_l_per_seq': [],
        'num_words_per_seq': [],
    }

    # Analyze each clip
    print(f"\nAnalyzing {len(manifest)} sequences...")

    for idx, row in manifest.iterrows():
        meta_pkl = row.get('meta_pkl')

        if pd.isna(meta_pkl) or not os.path.exists(meta_pkl):
            continue

        # Load metadata
        try:
            with open(meta_pkl, 'rb') as f:
                meta = pickle.load(f)
        except Exception as e:
            if stats['total_sequences'] < 5:  # Only print first few errors
                print(f"Warning: Could not load {meta_pkl}: {e}")
            continue

        # Get caption
        caption = meta.get(col_transcript, "")
        if not caption:
            continue

        # Get number of words (approximate with space splitting)
        num_words = meta.get('n_words', len(caption.split()))

        # Get POS-tagged words (nouns and verbs that get local tokens)
        pos_words = meta.get('pos_words', [])
        num_locals = len(pos_words)

        # In the actual model, tokens inserted are:
        # <bos> + <SOC_G> + caption_words + <SOC_L> (one per noun/verb) + <eos>

        num_soc_g = 1  # Always 1 global token
        num_soc_l = num_locals  # One per noun/verb

        # Approximate total tokens as:
        # Special tokens (bos, eos) + SOC_G + words + SOC_L tokens
        # Using words as proxy for text tokens (actual tokenization may split words)
        total_items = 2 + 1 + num_words + num_soc_l  # bos/eos + SOC_G + words + SOC_L

        # Total social tokens
        total_soc = num_soc_g + num_soc_l

        # Fraction
        soc_fraction = total_soc / total_items if total_items > 0 else 0
        soc_g_fraction = num_soc_g / total_items if total_items > 0 else 0
        soc_l_fraction = num_soc_l / total_items if total_items > 0 else 0

        # Update statistics
        stats['total_sequences'] += 1
        stats['total_words'] += num_words
        stats['total_soc_g'] += num_soc_g
        stats['total_soc_l'] += num_soc_l
        stats['total_soc'] += total_soc
        stats['fractions'].append(soc_fraction)
        stats['soc_g_fractions'].append(soc_g_fraction)
        stats['soc_l_fractions'].append(soc_l_fraction)
        stats['seq_lengths'].append(total_items)
        stats['num_soc_g_per_seq'].append(num_soc_g)
        stats['num_soc_l_per_seq'].append(num_soc_l)
        stats['num_words_per_seq'].append(num_words)

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(manifest)} sequences...")

    print(f"Analysis complete! Analyzed {stats['total_sequences']} sequences.")

    if stats['total_sequences'] == 0:
        print("\nERROR: No valid sequences found!")
        print("Possible issues:")
        print("  - meta_pkl files don't exist")
        print("  - meta_pkl files are corrupted")
        print("  - Caption column name is wrong")
        sys.exit(1)

    return stats


def plot_statistics(stats, output_dir=None):
    """Create visualization of token distribution statistics."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Social Token Distribution Analysis', fontsize=16, fontweight='bold')

    fractions = np.array(stats['fractions'])
    soc_g_fractions = np.array(stats['soc_g_fractions'])
    soc_l_fractions = np.array(stats['soc_l_fractions'])
    seq_lengths = np.array(stats['seq_lengths'])
    num_soc_l = np.array(stats['num_soc_l_per_seq'])

    # Plot 1: Histogram of total social token fraction
    ax1 = axes[0, 0]
    ax1.hist(fractions * 100, bins=50, color='purple', alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(fractions) * 100, color='red', linestyle='--',
                linewidth=2, label=f'Mean: {np.mean(fractions)*100:.1f}%')
    ax1.set_xlabel('Social Token Fraction (%)', fontsize=12)
    ax1.set_ylabel('Number of Sequences', fontsize=12)
    ax1.set_title('Distribution of Social Token Fraction', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Global vs Local token fractions
    ax2 = axes[0, 1]
    ax2.hist(soc_g_fractions * 100, bins=30, alpha=0.6, color='blue',
             label=f'Global (mean: {np.mean(soc_g_fractions)*100:.1f}%)', edgecolor='black')
    ax2.hist(soc_l_fractions * 100, bins=30, alpha=0.6, color='orange',
             label=f'Local (mean: {np.mean(soc_l_fractions)*100:.1f}%)', edgecolor='black')
    ax2.set_xlabel('Token Fraction (%)', fontsize=12)
    ax2.set_ylabel('Number of Sequences', fontsize=12)
    ax2.set_title('Global vs Local Token Fractions', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Sequence length distribution
    ax3 = axes[1, 0]
    ax3.hist(seq_lengths, bins=50, color='green', alpha=0.7, edgecolor='black')
    ax3.axvline(np.mean(seq_lengths), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {np.mean(seq_lengths):.1f} items')
    ax3.set_xlabel('Sequence Length (approximate tokens)', fontsize=12)
    ax3.set_ylabel('Number of Sequences', fontsize=12)
    ax3.set_title('Sequence Length Distribution', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Number of local tokens per sequence
    ax4 = axes[1, 1]
    max_locals = int(np.max(num_soc_l)) + 2
    bins = range(0, min(max_locals, 100))  # Cap at 100 for readability
    ax4.hist(num_soc_l, bins=bins, color='orange', alpha=0.7, edgecolor='black')
    ax4.axvline(np.mean(num_soc_l), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {np.mean(num_soc_l):.1f}')
    ax4.set_xlabel('Number of <SOC_L> tokens', fontsize=12)
    ax4.set_ylabel('Number of Sequences', fontsize=12)
    ax4.set_title('Local Tokens per Sequence', fontsize=13, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / 'token_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
    else:
        plt.show()


def print_summary(stats):
    """Print summary statistics."""

    print("\n" + "="*70)
    print("SOCIAL TOKEN DISTRIBUTION SUMMARY")
    print("="*70)

    total_items = stats['total_words'] + stats['total_soc'] + 2 * stats['total_sequences']  # +2 for bos/eos per seq

    print(f"\nDataset Statistics:")
    print(f"  Total sequences analyzed:     {stats['total_sequences']:,}")
    print(f"  Total words (all sequences):  {stats['total_words']:,}")
    print(f"  Total <SOC_G> tokens:         {stats['total_soc_g']:,}")
    print(f"  Total <SOC_L> tokens:         {stats['total_soc_l']:,}")
    print(f"  Total social tokens:          {stats['total_soc']:,}")
    print(f"  Total items (approx):         {total_items:,}")

    overall_fraction = stats['total_soc'] / total_items if total_items > 0 else 0
    global_fraction = stats['total_soc_g'] / total_items if total_items > 0 else 0
    local_fraction = stats['total_soc_l'] / total_items if total_items > 0 else 0

    print(f"\nOverall Token Fractions:")
    print(f"  Social tokens / Total:        {overall_fraction*100:.2f}%")
    print(f"  Global tokens / Total:        {global_fraction*100:.2f}%")
    print(f"  Local tokens / Total:         {local_fraction*100:.2f}%")
    print(f"  Text tokens / Total:          {(1-overall_fraction)*100:.2f}%")

    fractions = np.array(stats['fractions'])
    soc_g_fractions = np.array(stats['soc_g_fractions'])
    soc_l_fractions = np.array(stats['soc_l_fractions'])
    seq_lengths = np.array(stats['seq_lengths'])
    num_soc_l = np.array(stats['num_soc_l_per_seq'])
    num_words = np.array(stats['num_words_per_seq'])

    print(f"\nPer-Sequence Statistics:")
    print(f"  Mean social token fraction:   {np.mean(fractions)*100:.2f}% ± {np.std(fractions)*100:.2f}%")
    print(f"  Median social token fraction: {np.median(fractions)*100:.2f}%")
    print(f"  Min social token fraction:    {np.min(fractions)*100:.2f}%")
    print(f"  Max social token fraction:    {np.max(fractions)*100:.2f}%")

    print(f"\nSequence Length Statistics:")
    print(f"  Mean sequence length:         {np.mean(seq_lengths):.1f} items")
    print(f"  Median sequence length:       {np.median(seq_lengths):.0f} items")
    print(f"  Min sequence length:          {np.min(seq_lengths)} items")
    print(f"  Max sequence length:          {np.max(seq_lengths)} items")

    print(f"\nWords per Sequence:")
    print(f"  Mean words per sequence:      {np.mean(num_words):.1f}")
    print(f"  Median words per sequence:    {np.median(num_words):.0f}")

    print(f"\nLocal Tokens per Sequence:")
    print(f"  Mean <SOC_L> per sequence:    {np.mean(num_soc_l):.1f}")
    print(f"  Median <SOC_L> per sequence:  {np.median(num_soc_l):.0f}")
    print(f"  Min <SOC_L> per sequence:     {np.min(num_soc_l)}")
    print(f"  Max <SOC_L> per sequence:     {np.max(num_soc_l)}")

    print(f"\nInterpretation:")
    print(f"  On average, {np.mean(fractions)*100:.1f}% of each sequence consists of")
    print(f"  visual social tokens (both global and local), with the remaining")
    print(f"  {(1-np.mean(fractions))*100:.1f}% being text and special tokens.")
    print()
    print(f"  Each sequence has exactly 1 <SOC_G> token ({np.mean(soc_g_fractions)*100:.2f}%)")
    print(f"  and an average of {np.mean(num_soc_l):.1f} <SOC_L> tokens ({np.mean(soc_l_fractions)*100:.1f}%),")
    print(f"  corresponding to nouns and verbs in the caption.")
    print()
    print(f"  The average caption has {np.mean(num_words):.1f} words, of which")
    print(f"  {np.mean(num_soc_l):.1f} ({np.mean(num_soc_l)/np.mean(num_words)*100:.1f}%) are nouns/verbs")
    print(f"  that receive visual grounding via <SOC_L> tokens.")

    print("="*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze social token distribution in training sequences'
    )
    parser.add_argument('--parent-dir', type=str,
                       default=os.path.expanduser("~/data/seamless/outputs/social_tokens_fixed"),
                       help='Parent directory with social_packs_manifest.csv')
    parser.add_argument('--col-transcript', type=str, default='caption',
                       help='Column name for transcripts in metadata')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Max number of samples to analyze (default: all)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save plots (default: show plots)')

    args = parser.parse_args()

    # Analyze
    stats = analyze_token_distribution(
        parent_dir=args.parent_dir,
        col_transcript=args.col_transcript,
        max_samples=args.max_samples
    )

    # Print summary
    print_summary(stats)

    # Plot
    plot_statistics(stats, output_dir=args.output_dir)
