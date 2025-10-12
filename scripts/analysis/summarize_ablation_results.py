#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize ablation study results from metrics CSV files.
Compares perplexity across different ablation modes.
"""

import pandas as pd
import argparse
from pathlib import Path
import sys

def load_metrics(csv_path):
    """Load metrics CSV and return summary statistics."""
    if not Path(csv_path).exists():
        print(f"Warning: {csv_path} not found, skipping...")
        return None

    df = pd.read_csv(csv_path)
    return df

def summarize_ablation(base_dir, ablation_name):
    """Summarize results for a single ablation run."""
    metrics_path = Path(base_dir) / ablation_name / "logs" / "metrics.csv"

    if not metrics_path.exists():
        return None

    df = load_metrics(metrics_path)

    # Get final epoch validation results
    val_summary = df[df['split'] == 'val_summary'].copy()
    if val_summary.empty:
        return None

    last_epoch = val_summary.sort_values('epoch').iloc[-1]

    # Get final test results
    test_summary = df[df['split'] == 'test_summary']
    test_row = test_summary.iloc[0] if not test_summary.empty else None

    summary = {
        'ablation': ablation_name,
        'best_epoch': int(last_epoch['epoch']) if pd.notna(last_epoch['epoch']) else None,
        'val_ppl_vis': last_epoch.get('val_ppl_vis'),
        'val_ppl_no_vis': last_epoch.get('val_ppl_no_vis'),
    }

    if test_row is not None:
        summary['test_ppl_vis'] = test_row.get('test_ppl_vis')
        summary['test_ppl_no_vis'] = test_row.get('test_ppl_no_vis')

    # Get individual split results from last epoch
    val_splits = df[(df['epoch'] == last_epoch['epoch']) &
                    (df['split'].str.startswith('val_')) &
                    (df['split'] != 'val_summary') &
                    (df['step'] == -1)]

    for _, row in val_splits.iterrows():
        split_name = row['split'].replace('val_', '')
        summary[f'val_ppl_{split_name}'] = row.get('ppl')

    # Get test split results
    test_splits = df[(df['split'].str.startswith('test_')) &
                     (df['split'] != 'test_summary') &
                     (df['step'] == -1)]

    for _, row in test_splits.iterrows():
        split_name = row['split'].replace('test_', '')
        summary[f'test_ppl_{split_name}'] = row.get('ppl')

    return summary

def print_comparison_table(summaries):
    """Print a formatted comparison table."""
    if not summaries:
        print("No results to compare!")
        return

    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS SUMMARY")
    print("="*80)

    # Validation results
    print("\n--- VALIDATION PERPLEXITY ---")
    print(f"{'Ablation':<20} {'Both':<12} {'Global':<12} {'Local':<12} {'None':<12}")
    print("-" * 80)

    for s in summaries:
        abl = s['ablation']
        both = s.get('val_ppl_vis', s.get('val_ppl_both', '-'))
        glob = s.get('val_ppl_global_only', '-')
        loc = s.get('val_ppl_local_only', '-')
        none = s.get('val_ppl_no_vis', s.get('val_ppl_novis', '-'))

        both_str = f"{both:.2f}" if isinstance(both, (int, float)) else str(both)
        glob_str = f"{glob:.2f}" if isinstance(glob, (int, float)) else str(glob)
        loc_str = f"{loc:.2f}" if isinstance(loc, (int, float)) else str(loc)
        none_str = f"{none:.2f}" if isinstance(none, (int, float)) else str(none)

        print(f"{abl:<20} {both_str:<12} {glob_str:<12} {loc_str:<12} {none_str:<12}")

    # Test results
    print("\n--- TEST PERPLEXITY ---")
    print(f"{'Ablation':<20} {'Both':<12} {'Global':<12} {'Local':<12} {'None':<12}")
    print("-" * 80)

    for s in summaries:
        abl = s['ablation']
        both = s.get('test_ppl_vis', s.get('test_ppl_both', '-'))
        glob = s.get('test_ppl_global_only', '-')
        loc = s.get('test_ppl_local_only', '-')
        none = s.get('test_ppl_no_vis', s.get('test_ppl_novis', '-'))

        both_str = f"{both:.2f}" if isinstance(both, (int, float)) else str(both)
        glob_str = f"{glob:.2f}" if isinstance(glob, (int, float)) else str(glob)
        loc_str = f"{loc:.2f}" if isinstance(loc, (int, float)) else str(loc)
        none_str = f"{none:.2f}" if isinstance(none, (int, float)) else str(none)

        print(f"{abl:<20} {both_str:<12} {glob_str:<12} {loc_str:<12} {none_str:<12}")

    # Analysis
    print("\n--- ANALYSIS ---")

    # Find the best overall configuration
    best_abl = None
    best_ppl = float('inf')

    for s in summaries:
        both = s.get('test_ppl_vis', s.get('test_ppl_both'))
        if isinstance(both, (int, float)) and both < best_ppl:
            best_ppl = both
            best_abl = s['ablation']

    if best_abl:
        print(f"\nBest configuration: {best_abl} (test ppl = {best_ppl:.2f})")

    # Try to find the "both" trained model and analyze contributions
    both_results = next((s for s in summaries if s['ablation'] == 'both_tokens'), None)

    if both_results:
        both_val = both_results.get('val_ppl_vis', both_results.get('val_ppl_both'))
        glob_val = both_results.get('val_ppl_global_only')
        loc_val = both_results.get('val_ppl_local_only')
        none_val = both_results.get('val_ppl_no_vis', both_results.get('val_ppl_novis'))

        print("\nContributions (from 'both_tokens' model):")

        if isinstance(both_val, (int, float)) and isinstance(none_val, (int, float)):
            improvement = ((none_val - both_val) / none_val) * 100
            print(f"  Overall improvement: {improvement:.1f}% reduction in perplexity")

        if isinstance(both_val, (int, float)) and isinstance(glob_val, (int, float)):
            local_contrib = ((glob_val - both_val) / glob_val) * 100
            print(f"  Local tokens contribution: {local_contrib:.1f}% additional reduction")

        if isinstance(both_val, (int, float)) and isinstance(loc_val, (int, float)):
            global_contrib = ((loc_val - both_val) / loc_val) * 100
            print(f"  Global tokens contribution: {global_contrib:.1f}% additional reduction")

    print("\n" + "="*80 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Summarize ablation study results from metrics CSV files."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        required=True,
        help="Base directory containing ablation subdirectories (e.g., /path/to/ablation_study/)"
    )
    parser.add_argument(
        "--ablations",
        type=str,
        nargs="+",
        default=["both_tokens", "global_only", "local_only", "no_tokens"],
        help="List of ablation subdirectory names to analyze"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="",
        help="Optional: Save summary table to CSV file"
    )

    args = parser.parse_args()

    # Load all ablation results
    summaries = []
    for abl in args.ablations:
        summary = summarize_ablation(args.base_dir, abl)
        if summary:
            summaries.append(summary)
        else:
            print(f"Warning: No valid results found for ablation '{abl}'")

    if not summaries:
        print("Error: No valid ablation results found!")
        sys.exit(1)

    # Print comparison
    print_comparison_table(summaries)

    # Optionally save to CSV
    if args.output_csv:
        df = pd.DataFrame(summaries)
        df.to_csv(args.output_csv, index=False)
        print(f"Summary saved to: {args.output_csv}")

if __name__ == "__main__":
    main()
