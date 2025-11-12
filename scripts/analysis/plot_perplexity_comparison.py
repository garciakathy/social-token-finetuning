#!/usr/bin/env python3
"""
Plot perplexity comparison from training log file for a single model.
Compares test perplexity across different ablation modes:
  - both: Full model with both global and local social tokens
  - local_only: Only local social tokens (mask out global token)
  - global_only: Only global social token (mask out local tokens)
  - frozen_baseline: No visual information (text-only)
  - text_only_finetuned: Text-only model fine-tuned on captions (optional)
"""
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

def extract_model_name(log_path):
    """Extract model name from log file."""
    with open(log_path, 'r') as f:
        for line in f:
            # Look for "Model: google/gemma-2-2b" or "Model: google/gemma-2-2b-it"
            if line.startswith("Model:"):
                model = line.split("Model:")[1].strip()
                # Extract just the model name (gemma-2-2b or gemma-2-2b-it)
                if "gemma-2-2b-it" in model:
                    return "gemma-2-2b-it"
                elif "gemma-2-2b" in model:
                    return "gemma-2-2b"
    return None

def extract_test_perplexity(log_path):
    """Extract test perplexity from log file.

    Looks for line like:
    [TEST] ppl: both=2.79 | local_only=166.89 | global_only=5.84 | frozen_baseline=206.71
    """
    with open(log_path, 'r') as f:
        for line in f:
            if "[TEST] ppl:" in line:
                # Extract values using regex
                both_match = re.search(r'both=([\d.]+)', line)
                local_match = re.search(r'local_only=([\d.]+)', line)
                global_match = re.search(r'global_only=([\d.]+)', line)
                frozen_match = re.search(r'frozen_baseline=([\d.]+)', line)

                if all([both_match, local_match, global_match, frozen_match]):
                    return {
                        'both': float(both_match.group(1)),
                        'local_only': float(local_match.group(1)),
                        'global_only': float(global_match.group(1)),
                        'frozen_baseline': float(frozen_match.group(1))
                    }
    return None

def extract_text_only_perplexity_from_csv(csv_path):
    """Extract best validation perplexity from text-only training metrics CSV.

    CSV format expected:
    epoch,train_loss,train_ppl,val_loss,val_ppl,lr
    1,4.007867,55.029,3.226903,25.201498,1.4e-05
    ...

    Returns the minimum val_ppl across all epochs (best performance).
    """
    try:
        df = pd.read_csv(csv_path)

        if 'val_ppl' not in df.columns:
            print(f"Warning: 'val_ppl' column not found in {csv_path}")
            print(f"Available columns: {df.columns.tolist()}")
            return None

        # Get the best (minimum) validation perplexity
        best_val_ppl = df['val_ppl'].min()

        return best_val_ppl

    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")
        return None

def plot_ablation_comparison(log_path, text_only_csv=None, output_dir=None):
    """Create bar chart showing ablation modes for a single model."""

    # Extract model name
    model_name = extract_model_name(log_path)

    if not model_name:
        print("Could not extract model name from log file")
        return None

    print(f"Model: {model_name}")

    # Extract metrics
    metrics = extract_test_perplexity(log_path)

    if not metrics:
        print("Could not extract test perplexity from log file")
        return None

    print(f"\nMetrics: {metrics}")

    # Extract text-only finetuned perplexity if provided
    text_only_ppl = None
    if text_only_csv:
        text_only_ppl = extract_text_only_perplexity_from_csv(text_only_csv)
        if text_only_ppl:
            print(f"Text-only finetuned PPL: {text_only_ppl:.2f}")
            metrics['text_only_finetuned'] = text_only_ppl
        else:
            print(f"Warning: Could not extract perplexity from text-only CSV: {text_only_csv}")

    # Prepare data
    if text_only_ppl is not None:
        ablation_modes = ['All Social\nTokens', 'Global Only\nTokens', 'Local Only\nTokens',
                         'Text-Only\nFinetuned', 'Frozen Gemma\nBaseline']
        values = [
            metrics['both'],
            metrics['global_only'],
            metrics['local_only'],
            metrics['text_only_finetuned'],
            metrics['frozen_baseline']
        ]
        # Color scheme: green, blue, orange, purple, red
        colors = ['#2ecc71', '#3498db', '#e67e22', '#9b59b6', '#e74c3c']
    else:
        ablation_modes = ['All Social\nTokens', 'Global Only\nTokens', 'Local Only\nTokens',
                         'Frozen Gemma\nBaseline']
        values = [
            metrics['both'],
            metrics['global_only'],
            metrics['local_only'],
            metrics['frozen_baseline']
        ]
        # Color scheme: green, blue, orange, red
        colors = ['#2ecc71', '#3498db', '#e67e22', '#e74c3c']

    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 7))

    bars = ax.bar(ablation_modes, values,
                   color=colors, edgecolor='black', linewidth=1.5, alpha=0.85, width=0.6)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Formatting
    ax.set_xlabel('Ablation Mode', fontsize=14, fontweight='bold')
    ax.set_ylabel('Test Perplexity (log scale)', fontsize=14, fontweight='bold')

    # Model name in title
    if model_name == "gemma-2-2b":
        model_display = "Gemma-2-2B Perplexity"
    elif model_name == "gemma-2-2b-it":
        model_display = "Gemma-2-2B-IT Perplexity"
    else:
        model_display = model_name

    ax.set_title(f'{model_display}',
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Use log scale if difference is large
    max_val = max(values)
    min_val = values[0]  # 'both' mode
    if max_val / min_val > 10:
        ax.set_yscale('log')
        ax.set_ylabel('Test Perplexity (log scale)', fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "data" / "results" / "perplexity"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"ppl_ablation_comparison_{model_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    output_pdf = output_path.with_suffix('.pdf')
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"Plot saved to: {output_pdf}")

    plt.close()

    return metrics, model_name

def print_summary(metrics, model_name):
    """Print summary statistics."""

    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80)

    # Model display name
    if model_name == "gemma-2-2b":
        model_display = "Gemma-2-2B (Base Model)"
    elif model_name == "gemma-2-2b-it":
        model_display = "Gemma-2-2B-IT (Instruction-Tuned)"
    else:
        model_display = model_name

    print(f"\n{model_display}:")
    print(f"  Both tokens:        {metrics['both']:.2f} PPL (best)")
    print(f"  Global only:        {metrics['global_only']:.2f} PPL")
    print(f"  Local only:         {metrics['local_only']:.2f} PPL")

    if 'text_only_finetuned' in metrics:
        print(f"  Text-only finetuned:{metrics['text_only_finetuned']:.2f} PPL")

    print(f"  Frozen baseline:    {metrics['frozen_baseline']:.2f} PPL (worst)")

    # Calculate improvements
    improvement_both = metrics['frozen_baseline'] / metrics['both']
    improvement_global = metrics['frozen_baseline'] / metrics['global_only']
    improvement_local = metrics['frozen_baseline'] / metrics['local_only']

    print(f"\n  Improvements over frozen baseline:")
    print(f"    Both tokens:      {improvement_both:.1f}x better")
    print(f"    Global only:      {improvement_global:.1f}x better")
    print(f"    Local only:       {improvement_local:.1f}x better")

    if 'text_only_finetuned' in metrics:
        improvement_text_only = metrics['frozen_baseline'] / metrics['text_only_finetuned']
        print(f"    Text-only finetuned: {improvement_text_only:.1f}x better")

    # Key insights
    print("\n" + "-"*80)
    print("TOKEN CONTRIBUTION ANALYSIS")
    print("-"*80)

    # Which token type matters more?
    degradation_no_local = metrics['global_only'] / metrics['both']
    degradation_no_global = metrics['local_only'] / metrics['both']

    print(f"\n  Removing local tokens:  {degradation_no_local:.2f}x worse PPL")
    print(f"  Removing global token:  {degradation_no_global:.2f}x worse PPL")

    if degradation_no_global > degradation_no_local:
        print(f"\n  → Global token is MORE critical")
        print(f"    (Removing it causes {degradation_no_global:.1f}x degradation vs {degradation_no_local:.1f}x)")
    else:
        print(f"\n  → Local tokens are MORE critical")
        print(f"    (Removing them causes {degradation_no_local:.1f}x degradation vs {degradation_no_global:.1f}x)")

    # Absolute improvements
    improvement_local_tokens = metrics['global_only'] - metrics['both']
    improvement_global_token = metrics['local_only'] - metrics['both']

    print(f"\n  Contribution to performance:")
    print(f"    Local tokens reduce PPL by:  {improvement_local_tokens:.2f} points")
    print(f"    Global token reduces PPL by:  {improvement_global_token:.2f} points")

    if 'text_only_finetuned' in metrics:
        # Compare social tokens vs text-only finetuning
        print(f"\n  Social tokens vs text-only finetuning:")
        social_vs_text = metrics['text_only_finetuned'] / metrics['both']
        print(f"    Social tokens are {social_vs_text:.1f}x better than text-only finetuning")

    print("\n" + "="*80)

def main():
    parser = argparse.ArgumentParser(
        description="Plot perplexity comparison from training log file"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        required=True,
        help="Path to training log file with ablation results"
    )
    parser.add_argument(
        "--text-only-csv",
        type=str,
        default=None,
        help="Optional: Path to text-only finetuning metrics CSV file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plots (default: data/results/perplexity)"
    )

    args = parser.parse_args()

    print("="*80)
    print("ABLATION STUDY ANALYSIS")
    print("="*80)
    print(f"\nReading log: {args.log_file}")
    if args.text_only_csv:
        print(f"Text-only CSV: {args.text_only_csv}")

    # Create ablation comparison plot
    print("\nGenerating ablation comparison plot...")
    result = plot_ablation_comparison(args.log_file, args.text_only_csv, args.output_dir)

    if result:
        metrics, model_name = result
        # Print summary
        print_summary(metrics, model_name)

if __name__ == "__main__":
    main()
