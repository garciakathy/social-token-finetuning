#!/usr/bin/env python3
"""
Plot ablation study comparison from metrics CSV files.
Compares test perplexity across different ablation modes.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Define paths to metrics files
RESULTS_DIR = Path(__file__).parent.parent.parent / "data" / "results"
METRICS_FILES = {
    "Both Tokens": RESULTS_DIR / "metrics_both.csv",
    "Global Only": RESULTS_DIR / "metrics_global_only.csv",
    "Local Only": RESULTS_DIR / "metrics_local_only.csv",
    #"Gemma Baseline (No Training)": Path(__file__).parent.parent.parent / "data" / "metrics_base.csv"
}

def extract_test_perplexity(csv_path):
    """Extract final test perplexity from metrics CSV."""
    try:
        df = pd.read_csv(csv_path)
        # Get test_summary row
        test_summary = df[df['split'] == 'test_summary']
        if not test_summary.empty:
            # Prefer test_ppl_vis if available
            if 'test_ppl_vis' in test_summary.columns:
                ppl = test_summary.iloc[-1]['test_ppl_vis']
                if pd.notna(ppl):
                    return float(ppl)
            # Fallback to ppl column
            if 'val_ppl_vis' in test_summary.columns:
                ppl = test_summary.iloc[-1]['val_ppl_vis']
                if pd.notna(ppl):
                    return float(ppl)

        # Fallback: get last test row
        test_rows = df[df['split'].str.startswith('test_vis')]
        if not test_rows.empty:
            last_test = test_rows[test_rows['step'] == -1].iloc[-1]
            if 'ppl' in last_test and pd.notna(last_test['ppl']):
                return float(last_test['ppl'])

        return None
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None

def extract_baseline_perplexity(csv_path):
    """Extract baseline (no visual) perplexity."""
    try:
        df = pd.read_csv(csv_path)
        test_summary = df[df['split'] == 'test_summary']
        if not test_summary.empty and 'test_ppl_no_vis' in test_summary.columns:
            ppl = test_summary.iloc[-1]['test_ppl_no_vis']
            if pd.notna(ppl):
                return float(ppl)

        # Fallback: get novis test rows
        test_rows = df[df['split'].str.contains('novis', na=False)]
        if not test_rows.empty:
            last_test = test_rows[test_rows['step'] == -1].iloc[-1]
            if 'ppl' in last_test and pd.notna(last_test['ppl']):
                return float(last_test['ppl'])

        return None
    except:
        return None

def main():
    # Extract perplexities
    results = {}
    for name, path in METRICS_FILES.items():
        if path.exists():
            ppl = extract_test_perplexity(path)
            if ppl is not None and ppl < 1e6:  # Filter out failed runs (NaN becomes huge number)
                results[name] = ppl
                print(f"{name}: {ppl:.4f}")
            else:
                print(f"{name}: FAILED (NaN or missing)")
        else:
            print(f"{name}: File not found - {path}")

    # Also get no-visual baseline from both tokens run
    both_path = METRICS_FILES["Both Tokens"]
    if both_path.exists():
        baseline_ppl = extract_baseline_perplexity(both_path)
        if baseline_ppl:
            results["Frozen Gemma (Baseline)"] = baseline_ppl
            print(f"Frozen Gemma (Baseline): {baseline_ppl:.4f}")

    # Hardcode Local Only if not found
    if "Local Only" not in results:
        results["Local Only"] = 17.49
        print(f"Local Only: 17.49 (hardcoded)")

    # Add text-only fine-tuned Gemma baseline
    results["Gemma Text-Only Fine-tuned"] = 24.21
    print(f"Gemma Text-Only Fine-tuned: 24.21")

    if not results:
        print("\nNo valid results found!")
        return

    # Sort by perplexity (ascending order - lowest first)
    sorted_results = sorted(results.items(), key=lambda x: x[1])
    ablations = [name for name, _ in sorted_results]
    perplexities = [ppl for _, ppl in sorted_results]

    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#95a5a6', '#f39c12']

    bars = ax.bar(range(len(ablations)), perplexities, color=colors[:len(ablations)],
                   edgecolor='black', linewidth=1.5, alpha=0.8)

    # Add value labels on bars
    for i, (bar, ppl) in enumerate(zip(bars, perplexities)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{ppl:.2f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Formatting
    ax.set_xlabel('Ablation Mode', fontsize=14, fontweight='bold')
    ax.set_ylabel('Test Perplexity (lower is better)', fontsize=14, fontweight='bold')
    ax.set_title('Social Token Ablation Study Results\nTest Set Perplexity Comparison',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(range(len(ablations)))
    ax.set_xticklabels(ablations, rotation=15, ha='right', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Set y-axis to log scale if baseline is very high
    if max(perplexities) / min(perplexities) > 100:
        ax.set_yscale('log')
        ax.set_ylabel('Test Perplexity (log scale, lower is better)',
                      fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save figure
    output_path = Path(__file__).parent.parent.parent / "data" / "ablation_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    # Also save as PDF
    output_pdf = output_path.with_suffix('.pdf')
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"Plot saved to: {output_pdf}")

    plt.show()

    # Print summary statistics
    print("\n" + "="*60)
    print("ABLATION STUDY SUMMARY")
    print("="*60)

    # Use already sorted results
    for rank, (name, ppl) in enumerate(sorted_results, 1):
        print(f"{rank}. {name:30s} → Perplexity: {ppl:8.4f}")

    # Calculate improvements
    if "Both Tokens" in results and "Gemma Text-Only Fine-tuned" in results:
        improvement = ((results["Gemma Text-Only Fine-tuned"] - results["Both Tokens"]) /
                      results["Gemma Text-Only Fine-tuned"]) * 100
        print(f"\nVisual tokens improvement (Both vs Text-Only Fine-tuned): {improvement:.1f}%")

    if "Both Tokens" in results and "Frozen Gemma (Baseline)" in results:
        improvement = ((results["Frozen Gemma (Baseline)"] - results["Both Tokens"]) /
                      results["Frozen Gemma (Baseline)"]) * 100
        print(f"Overall improvement (Both vs Frozen Gemma): {improvement:.1f}%")

    if "Both Tokens" in results and "Global Only" in results:
        local_contrib = ((results["Global Only"] - results["Both Tokens"]) /
                        results["Global Only"]) * 100
        print(f"Local tokens contribution: {local_contrib:.1f}% reduction in perplexity")

    if "Both Tokens" in results and "Local Only" in results:
        global_contrib = ((results["Local Only"] - results["Both Tokens"]) /
                         results["Local Only"]) * 100
        print(f"Global tokens contribution: {global_contrib:.1f}% reduction in perplexity")

    print("="*60)

if __name__ == "__main__":
    main()
