#!/usr/bin/env python3
"""Verify if embeddings are actually random noise."""
import numpy as np
import pandas as pd
from pathlib import Path
import sys

if len(sys.argv) > 1:
    parent_dir = sys.argv[1]
else:
    print("Usage: python verify_random_noise.py <parent_dir>")
    sys.exit(1)

print("="*80)
print("CHECKING IF EMBEDDINGS ARE RANDOM NOISE")
print("="*80)

# Expected std for random unit vectors in 768 dims
expected_std = 1.0 / np.sqrt(768)
print(f"\nExpected std for random normalized 768-D vectors: {expected_std:.6f}")

manifest_path = Path(parent_dir) / "scene_packs_manifest_recovered.csv"
df = pd.read_csv(manifest_path)

all_embeddings = []
all_stds = []
all_means = []

print(f"\nLoading embeddings from {min(20, len(df))} samples...")

for idx in range(min(20, len(df))):
    row = df.iloc[idx]
    locals_path = row['locals_npz']

    if pd.notna(locals_path) and Path(locals_path).exists():
        try:
            data = np.load(locals_path)
            for key in data.keys():
                emb = data[key]
                all_embeddings.append(emb)
                all_stds.append(emb.std())
                all_means.append(emb.mean())
        except:
            pass

if not all_embeddings:
    print("ERROR: Could not load any embeddings!")
    sys.exit(1)

all_embeddings = np.array(all_embeddings)
print(f"\nLoaded {len(all_embeddings)} embeddings")
print(f"Shape per embedding: {all_embeddings.shape[1]}")

# Statistics
print(f"\n" + "="*80)
print("STATISTICS ACROSS ALL EMBEDDINGS:")
print("="*80)
print(f"Mean of all means: {np.mean(all_means):.10f}")
print(f"Std of all means: {np.std(all_means):.10f}")
print(f"Mean of all stds: {np.mean(all_stds):.10f}")
print(f"Std of all stds: {np.std(all_stds):.10f}")

# Check if stds cluster around expected value
print(f"\n" + "="*80)
print("RANDOM NOISE TEST:")
print("="*80)
print(f"Expected std for random: {expected_std:.6f}")
print(f"Observed mean std: {np.mean(all_stds):.6f}")
print(f"Difference: {abs(np.mean(all_stds) - expected_std):.10f}")

if abs(np.mean(all_stds) - expected_std) < 0.0001:
    print("\n⚠️  WARNING: Std matches random noise exactly!")
    print("    This suggests embeddings are NOT from real DINO features.")
else:
    print("\n✓ Std does not match random noise pattern.")

# Check pairwise distances
print(f"\n" + "="*80)
print("PAIRWISE SIMILARITY TEST:")
print("="*80)

# Compare first 10 embeddings
n_compare = min(10, len(all_embeddings))
similarities = []

for i in range(n_compare):
    for j in range(i+1, n_compare):
        cos_sim = np.dot(all_embeddings[i], all_embeddings[j])
        similarities.append(cos_sim)

print(f"Comparing {n_compare} embeddings ({len(similarities)} pairs)")
print(f"Mean cosine similarity: {np.mean(similarities):.6f}")
print(f"Std of similarities: {np.std(similarities):.6f}")

if abs(np.mean(similarities)) < 0.05:
    print("\n⚠️  WARNING: Mean similarity near zero (expected for random vectors)")
    print("    Real DINO embeddings should have higher similarities for same video.")
else:
    print(f"\n✓ Mean similarity is {np.mean(similarities):.4f} (not random)")

# Check within-video vs across-video similarity
print(f"\n" + "="*80)
print("WITHIN-VIDEO DIVERSITY TEST:")
print("="*80)

within_sims = []
for idx in range(min(5, len(df))):
    row = df.iloc[idx]
    locals_path = row['locals_npz']

    if pd.notna(locals_path) and Path(locals_path).exists():
        try:
            data = np.load(locals_path)
            keys = list(data.keys())

            if len(keys) >= 2:
                # Compare first two tokens in same video
                emb1 = data[keys[0]]
                emb2 = data[keys[1]]
                sim = np.dot(emb1, emb2)
                within_sims.append(sim)

                print(f"Video {idx} ({row['clip_id']}): {keys[0]} vs {keys[1]}")
                print(f"  Cosine similarity: {sim:.6f}")

                if sim > 0.99:
                    print(f"  ⚠️  IDENTICAL!")
                elif abs(sim) < 0.05:
                    print(f"  ⚠️  UNCORRELATED (random noise)")
        except:
            pass

if within_sims:
    print(f"\nMean within-video similarity: {np.mean(within_sims):.6f}")
    if np.mean(within_sims) > 0.95:
        print("⚠️  Tokens within same video are nearly IDENTICAL!")
    elif abs(np.mean(within_sims)) < 0.1:
        print("⚠️  Tokens within same video are UNCORRELATED (random noise)")

# DIAGNOSIS
print(f"\n" + "="*80)
print("DIAGNOSIS:")
print("="*80)

issues = []

if abs(np.mean(all_stds) - expected_std) < 0.0001:
    issues.append("Standard deviation matches random noise (1/sqrt(768))")

if abs(np.mean(similarities)) < 0.05:
    issues.append("Mean pairwise similarity near zero (random vectors)")

if abs(np.mean(all_means)) < 1e-8:
    issues.append("All means exactly zero (suspicious centering)")

if within_sims and np.mean(within_sims) > 0.95:
    issues.append("Tokens within same video are identical")

if issues:
    print("❌ CRITICAL ISSUES DETECTED:")
    for issue in issues:
        print(f"  - {issue}")

    print("\n" + "="*80)
    print("ROOT CAUSE: Embeddings are NOT real DINO features!")
    print("="*80)
    print("\nPossible explanations:")
    print("  1. DINO model checkpoint not loaded correctly")
    print("  2. Preprocessing script generated random noise instead of features")
    print("  3. All frames are identical (unlikely given stats)")
    print("  4. Bug in preprocessing code that outputs random vectors")

    print("\nRECOMMENDED ACTION:")
    print("  Check preprocessing script: build_scene_packs_ooo.py")
    print("  Verify DINO checkpoint is loaded: --dino-checkpoint argument")
    print("  Re-run preprocessing with correct DINO model")
else:
    print("✓ Embeddings appear to be real DINO features")

print("="*80)
