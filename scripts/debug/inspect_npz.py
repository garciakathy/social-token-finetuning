#!/usr/bin/env python3
"""Inspect the contents of local embeddings .npz files."""
import numpy as np
import pandas as pd
import sys
from pathlib import Path

if len(sys.argv) > 1:
    parent_dir = sys.argv[1]
else:
    print("Usage: python inspect_npz.py <parent_dir>")
    sys.exit(1)

manifest_path = Path(parent_dir) / "scene_packs_manifest_recovered.csv"
df = pd.read_csv(manifest_path)

print("="*80)
print("INSPECTING LOCAL EMBEDDINGS FILES")
print("="*80)

for idx in range(min(5, len(df))):
    row = df.iloc[idx]
    print(f"\n--- Sample {idx}: {row['clip_id']} ---")

    # Load locals
    locals_path = row['locals_npz']
    if pd.notna(locals_path) and Path(locals_path).exists():
        print(f"  locals_npz: {Path(locals_path).name}")
        try:
            data = np.load(locals_path)
            print(f"  Keys in .npz: {list(data.keys())}")

            for key in data.keys():
                arr = data[key]
                print(f"\n  '{key}':")
                print(f"    Shape: {arr.shape}")
                print(f"    Dtype: {arr.dtype}")
                if arr.size > 0:
                    print(f"    Min: {arr.min():.6f}, Max: {arr.max():.6f}")
                    print(f"    Mean: {arr.mean():.6f}, Std: {arr.std():.6f}")

                    # Check for NaN/Inf
                    if np.isnan(arr).any():
                        print(f"    ⚠️  Contains NaN: {np.isnan(arr).sum()} values")
                    if np.isinf(arr).any():
                        print(f"    ⚠️  Contains Inf: {np.isinf(arr).sum()} values")

                    # Check diversity
                    if len(arr.shape) == 2 and arr.shape[0] > 1:
                        # Calculate pairwise differences
                        pairwise_diffs = np.abs(arr[:-1] - arr[1:]).max()
                        print(f"    Max pairwise diff: {pairwise_diffs:.6f}")

                        # Calculate cosine similarities
                        from numpy.linalg import norm
                        sims = []
                        for i in range(min(10, arr.shape[0] - 1)):
                            v1, v2 = arr[i], arr[i+1]
                            cos_sim = np.dot(v1, v2) / (norm(v1) * norm(v2) + 1e-8)
                            sims.append(cos_sim)
                        if sims:
                            print(f"    Avg cosine sim (first 10 pairs): {np.mean(sims):.6f}")
                            if np.mean(sims) > 0.99:
                                print(f"    ⚠️  WARNING: Very high similarity (tokens nearly identical!)")

                        if pairwise_diffs < 1e-6:
                            print(f"    ⚠️  WARNING: All embeddings are identical!")

        except Exception as e:
            print(f"  ❌ Error loading: {e}")
    else:
        print(f"  ❌ locals_npz not found at {locals_path}")

    # Load global
    global_path = row['global_vec']
    if pd.notna(global_path) and Path(global_path).exists():
        print(f"\n  global_vec: {Path(global_path).name}")
        try:
            global_emb = np.load(global_path)
            print(f"    Shape: {global_emb.shape}")
            print(f"    Min: {global_emb.min():.6f}, Max: {global_emb.max():.6f}")
            print(f"    Mean: {global_emb.mean():.6f}, Std: {global_emb.std():.6f}")
            print(f"    L2 norm: {np.linalg.norm(global_emb):.6f}")

            if np.isnan(global_emb).any():
                print(f"    ⚠️  Contains NaN!")
            if np.isinf(global_emb).any():
                print(f"    ⚠️  Contains Inf!")

        except Exception as e:
            print(f"  ❌ Error loading: {e}")
    else:
        print(f"  ❌ global_vec not found at {global_path}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Checked {min(5, len(df))} samples from {len(df)} total scenes")
print("\nIf you see 'identical embeddings' or 'very high similarity',")
print("this is the root cause of NaN during local-only training!")
