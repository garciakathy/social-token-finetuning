#!/usr/bin/env python3
"""Quick script to inspect manifest structure."""
import pandas as pd
import sys
from pathlib import Path

if len(sys.argv) > 1:
    parent_dir = sys.argv[1]
else:
    print("Usage: python inspect_manifest.py <parent_dir>")
    sys.exit(1)

manifest_path = Path(parent_dir) / "scene_packs_manifest_recovered.csv"
if not manifest_path.exists():
    print(f"ERROR: {manifest_path} not found")
    sys.exit(1)

df = pd.read_csv(manifest_path)

print("="*80)
print("MANIFEST INSPECTION")
print("="*80)
print(f"\nTotal rows: {len(df)}")
print(f"\nColumns: {list(df.columns)}")
print("\n" + "="*80)
print("FIRST 3 ROWS:")
print("="*80)

for idx in range(min(3, len(df))):
    row = df.iloc[idx]
    print(f"\n--- Row {idx} ---")
    for col in df.columns:
        val = row[col]
        if pd.notna(val):
            print(f"  {col:20s}: {val}")

    # Check if paths exist
    print("\n  PATH CHECKS:")
    for col in ['locals_npz', 'global_vec', 'frames_dir']:
        if col in df.columns and pd.notna(row[col]):
            # Try relative to parent_dir
            path1 = Path(parent_dir) / row[col]
            # Try as absolute path
            path2 = Path(row[col])

            print(f"    {col}:")
            print(f"      Relative: {path1} -> {'EXISTS' if path1.exists() else 'NOT FOUND'}")
            if path1.exists() and path1.is_dir():
                files = list(path1.glob('*'))[:5]
                print(f"        Contains: {[f.name for f in files]}")
