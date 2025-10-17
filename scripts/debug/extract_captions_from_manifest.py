#!/usr/bin/env python3
"""
Extract captions from existing manifest to create captions CSV for reprocessing.
"""
import pandas as pd
import sys
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_captions_from_manifest.py <manifest_path> [output_csv]")
        print()
        print("Example:")
        print("  python extract_captions_from_manifest.py \\")
        print("    /home/kgarci18/data_lisik3/kgarci18/ooo/train/social_tokens/scene_packs_manifest_recovered.csv \\")
        print("    /home/kgarci18/data_lisik3/kgarci18/ooo/train/captions.csv")
        sys.exit(1)

    manifest_path = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "captions.csv"

    if not Path(manifest_path).exists():
        print(f"ERROR: Manifest not found: {manifest_path}")
        sys.exit(1)

    print(f"Reading manifest: {manifest_path}")
    df = pd.read_csv(manifest_path)

    # Check required columns
    if 'clip_id' not in df.columns:
        print(f"ERROR: 'clip_id' column not found in manifest")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    # Find caption column
    caption_col = None
    for col in ['transcript_json', 'caption', 'text']:
        if col in df.columns:
            caption_col = col
            break

    if caption_col is None:
        print(f"ERROR: No caption column found")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    print(f"Using caption column: '{caption_col}'")

    # Filter out failed rows
    if 'status' in df.columns:
        before = len(df)
        df = df[df['status'] == 'ok']
        print(f"Filtered to {len(df)} successful entries (was {before})")

    # Create output dataframe
    output_df = pd.DataFrame({
        'video_name': df['clip_id'].apply(lambda x: f"{x}.mp4"),  # Add .mp4 extension
        'caption': df[caption_col]
    })

    # Remove any rows with missing captions
    before = len(output_df)
    output_df = output_df.dropna(subset=['caption'])
    if len(output_df) < before:
        print(f"Removed {before - len(output_df)} rows with missing captions")

    # Save
    output_df.to_csv(output_csv, index=False)

    print(f"\n✅ Created captions CSV: {output_csv}")
    print(f"   Total entries: {len(output_df)}")
    print(f"\nFirst 3 entries:")
    print(output_df.head(3).to_string(index=False))

    print(f"\nYou can now run reprocessing with:")
    print(f"  --captions_csv {output_csv}")

if __name__ == "__main__":
    main()
