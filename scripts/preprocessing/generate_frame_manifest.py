import csv
import re
import argparse
from pathlib import Path
import natsort

def generate_manifest_for_existing(
    local_dir: str,
    label: str,
    split: str,
    fps: float = 1.0,
    manifest_path: str | None = None,
    incremental: bool = True,
) -> tuple[str, int]:
    local_dir = Path(local_dir).expanduser().resolve()
    base = local_dir / "preprocess" / label / split
    print(f"[INFO] Looking under: {base}")

    if manifest_path is None:
        manifest_path = base / "manifest.csv"
    else:
        manifest_path = Path(manifest_path).expanduser()

    base.mkdir(parents=True, exist_ok=True)

    existing = set()
    if incremental and manifest_path.exists():
        with open(manifest_path, "r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                existing.add(row["frame_path"])
        print(f"[INFO] Found existing manifest with {len(existing)} frame paths.")

    write_header = not manifest_path.exists() or not incremental
    mode = "a" if (incremental and manifest_path.exists()) else "w"

    num_written = 0
    pattern = re.compile(r"frame_(\d+)\.jpg$", re.IGNORECASE)

    with open(manifest_path, mode, newline="") as fcsv:
        w = csv.writer(fcsv)
        if write_header:
            w.writerow(["split","clip_id","frame_path","t_sec"])

        frames_dirs = [p for p in base.glob("*/frames") if p.is_dir()]
        print(f"[INFO] Found {len(frames_dirs)} 'frames' directories to scan.")

        for frames_dir in natsort.natsorted(frames_dirs):
            clip_id = frames_dir.parent.name
            frames = natsort.natsorted(frames_dir.glob("*.jpg"))
            print(f"[SCAN] Clip: {clip_id} → {len(frames)} frames found.")

            for fp in frames:
                fp_str = str(fp)
                if incremental and fp_str in existing:
                    continue

                m = pattern.search(fp.name)
                idx = int(m.group(1)) if m else 0
                t_sec = (idx / fps) if fps and fps > 0 else 0.0

                w.writerow([split, clip_id, fp_str, f"{t_sec:.3f}"])
                num_written += 1

    print(f"✓ Wrote {num_written} rows → {manifest_path}")
    return str(manifest_path), num_written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a manifest for existing extracted frames.")
    parser.add_argument("--local_dir", required=True, help="Path to the local dataset directory.")
    parser.add_argument("--label", required=True, help="Dataset label (e.g., 'naturalistic').")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"], help="Dataset split.")
    parser.add_argument("--fps", type=float, default=1.0, help="FPS used during extraction (default: 1.0).")
    parser.add_argument("--manifest_path", type=str, default=None, help="Optional path to save manifest.")
    parser.add_argument("--no_incremental", action="store_true", help="Disable incremental mode (overwrite).")

    args = parser.parse_args()

    manifest_path, added = generate_manifest_for_existing(
        local_dir=args.local_dir,
        label=args.label,
        split=args.split,
        fps=args.fps,
        manifest_path=args.manifest_path,
        incremental=not args.no_incremental
    )

    print(manifest_path, added)

