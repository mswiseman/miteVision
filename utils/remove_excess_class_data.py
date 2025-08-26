import os
import argparse
from pathlib import Path

def count_class_instances(label_file: Path, class_id: int) -> int:
    """Count how many times class_id appears in a YOLO label file."""
    count = 0
    try:
        with label_file.open('r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    if int(parts[0]) == class_id:
                        count += 1
                except ValueError:
                    # skip malformed lines
                    continue
    except Exception as e:
        print(f"Error reading {label_file}: {e}")
    return count

def process(dataset_path: Path,
            target_class_id: int,
            max_allowed_instances: int,
            limit_class_id: int,
            limit_min_count: int,
            image_ext: str,
            dry_run: bool) -> int:
    """
    Remove (or report) image/label pairs where:
      count(target_class_id) > max_allowed_instances AND
      count(limit_class_id)   < limit_min_count
    """
    images_path = dataset_path / "images"
    labels_path = dataset_path / "labels"

    if not images_path.is_dir() or not labels_path.is_dir():
        raise FileNotFoundError(f"Expected {images_path} and {labels_path} to exist.")

    removed = 0
    image_ext = image_ext.lstrip(".").lower()

    for label_name in os.listdir(labels_path):
        if not label_name.endswith(".txt"):
            continue

        label_file = labels_path / label_name
        tgt_count = count_class_instances(label_file, target_class_id)
        lim_count = count_class_instances(label_file, limit_class_id)

        if tgt_count > max_allowed_instances and lim_count < limit_min_count:
            img_name = label_name.replace(".txt", f".{image_ext}")
            img_file = images_path / img_name

            action = "Would remove" if dry_run else "Removed"
            print(f"{action}: {img_file} and {label_file} "
                  f"(target #{target_class_id}={tgt_count} > {max_allowed_instances}, "
                  f"limit #{limit_class_id}={lim_count} < {limit_min_count})")

            if not dry_run:
                try:
                    if img_file.exists():
                        img_file.unlink()
                except Exception as e:
                    print(f"  Error removing image {img_file}: {e}")
                try:
                    label_file.unlink()
                    removed += 1
                except Exception as e:
                    print(f"  Error removing label {label_file}: {e}")

    return removed

def parse_args():
    p = argparse.ArgumentParser(
        description=("Filter a YOLO dataset by removing image/label pairs where the target class "
                     "appears too many times unless enough instances of a limiting class are present.")
    )
    p.add_argument("--dataset_path", type=Path, required=True,
                   help="Path to the dataset root containing 'images/' and 'labels/'.")
    p.add_argument("--target_class_id", type=int, required=True,
                   help="Class ID to check for overabundance (e.g., 6).")
    p.add_argument("--max_allowed_instances", type=int, default=5,
                   help="Maximum allowed instances of the target class (default: 5).")
    p.add_argument("--limit_class_id", type=int, required=True,
                   help="Class ID that limits removals (e.g., 0).")
    p.add_argument("--limit_min_count", type=int, default=2,
                   help="Minimum required instances of the limiting class (default: 2).")
    p.add_argument("--image_ext", type=str, default="jpg",
                   help="Image extension (without dot) to match label stems (default: jpg).")
    p.add_argument("--dry_run", action="store_true",
                   help="Print what would be removed, but do not delete files.")
    return p.parse_args()

def main():
    args = parse_args()
    removed = process(
        dataset_path=args.dataset_path,
        target_class_id=args.target_class_id,
        max_allowed_instances=args.max_allowed_instances,
        limit_class_id=args.limit_class_id,
        limit_min_count=args.limit_min_count,
        image_ext=args.image_ext,
        dry_run=args.dry_run
    )
    if args.dry_run:
        print("\nDry run complete. No files were deleted.")
    else:
        print(f"\nCompleted. Removed {removed} image/label pair(s).")

if __name__ == "__main__":
    main()
