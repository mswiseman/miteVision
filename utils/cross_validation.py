#!/usr/bin/env python3
import os
import re
import shutil
import random
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.model_selection import KFold
import argparse
import sys

def parse_args():
    p = argparse.ArgumentParser(
        description="Create K-fold cross-validation splits for a YOLO dataset, "
                    "including synthetic images in TRAIN ONLY."
    )
    p.add_argument("--dataset_dir", type=Path, required=True,
                   help="Root dataset directory containing 'images/' and 'labels/' subfolders.")
    p.add_argument("--output_dir", type=Path, required=True,
                   help="Output directory where fold folders will be created.")
    p.add_argument("--num_folds", type=int, default=5,
                   help="Number of CV folds (default: 5).")
    p.add_argument("--image_ext", type=str, default="jpg",
                   help="Image file extension without dot (default: jpg).")
    p.add_argument("--synthetic_regex", type=str, default=r"^(image_\d+|\d+)_",
                   help="Regex to identify synthetic images from filename stem (default: '^(image_\\d+|\\d+)_').")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility (default: 42).")
    return p.parse_args()

def is_synthetic(path: Path, pattern: re.Pattern) -> bool:
    return bool(pattern.match(path.stem))

def count_classes(label_files):
    class_counter = Counter()
    for lbl_path in label_files:
        try:
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls_id = int(parts[0])
                        class_counter[cls_id] += 1
        except FileNotFoundError:
            # Skip if label is missing (shouldn't happen because we filtered earlier)
            continue
    return dict(class_counter)

def main():
    args = parse_args()
    random.seed(args.seed)

    images_dir = args.dataset_dir / "images"
    labels_dir = args.dataset_dir / "labels"
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not images_dir.is_dir() or not labels_dir.is_dir():
        print(f"ERROR: '{images_dir}' or '{labels_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Compile synthetic filename pattern
    synth_pat = re.compile(args.synthetic_regex)

    # Gather image files (case-insensitive extension match)
    ext = args.image_ext.lower().lstrip(".")
    image_paths = [p for p in images_dir.iterdir()
                   if p.is_file() and p.suffix.lower() == f".{ext}"]

    # Keep only those with a corresponding label file
    image_paths = [p for p in sorted(image_paths)
                   if (labels_dir / f"{p.stem}.txt").exists()]

    if not image_paths:
        print(f"ERROR: Found 0 valid image/label pairs in {images_dir} and {labels_dir}.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(image_paths)} valid image-label pairs.")

    # Split synthetic vs real by filename stem
    synthetic_imgs = [p for p in image_paths if is_synthetic(p, synth_pat)]
    real_imgs = [p for p in image_paths if not is_synthetic(p, synth_pat)]
    print(f"Identified {len(synthetic_imgs)} synthetic images (used only in training).")
    print(f"Real images available for k-fold split: {len(real_imgs)}")

    if len(real_imgs) < args.num_folds:
        print(f"WARNING: num_folds ({args.num_folds}) > real images ({len(real_imgs)}). "
              f"KFold will create some folds with very few samples.", file=sys.stderr)

    kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(real_imgs)):
        print(f"\n=== Processing fold {fold_idx} ===")

        fold_dir = output_dir / f"fold{fold_idx}"
        for sub in ["train/images", "train/labels", "val/images", "val/labels"]:
            (fold_dir / sub).mkdir(parents=True, exist_ok=True)

        split_label_paths = defaultdict(list)

        # Train = real training images + ALL synthetic images
        train_imgs = [real_imgs[i] for i in train_idx] + synthetic_imgs
        val_imgs = [real_imgs[i] for i in val_idx]

        for split, image_list in [("train", train_imgs), ("val", val_imgs)]:
            for img_path in image_list:
                lbl_path = labels_dir / f"{img_path.stem}.txt"
                try:
                    shutil.copyfile(img_path, fold_dir / f"{split}/images" / img_path.name)
                    shutil.copyfile(lbl_path, fold_dir / f"{split}/labels" / lbl_path.name)
                    split_label_paths[split].append(lbl_path)
                except Exception as e:
                    print(f"Error copying {img_path} or {lbl_path}: {e}", file=sys.stderr)

        print(f"Fold {fold_idx} created: {len(train_imgs)} train (incl. synthetic), {len(val_imgs)} val examples.")

        # Class distributions
        for split in ["train", "val"]:
            counts = count_classes(split_label_paths[split])
            print(f"Class distribution in {split}: {dict(sorted(counts.items()))}")

    print(f"\n✅ Done creating {args.num_folds} CV folds at: {output_dir}")
    print("   Synthetic data was included only in TRAIN splits.")

if __name__ == "__main__":
    main()
