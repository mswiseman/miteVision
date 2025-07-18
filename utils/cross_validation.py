import os
import shutil
import random
from pathlib import Path
from sklearn.model_selection import KFold
from collections import Counter, defaultdict
import re

# Set seed for reproducibility
random.seed(42)

# === CONFIGURATION ===
DATASET_DIR = Path("/Users/michelewiseman/Downloads/TSSM-Detection-v2-209/train")
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"
OUTPUT_DIR = Path("/Users/michelewiseman/Downloads/TSSM-Detection-v2-209/cv_folds")

NUM_FOLDS = 5

# === IDENTIFY SYNTHETIC ===
SYNTHETIC_PATTERN = re.compile(r"^(image_\d+|\d+)_")

def is_synthetic(path):
    return bool(SYNTHETIC_PATTERN.match(path.stem))

# === GATHER IMAGE FILES ===
image_paths = sorted([p for p in IMAGES_DIR.glob("*.jpg")])
image_paths = [p for p in image_paths if (LABELS_DIR / (p.stem + ".txt")).exists()]
print(f"Found {len(image_paths)} valid image-label pairs.")

synthetic_imgs = [p for p in image_paths if is_synthetic(p)]
real_imgs = [p for p in image_paths if not is_synthetic(p)]
print(f"Identified {len(synthetic_imgs)} synthetic images (used only in training).")

# === K-FOLD SPLIT ON REAL IMAGES ONLY ===
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

def count_classes(label_files):
    class_counter = Counter()
    for lbl_path in label_files:
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    class_counter[cls_id] += 1
    return dict(class_counter)

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(real_imgs)):
    print(f"\n=== Processing fold {fold_idx} ===")

    fold_dir = OUTPUT_DIR / f"fold{fold_idx}"
    for subdir in ["train/images", "train/labels", "val/images", "val/labels"]:
        os.makedirs(fold_dir / subdir, exist_ok=True)

    split_label_paths = defaultdict(list)

    # Train = real training images + synthetic images
    train_imgs = [real_imgs[i] for i in train_idx] + synthetic_imgs
    val_imgs = [real_imgs[i] for i in val_idx]

    for split, image_list in [("train", train_imgs), ("val", val_imgs)]:
        for img_path in image_list:
            label_path = LABELS_DIR / (img_path.stem + ".txt")

            shutil.copyfile(img_path, fold_dir / f"{split}/images" / img_path.name)
            shutil.copyfile(label_path, fold_dir / f"{split}/labels" / label_path.name)

            split_label_paths[split].append(label_path)

    print(f"Fold {fold_idx} created: {len(train_imgs)} train (incl. synthetic), {len(val_imgs)} val examples.")

    # Print class distributions
    for split in ["train", "val"]:
        counts = count_classes(split_label_paths[split])
        print(f"Class distribution in {split}: {dict(sorted(counts.items()))}")

print("\n✅ Done creating 5 CV folds. Synthetic data used only in training.")
