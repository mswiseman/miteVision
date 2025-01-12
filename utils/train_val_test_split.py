import os
import shutil
import random
import argparse
from pathlib import Path


def is_image_file(filename):
    """
    Check if a file is an image based on its extension.
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    return filename.suffix.lower() in image_extensions


def create_directory(path):
    """
    Create a directory if it doesn't exist.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory {path}: {e}")
        exit(1)


def split_dataset(source_dir, train_dir, val_dir, test_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Split the dataset into train, validation, and test sets and copy the files accordingly.

    Args:
        source_dir (Path): Path to the source directory containing images.
        train_dir (Path): Path to the training directory.
        val_dir (Path): Path to the validation directory.
        test_dir (Path): Path to the testing directory.
        train_ratio (float): Proportion of images to include in the training set.
        val_ratio (float): Proportion of images to include in the validation set.
        test_ratio (float): Proportion of images to include in the testing set.
        seed (int): Random seed for reproducibility.
    """
    # Ensure the ratios sum to 1
    total = train_ratio + val_ratio + test_ratio
    if not abs(total - 1.0) < 1e-6:
        print("Error: Train, validation, and test ratios must sum to 1.")
        exit(1)

    # Gather all image files
    all_images = [f for f in source_dir.iterdir() if f.is_file() and is_image_file(f)]
    total_images = len(all_images)

    if total_images == 0:
        print(f"No image files found in {source_dir}.")
        exit(1)

    print(f"Found {total_images} image files in {source_dir}.")

    # Shuffle the images
    random.seed(seed)
    random.shuffle(all_images)

    # Calculate split indices
    train_end = int(train_ratio * total_images)
    val_end = train_end + int(val_ratio * total_images)

    train_images = all_images[:train_end]
    val_images = all_images[train_end:val_end]
    test_images = all_images[val_end:]

    # Function to copy files
    def copy_files(file_list, destination):
        for file_path in file_list:
            try:
                shutil.copy2(file_path, destination)
            except Exception as e:
                print(f"Error copying {file_path} to {destination}: {e}")

    # Copy the files
    print(f"\nCopying {len(train_images)} images to {train_dir}")
    copy_files(train_images, train_dir)

    print(f"Copying {len(val_images)} images to {val_dir}")
    copy_files(val_images, val_dir)

    print(f"Copying {len(test_images)} images to {test_dir}")
    copy_files(test_images, test_dir)

    print("\nDataset splitting completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Split a directory of images into train, val, and test sets.")
    parser.add_argument('--source', type=str, default='/Users/michelewiseman/PycharmProjects/miteModeling/images/adult_males',
                        help='Path to the source directory containing images.')
    parser.add_argument('--train', type=str,
                        default='/Users/michelewiseman/PycharmProjects/miteModeling/dataset_alive_dead/train/alive',
                        help='Path to the training directory.')
    parser.add_argument('--val', type=str,
                        default='/Users/michelewiseman/PycharmProjects/miteModeling/dataset_alive_dead/val/alive',
                        help='Path to the validation directory.')
    parser.add_argument('--test', type=str,
                        default='/Users/michelewiseman/PycharmProjects/miteModeling/dataset_alive_dead/test/alive',
                        help='Path to the testing directory.')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Proportion of images to include in the training set.')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Proportion of images to include in the validation set.')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='Proportion of images to include in the testing set.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility.')

    args = parser.parse_args()

    source_dir = Path(args.source)
    train_dir = Path(args.train)
    val_dir = Path(args.val)
    test_dir = Path(args.test)

    # Create destination directories if they don't exist
    create_directory(train_dir)
    create_directory(val_dir)
    create_directory(test_dir)

    # Perform the split
    split_dataset(source_dir, train_dir, val_dir, test_dir,
                  train_ratio=args.train_ratio,
                  val_ratio=args.val_ratio,
                  test_ratio=args.test_ratio,
                  seed=args.seed)


if __name__ == "__main__":
    main()
