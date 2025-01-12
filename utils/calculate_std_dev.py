import os
import argparse
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging

# example usage: python calculate_std_dev.py /path/to/images

def is_image_file(filename):
    """
    Check if a file is an image based on its extension.
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    ext = os.path.splitext(filename)[1].lower()
    return ext in image_extensions


def setup_logging():
    """
    Setup logging configuration.
    """
    logging.basicConfig(
        filename='rgb_stats.log',
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def calculate_mean_std(directory, threshold=0):
    """
    Calculate the average and standard deviation of RGB values for each subdirectory in a directory,
    excluding black padding (pixels where R=G=B=0).

    Args:
        directory (str): Path to the root directory containing category subdirectories.
        threshold (int): Threshold to consider a pixel as non-padding. Pixels with all RGB <= threshold are treated as padding.

    Returns:
        pd.DataFrame: DataFrame containing mean and std RGB values for each category.
    """
    # Initialize a list to store statistics for each category
    stats_list = []

    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

    # List all immediate subdirectories (categories)
    categories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

    if not categories:
        logging.warning(
            f"No subdirectories found in '{directory}'. Ensure that categories are organized in subdirectories.")
        return pd.DataFrame()

    # Iterate through each category
    for category in categories:
        category_path = os.path.join(directory, category)
        sum_rgb = np.zeros(3, dtype=np.float64)
        sum_sq_rgb = np.zeros(3, dtype=np.float64)
        count = 0
        num_images = 0

        logging.info(f"Processing category: '{category}'")

        # Get list of all image files in the category
        image_files = []
        for root, dirs, files in os.walk(category_path):
            for filename in files:
                if is_image_file(filename):
                    image_files.append(os.path.join(root, filename))

        # Use tqdm for progress bar
        for file_path in tqdm(image_files, desc=f"  {category}", unit="image"):
            try:
                with Image.open(file_path) as img:
                    img = img.convert('RGB')  # Ensure image is in RGB mode
                    img_array = np.array(img)

                    # Create a mask for non-padding pixels based on the threshold
                    non_padding_mask = np.any(img_array > threshold, axis=2)

                    # Extract non-padding pixels
                    non_padding_pixels = img_array[non_padding_mask]

                    if non_padding_pixels.size == 0:
                        logging.warning(f"    No non-padding pixels found in '{file_path}'. Skipping.")
                        continue

                    # Update sums
                    sum_rgb += non_padding_pixels.sum(axis=0)
                    sum_sq_rgb += (non_padding_pixels ** 2).sum(axis=0)
                    count += non_padding_pixels.shape[0]
                    num_images += 1

            except Exception as e:
                logging.error(f"  Error processing '{file_path}': {e}")
                continue

        if count == 0:
            logging.warning(f"  No valid non-padding pixels found in category '{category}'.")
            mean_rgb = [0, 0, 0]
            std_rgb = [0, 0, 0]
        else:
            # Calculate mean
            mean_rgb = sum_rgb / count

            # Calculate standard deviation
            # std = sqrt(E[X^2] - (E[X])^2)
            variance = (sum_sq_rgb / count) - (mean_rgb ** 2)
            variance = np.maximum(variance, 0)
            std_rgb = np.sqrt(variance)

        # Append the statistics to the list
        stats_list.append({
            'Category': category,
            'Num Images': num_images,
            'Num Pixels': count,
            'Mean_R': mean_rgb[0],
            'Mean_G': mean_rgb[1],
            'Mean_B': mean_rgb[2],
            'Std_R': std_rgb[0],
            'Std_G': std_rgb[1],
            'Std_B': std_rgb[2]
        })

        logging.info(f"  Processed {num_images} images with {count} non-padding pixels.")
        logging.info(f"  Mean RGB: R={mean_rgb[0]:.2f}, G={mean_rgb[1]:.2f}, B={mean_rgb[2]:.2f}")
        logging.info(f"  Std RGB : R={std_rgb[0]:.2f}, G={std_rgb[1]:.2f}, B={std_rgb[2]:.2f}")

    # Create a DataFrame from the statistics list
    stats_df = pd.DataFrame(stats_list, columns=[
        'Category', 'Num Images', 'Num Pixels',
        'Mean_R', 'Mean_G', 'Mean_B',
        'Std_R', 'Std_G', 'Std_B'
    ])

    return stats_df


def main():
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Calculate average and std RGB values for each category in nested directories, excluding black padding.")
    parser.add_argument("directory", type=str, help="Path to the root directory containing category subdirectories.")
    parser.add_argument("--threshold", type=int, default=0,
                        help="Threshold to consider a pixel as non-padding (default: 0). Pixels with all RGB <= threshold are treated as padding.")
    parser.add_argument("--output", type=str, default=None, help="Path to save the results as a CSV file (optional).")
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        logging.error(f"The directory '{args.directory}' does not exist or is not accessible.")
        return

    logging.info(f"Starting RGB statistics calculation for directory: '{args.directory}'")
    logging.info(f"Excluding pixels with all RGB <= {args.threshold} as padding.")

    stats_df = calculate_mean_std(args.directory, threshold=args.threshold)

    if stats_df.empty:
        logging.info("No statistics to display.")
        return

    # Display the statistics
    print("\n--- Per-Category RGB Statistics (Excluding Padding) ---")
    print(stats_df.to_string(index=False))

    # Save to CSV if requested
    if args.output:
        try:
            stats_df.to_csv(args.output, index=False)
            logging.info(f"Statistics saved to '{args.output}'.")
        except Exception as e:
            logging.error(f"Error saving to '{args.output}': {e}")


if __name__ == "__main__":
    main()
