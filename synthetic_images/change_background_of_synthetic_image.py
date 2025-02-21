import os
import cv2
import numpy as np
import random

"""

This script processes synthetic images by replacing the background with a new background image. It also renames the 
label files to match the new image names. This script was designed to test if background of synthetic image affects the 
model's performance on our other host test set. Before running, define the folder paths for the synthetic images, 
labels, and new background images.

February 21 2025 version 1
Michele Wiseman
"""



# Set random seed for reproducibility.
random.seed(42)

# Define folder paths.
images_dir = "./background_swap/old_images"                        # Folder with synthetic images to process.
labels_dir = "./background_swap/old_labels"                        # Folder with corresponding annotation labels.
backgrounds_dir = "./background_swap/replacement_backgrounds"  # Folder with new background images.
output_images_dir = "./background_swap/new_images"             # Output folder for processed images.
output_labels_dir = "./background_swap/new_labels"             # Output folder for renamed label files.

# Create output directories if they do not exist.
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)

# Get list of image files (modify the extensions if needed).
image_files = [f for f in os.listdir(images_dir)
               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Get list of background files.
background_files = [f for f in os.listdir(backgrounds_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if not background_files:
    raise ValueError("No background images found in the replacement_background folder.")

for image_file in image_files:
    # Construct full path for the image.
    image_path = os.path.join(images_dir, image_file)
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"Failed to load image {image_path}. Skipping.")
        continue

    # Select a random background image and load it.
    bg_file = random.choice(background_files)
    bg_path = os.path.join(backgrounds_dir, bg_file)
    new_bg_img = cv2.imread(bg_path)
    if new_bg_img is None:
        print(f"Failed to load background {bg_path}. Skipping.")
        continue

    # Get dimensions from the original image.
    img_height, img_width = original_img.shape[:2]
    # Resize background to match the original image dimensions.
    new_bg_img = cv2.resize(new_bg_img, (img_width, img_height))

    # Determine the corresponding label file.
    base_name, _ = os.path.splitext(image_file)
    label_file = base_name + ".txt"
    label_path = os.path.join(labels_dir, label_file)
    if not os.path.exists(label_path):
        print(f"Label file {label_path} not found for image {image_file}. Skipping.")
        continue

    # Parse the label file to extract polygons.
    polygons = []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Parse numbers; if odd, assume the first is a label.
            values = list(map(float, line.split()))
            if len(values) % 2 == 1:
                coords = values[1:]
            else:
                coords = values
            polygon = []
            for i in range(0, len(coords), 2):
                x_norm = coords[i]
                y_norm = coords[i + 1]
                # Convert normalized coordinates to absolute pixel positions.
                x = int(x_norm * img_width)
                y = int(y_norm * img_height)
                polygon.append((x, y))
            polygons.append(polygon)

    # Create a binary mask for the segmentation.
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    for poly in polygons:
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)

    # Extract the segmented objects.
    object_roi = cv2.bitwise_and(original_img, original_img, mask=mask)
    # Optionally remove the objects from the original image.
    inverted_mask = cv2.bitwise_not(mask)
    original_without_object = cv2.bitwise_and(original_img, original_img, mask=inverted_mask)

    # Composite the segmented objects onto the new background.
    new_bg_with_object = new_bg_img.copy()
    new_bg_with_object[mask == 255] = object_roi[mask == 255]

    # Save the processed image in the new_images folder.
    output_image_path = os.path.join(output_images_dir, image_file)
    cv2.imwrite(output_image_path, new_bg_with_object)

    # Copy (or rename) the label file to the new_labels folder.
    # If you need to update any content inside the label file to reflect the new image name, modify here.
    output_label_path = os.path.join(output_labels_dir, label_file)
    with open(label_path, "r") as infile, open(output_label_path, "w") as outfile:
        outfile.write(infile.read())

    print(f"Processed {image_file} -> {output_image_path}")
