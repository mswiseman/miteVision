import os
import cv2
import numpy as np

'''
This script converts and masks to polygons and then to YOLO format.
'''

# Define a mapping of colors to class IDs
COLOR_TO_CLASS = {
    (0, 0, 255): 0,    # Red -> Class 0
    (0, 255, 0): 1,    # Green -> Class 1
    (255, 0, 0): 2,    # Blue -> Class 2
    (0, 255, 255): 3,  # Yellow -> Class 3
    (255, 0, 255): 4,  # Magenta -> Class 4
}

# Minimum object size (in pixels) for QC
MIN_OBJECT_SIZE = 50


def convert_mask_to_polygons(mask):
    polygons = []
    unique_colors = np.unique(mask.reshape(-1, mask.shape[2]), axis=0)
    for color in unique_colors:
        if np.all(color == [0, 0, 0]):  # Skip the background
            continue

        # Get the class ID for this color
        color_tuple = tuple(color[:3]) # Convert to tuple for hashing
        class_id = COLOR_TO_CLASS.get(color_tuple) # Get the class ID
        if class_id is None:
            # print(f"Warning: Unmapped color {color_tuple} found in mask.")
            continue

        # Create a binary mask for the current color
        mask_label = cv2.inRange(mask, color, color)
        contours, _ = cv2.findContours(mask_label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # Filter out small objects based on area
            area = cv2.contourArea(contour)
            if area < MIN_OBJECT_SIZE:
                continue  # Skip this contour if it's too small

            if len(contour) > 2:
                polygon = contour.flatten().tolist()
                polygons.append((class_id, polygon))
    return polygons


def create_yolo_annotation(class_id, width, height, polygon):
    normalized_polygon = []
    for i in range(0, len(polygon), 2):
        x = polygon[i] / width
        y = polygon[i + 1] / height
        normalized_polygon.extend([x, y])

    # Ensure all coordinates are within bounds [0, 1]
    if any(coord < 0 or coord > 1 for coord in normalized_polygon):
        return None

    # Check if the polygon is valid (at least 3 points)
    if len(normalized_polygon) < 6:
        return None

    poly_str = " ".join(map(str, normalized_polygon))
    return f"{class_id} {poly_str}"


def process_combined_masks(images_dir, masks_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image_files = os.listdir(images_dir)

    for image_file in image_files:
        if not image_file.endswith(('.png', '.jpg', '.jpeg')):
            continue

        image_path = os.path.join(images_dir, image_file)

        # Match the mask filename by replacing the image extension with `.png`
        mask_file = f"{os.path.splitext(image_file)[0]}_combined.png"
        mask_path = os.path.join(masks_dir, mask_file)

        if not os.path.exists(mask_path):
            #print(f"Warning: No corresponding mask found for {image_file}")
            continue

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path)
        height, width, _ = image.shape

        polygons = convert_mask_to_polygons(mask)
        annotations = []
        for class_id, polygon in polygons:
            annotation = create_yolo_annotation(class_id, width, height, polygon)
            if annotation:
                annotations.append(annotation)

        annotation_file = os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}.txt")
        with open(annotation_file, 'w') as file:
            file.write("\n".join(annotations))


# Example usage
images_dir = './Output'
masks_dir = './Masks'  # Path to combined masks
output_dir = './Segmentation_Annotations'
process_combined_masks(images_dir, masks_dir, output_dir)

print("Converting combined masks to YOLO format completed.")
