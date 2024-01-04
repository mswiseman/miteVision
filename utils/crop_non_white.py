from PIL import Image
import numpy as np
import os


def crop_image(image_path, output_path):
    # Load the image
    with Image.open(image_path) as img:
        # Ensure the image has an alpha channel
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        # Convert image to numpy array
        np_img = np.array(img)

        # Find all pixels that are not fully transparent (alpha channel is not 0)
        non_transparent_pixels = np_img[:, :, 3] != 0

        # Get the indices of non-transparent pixels
        rows, cols = np.where(non_transparent_pixels)

        # Calculate the bounding box
        if rows.size > 0 and cols.size > 0:  # only if there are non-transparent pixels
            min_row, max_row = rows.min(), rows.max()
            min_col, max_col = cols.min(), cols.max()

            # Crop the image to the bounding box
            cropped_img = img.crop((min_col, min_row, max_col + 1, max_row + 1))
        else:
            # If no non-transparent pixels are found, use the whole image
            cropped_img = img

        # Save the cropped image
        cropped_img.save(output_path)


# Example usage
input_directory = '/Users/michelewiseman/OpenImages/Training/mites'
output_directory = '/Users/michelewiseman/OpenImages/Training/cropped'

# Create output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Process each image in the directory
for filename in os.listdir(input_directory):
    if filename.endswith('.png'):  # or any other image format
        input_path = os.path.join(input_directory, filename)
        output_path = os.path.join(output_directory, f"cropped_{filename}")
        crop_image(input_path, output_path)
