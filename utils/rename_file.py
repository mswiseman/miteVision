import os

# Directory setup
IMAGES_DIR = "/Users/michelewiseman/Downloads/immature/train/images"
LABELS_DIR = "/Users/michelewiseman/Downloads/immature/train/labels"

# Class ID to name mapping
class_names = {
    0: "Immature"
    #0: "Adult_female",
    #1: "Adult_male"
    #2: "Dead_mite",
    #3: "Immature"
}

# Keep counters for each class so we can increment filenames:
#   Adult_female_1.png, Adult_female_2.png, etc.
counters = {name: 1 for name in class_names.values()}


def main():
    # List all label files in LABELS_DIR
    label_files = [f for f in os.listdir(LABELS_DIR) if f.endswith(".txt")]

    for label_file in label_files:
        label_path_old = os.path.join(LABELS_DIR, label_file)

        # Extract base name (e.g., "IMG_0001" from "IMG_0001.txt")
        base_name = os.path.splitext(label_file)[0]

        # The matching image is assumed to have the same base name + ".png"
        # Adjust the extension here if your images are ".jpg" or something else
        image_file_old = base_name + ".jpg"
        image_path_old = os.path.join(IMAGES_DIR, image_file_old)

        # Read the first line of the label file to get class ID
        with open(label_path_old, "r") as f:
            lines = f.readlines()
        if not lines:
            print(f"Warning: {label_file} is empty. Skipping.")
            continue

        # Parse class ID from the first token (e.g. "0 0.5 0.5 0.1 0.1")
        first_line = lines[0].strip().split()
        class_id = int(first_line[0])

        # Map class ID to class name
        if class_id not in class_names:
            print(f"Warning: Class ID {class_id} not in mapping. Skipping {label_file}.")
            continue
        class_name = class_names[class_id]

        # Construct new filenames
        new_file_index = counters[class_name]  # e.g., if "Adult_female" is at 2, this is 2
        counters[class_name] += 1  # increment for the next file

        # e.g. "Adult_female_2.png" and "Adult_female_2.txt"
        image_file_new = f"{class_name}_{new_file_index}.png"
        label_file_new = f"{class_name}_{new_file_index}.txt"

        image_path_new = os.path.join(IMAGES_DIR, image_file_new)
        label_path_new = os.path.join(LABELS_DIR, label_file_new)

        # Rename the image (if it exists)
        if os.path.exists(image_path_old):
            os.rename(image_path_old, image_path_new)
            print(f"Renamed image: {image_file_old} -> {image_file_new}")
        else:
            print(f"Image file does not exist: {image_file_old}")

        # Rename the label
        os.rename(label_path_old, label_path_new)
        print(f"Renamed label: {label_file} -> {label_file_new}")


if __name__ == "__main__":
    main()
