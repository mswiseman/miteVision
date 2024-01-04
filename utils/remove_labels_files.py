import os
from collections import defaultdict
import shutil
import random


# Set dataset paths and the target class ID
dataset_path = '/Users/michelewiseman/Downloads/TSSM Detection v2.v44-2023-12-11-2-15pm-no-augmentations.yolov8/valid/undersampled'
images_path = os.path.join(dataset_path, 'images')
labels_path = os.path.join(dataset_path, 'labels')
target_class_id = 5  # replace with the ID of the class you want to remove

# Function to check if the target class ID is in the label file
def get_labels(labels_path):
    labels = defaultdict(list)
    for label_file in os.listdir(labels_path):
        with open(os.path.join(labels_path, label_file), 'r') as file:
            for line in file:
                class_id = int(line.split()[0])
                labels[class_id].append(label_file)
    return labels

def contains_target_class(label_file):
    with open(label_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            class_id = int(line.split()[0])
            if class_id == target_class_id:
                return True
    return False


def print_label_counts(labels, message):
    print(message)
    for class_id, files in labels.items():
        print(f"Class {class_id}: {len(files)} instances")
    print()

labels = get_labels(labels_path)
# Print initial label counts
print_label_counts(labels, "Label counts:")

# Iterate over the label files and remove files if needed
for label_file in os.listdir(labels_path):
    label_file_path = os.path.join(labels_path, label_file)
    if contains_target_class(label_file_path):
        os.remove(label_file_path)  # Remove label file
        print(f"Removed {label_file_path} due to containing the target class ID ({target_class_id}).")

        # Construct the corresponding image file name and remove
        image_file_name = label_file.replace('.txt', '.jpg')  # change file extension as per your dataset
        image_file_path = os.path.join(images_path, image_file_name)
        if os.path.exists(image_file_path):
            os.remove(image_file_path)
            print(f"Removed {image_file_path} due to containing the target class ID ({target_class_id}).")

print("Completed removing images and labels containing the specified class.")
