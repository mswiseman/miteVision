import os
import random
from collections import defaultdict
import shutil

# Path to your dataset
images_path = '/Users/michelewiseman/Downloads/TSSM-Detection-v2-35/train/images'
labels_path = '/Users/michelewiseman/Downloads/TSSM-Detection-v2-35/train/labels'


# Path for undersampled dataset
undersampled_images_path = '/Users/michelewiseman/Downloads/TSSM-Detection-v2-35/train/undersampled/images'
undersampled_labels_path = '/Users/michelewiseman/Downloads/TSSM-Detection-v2-35/train/undersampled/labels'


# Create directories for the undersampled dataset if they don't exist
os.makedirs(undersampled_images_path, exist_ok=True)
os.makedirs(undersampled_labels_path, exist_ok=True)

# Function to get all labels
def get_labels(labels_path):
    labels = defaultdict(list)
    for label_file in os.listdir(labels_path):
        with open(os.path.join(labels_path, label_file), 'r') as file:
            for line in file:
                class_id = int(line.split()[0])
                labels[class_id].append(label_file)
    return labels

# Function to print label counts
def print_label_counts(labels, message):
    print(message)
    for class_id, files in labels.items():
        print(f"Class {class_id}: {len(files)} instances")
    print()

# Function to undersample dataset
def undersample_dataset(labels, target_count):
    for class_id, files in labels.items():
        selected_files = random.sample(files, min(len(files), target_count))
        for file in selected_files:
            # Copy label file
            shutil.copyfile(os.path.join(labels_path, file), os.path.join(undersampled_labels_path, file))
            # Copy corresponding image file
            image_file = file.replace('.txt', '.jpg') # change the extension as per your dataset
            shutil.copyfile(os.path.join(images_path, image_file), os.path.join(undersampled_images_path, image_file))

# Analyzing the dataset
labels = get_labels(labels_path)

# Print initial label counts
print_label_counts(labels, "Label counts before undersampling:")

# Finding the minimum number of instances across all classes
min_instances = min([len(files) for files in labels.values()])

# Undersample the dataset to have the same number of instances for each class (min_instances)
undersample_dataset(labels, min_instances)

# Print label counts for the undersampled dataset
print_label_counts(get_labels(undersampled_labels_path), "Label counts in the undersampled dataset:")

print("Dataset has been undersampled and saved to the 'undersampled' directory.")
