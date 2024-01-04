import os

# Set the path to your dataset labels
labels_path = '/Users/michelewiseman/Downloads/TSSM Detection v2.v44-2023-12-11-2-15pm-no-augmentations.yolov8/valid/undersampled/labels'

# Function to change class 6 to class 5 in a label file
def change_class_in_label(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    changed = False
    with open(file_path, 'w') as file:
        for line in lines:
            parts = line.split()
            class_id = int(parts[0])
            if class_id == 6:
                parts[0] = '5'  # Change class 6 to class 5
                changed = True
            file.write(' '.join(parts) + '\n')

    return changed

# Iterate over all label files and modify them if needed
print("Files to check: ", end='')
changed_files = 0
for label_file in os.listdir(labels_path):
    label_file_path = os.path.join(labels_path, label_file)
    if change_class_in_label(label_file_path):
        changed_files += 1

print(f"Completed modifying {changed_files} files.")
