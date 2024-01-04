import os

# Set dataset paths
dataset_path = '/Users/michelewiseman/Downloads/TSSM Detection v2.v44-2023-12-11-2-15pm-no-augmentations.yolov8/test/undersampled'
images_path = os.path.join(dataset_path, 'images')
labels_path = os.path.join(dataset_path, 'labels')
target_class_id = 6  # Class ID to check
max_allowed_instances = 5  # Maximum allowed instances of the target class
limit_class_id = 0  # Class ID to limit

# Function to count instances of the target class in the label file
def count_target_class_instances(label_file):
    count = 0
    with open(label_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            class_id = int(line.split()[0])
            if class_id == target_class_id:
                count += 1
    return count

def limiting_class_instances(label_file):
    limit_count = 0
    with open(label_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            class_id = int(line.split()[0])
            if class_id == limit_class_id:
                limit_count += 1
    return limit_count

# Iterate over the label files and remove files if the condition is met
for label_file in os.listdir(labels_path):
    label_file_path = os.path.join(labels_path, label_file)
    if count_target_class_instances(label_file_path) > max_allowed_instances and limiting_class_instances(label_file_path) < 2:
        os.remove(label_file_path)  # Remove label file

        # Construct the corresponding image file name and remove
        image_file_name = label_file.replace('.txt', '.jpg')  # Change file extension as per your dataset
        image_file_path = os.path.join(images_path, image_file_name)
        if os.path.exists(image_file_path):
            os.remove(image_file_path)
            print(f"Removed {image_file_path} due to containing more than {max_allowed_instances} instances of the target class ID ({target_class_id}).")
print("Completed removing images and labels with more than 5 instances of the specified class.")
