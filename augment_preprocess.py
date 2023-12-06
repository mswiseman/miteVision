import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
import os
import random

'''
This script is used to augment images and their annotations for object detection. 
It uses the YOLO format for annotations. 

Written by Michele Wiseman, December 5th 2023
'''

def rotate_bbox_content(image, bbox, angle, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT,
                        border_value=(0, 0, 0)):
    """
    Rotate the content inside a bounding box.
    bbox: [x_center, y_center, width, height]
    angle: Rotation angle in degrees.
    """
    h, w = image.shape[:2]
    x_center, y_center, width, height = bbox
    x_min = int((x_center - width / 2) * w)
    y_min = int((y_center - height / 2) * h)
    x_max = int((x_center + width / 2) * w)
    y_max = int((y_center + height / 2) * h)

    # Extract the region of interest (ROI)
    roi = image[y_min:y_max, x_min:x_max]

    # Calculate the rotation matrix
    roi_center = ((x_max - x_min) / 2, (y_max - y_min) / 2)
    rotation_matrix = cv2.getRotationMatrix2D(roi_center, angle, 1.0)

    # Perform the rotation
    rotated_roi = cv2.warpAffine(roi, rotation_matrix, (x_max - x_min, y_max - y_min), flags=interpolation,
                                 borderMode=border_mode, borderValue=border_value)

    # Place the rotated ROI back on the image
    image[y_min:y_max, x_min:x_max] = rotated_roi

    return image

def clip_bbox(bbox):
    """
    Clip the bounding box coordinates to be within the range [0.0, 1.0].
    """
    x_min, y_min, x_max, y_max, class_label = bbox
    x_min = max(min(x_min, 1.0), 0.0)
    y_min = max(min(y_min, 1.0), 0.0)
    x_max = max(min(x_max, 1.0), 0.0)
    y_max = max(min(y_max, 1.0), 0.0)
    return [x_min, y_min, x_max, y_max, class_label]

def convert_to_yolov5_format(annotations):
    """
    Convert annotations from [x_center, y_center, width, height, class] format
    to [class, x_center, y_center, width, height] format.
    """
    yolov5_annotations = []
    for ann in annotations:
        class_label = int(ann[-1])
        bbox = ann[:-1]
        yolov5_annotations.append([class_label] + bbox)

    return yolov5_annotations

def get_augmentation_pipeline():
    """
    Create an Albumentations augmentation pipeline.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # Add more augmentations here as needed
    ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.3, label_fields=['class_labels']))

def apply_augmentations(image, annotations, augmentation_pipeline):
    """
    Apply augmentations to an image and its annotations, including conditional rotation of bbox contents.
    """
    class_labels = [ann[0] for ann in annotations]
    bboxes = [ann[1:] for ann in annotations]  # Separate class labels from bbox coordinates

    # Apply other augmentations
    augmented = augmentation_pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
    augmented_image = augmented['image']
    augmented_bboxes = augmented['bboxes']

    # Rotate some bounding boxes' contents
    for i, bbox in enumerate(augmented_bboxes):
        if random.random() < 0.1:  # 10% chance to rotate
            augmented_image = rotate_bbox_content(augmented_image, bbox, angle=180)

    # Reattach class labels and clip the bounding boxes
    clipped_bboxes = []
    for bbox, class_label in zip(augmented_bboxes, class_labels):
        clipped_bbox = clip_bbox((*bbox, class_label))
        clipped_bboxes.append(clipped_bbox)

    return augmented_image, clipped_bboxes

def save_image(image, output_dir, file_name):
    """
    Save an image to the specified directory with the given file name.
    """
    cv2.imwrite(os.path.join(output_dir, file_name), image)

def save_annotations(annotations, output_dir, file_name):
    """
    Save YOLO formatted annotations to the specified directory with the given file name.
    """
    annotation_path = os.path.join(output_dir, file_name.replace('.jpg', '.txt'))
    with open(annotation_path, 'w') as file:
        for ann in annotations:
            file.write(' '.join(map(str, ann)) + '\n')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default="/Users/michelewiseman/Downloads/output", required=True, help="Directory to save augmented images and annotations")
    parser.add_argument('--image_dir', type=str, default="/Users/michelewiseman/Downloads/TSSM Detection v2.v34i.yolov5pytorch/valid/images", required=True, help="Directory containing images to augment")
    parser.add_argument('--annotation_dir', default="/Users/michelewiseman/Downloads/TSSM Detection v2.v34i.yolov5pytorch/valid/labels", type=str, required=True, help="Directory containing annotations to augment")
    parser.add_argument('--num_augmentations_per_image', type=int, default=5, required=True, help="Number of augmentations to apply to each image")

    args = parser.parse_args()

    output_dir = args.output_dir
    image_dir = args.image_dir
    annotation_dir = args.annotation_dir
    num_augmentations_per_image = args.num_augmentations_per_image

    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        annotation_path = os.path.join(annotation_dir, image_name.replace('.jpg', '.txt'))

        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Image at {image_path} could not be loaded")

            annotations = []
            with open(annotation_path, 'r') as file:
                for line in file:
                    annotations.append([float(x) for x in line.strip().split()])

            augmentation_pipeline = get_augmentation_pipeline()
            for i in range(num_augmentations_per_image):
                augmented_image, augmented_annotations = apply_augmentations(image, annotations, augmentation_pipeline)
                augmented_image_name = f'{image_name.split(".")[0]}_augmented_{i}.jpg'
                save_image(augmented_image, output_dir, augmented_image_name)
                yolo_augmented_annotations = convert_to_yolov5_format(augmented_annotations)
                save_annotations(yolo_augmented_annotations, output_dir, f'{augmented_image_name}')
        except IOError as e:
            print(f"Error processing {image_name}: {e}")
        except ValueError as e:
            print(f"Value Error: {e}")
