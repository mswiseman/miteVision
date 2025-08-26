# YOLO Dataset Utilities  

A collection of Python scripts for preparing, cleaning, and augmenting **YOLO-format object detection datasets**.  
These tools help with tasks like augmentation, cross-validation splits, deduplication, class balancing, and class remapping.  

---

## 📦 Requirements  
- Python 3.8+  
- Common libraries: `opencv-python`, `albumentations`, `scikit-learn`, etc.  
- Some scripts may require additional tools (e.g., Helicon Focus, XnView for `stackPhotosParallel.py`).  


---

## 🛠 Scripts  

### [`augment_preprocess.py`](augment_preprocess.py)  
Image augmentation tool for object detection datasets in YOLO format.  

> *Note: Modern YOLO implementations already include extensive augmentations during training. This script is useful for offline augmentation or when you want full control.*  

**Example usage:**  
```bash
python augment_preprocess.py   --image_dir /path/to/images   --annotation_dir /path/to/labels   --output_dir /path/to/output   --num_augmentations_per_image 5
```

---

### [`calculate_mean_std.py`](calculate_mean_std.py)
Calculate the mean and standard deviation of RGB values for each subdirectory in a directory, excluding black padding (pixels where R=G=B=0).  

**Example usage:**  
```bash
python calculate_mean_std.py --directory /path/to/images
```

---

### [`cross_validation.py`](cross_validation.py)
Create K-fold cross-validation splits for a YOLO dataset with custom regex matching to ensure **synthetic images remain exclusively in the training splits**.  

**Example usage:**  
```bash
python cross_validation.py   --dataset_dir /TSSM-Detection-v2-209/train   --output_dir /TSSM-Detection-v2-209/cv_folds   --num_folds 5   --image_ext jpg   --synthetic_regex '^(image_[0-9]+|[0-9]+)_'   --seed 42
```

---

### [`find_exact_duplicates.py`](find_exact_duplicates.py)  
Detects and optionally removes **exact, byte-for-byte duplicate images** to help prevent train/test leakage.  

**Example usage:**  
```bash
python find_exact_duplicates.py /path/to/directory --delete
```

---

### [`remove_excess_class_data.py`](remove_excess_class_data.py)  
Removes image/label pairs if a label file contains **too many instances of a target class** and **not enough of a limiting class**.  
Useful for controlling class imbalance.  

**Example usage:**  
```bash
python remove_excess_class_data.py   --dataset_path "/path/to/dataset/undersampled"   --target_class_id 4   --max_allowed_instances 5   --limit_class_id 0   --limit_min_count 2   --image_ext jpg
```

---

### [`remap_classes.py`](remap_classes.py)
Remap or drop YOLO class IDs across all label files.  

**Example usage:**  
```bash
python remap_classes.py   --label_dir "/path/to/labels"   --class_map "0:0,1:1,2:drop,3:2,4:3"
```

---

### [`stackPhotosParallel.py`](stackPhotosParallel.py)  
Focus-stacks raw photos from Blackbird using **Helicon Focus**, then compresses them by converting `.tif` to `.png`.  
Supports optional raw image backup. Need Helicon Focus license to run. 

**Example usage:**  
```bash
python stackPhotosParallel.py   --input_dir "/d/Unstacked"   --output_dir_base "/d/Stacked"   --helicon_focus_path "/path/to/heliconfocus"   --nconvert_path "path/to/XnView"   --max_workers 4
```

---

### [`undersampling.py`](undersampling.py) 
Scans a dataset of images and YOLO-format labels, then creates an **undersampled dataset** where each class has the same number of instances.  

**Example usage:**  
```bash
python undersampling.py   --images_path /path/to/train/images   --labels_path /path/to/train/labels   --output_dir undersampled
```

---
