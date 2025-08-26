# Overview

This folder contains a bash script ([`jobs.sh`](jobs.sh)) that provides examples of how call [`inference_testing.py`](inference_testing.py) jobs in series as well as the actual inference testing script. 

![example of custom inference testing outputs](/assets/images/example_outputs.png)
**Examples of a few types of outputs provided by `inference_testing.py`.** A. Output of predictions and ground truth overlaid on the original image. B. Missed detection map output - shows where the missed detections were found and helps diagnose performance issues. 

## Requirements
Python 3.9-3.11, [CUDA](https://developer.nvidia.com/cuda-toolkit) (if you want to utilize GPU) as well as:

```
# Pytorch framework
torch>=2.1,<3
torchvision>=0.16

# YOLOv11 
ultralytics>=8.3.4

# Image/vision utilities
supervision>=0.20.0
pillow>=9.1.0
opencv-python>=4.8.0
numpy>=1.24

# Plotting / stats
matplotlib>=3.7
scipy>=1.10
```

## Example of file layout
To successfully run ground truth evaluation, the files should be in the standard YOLO format (`./test/images` and `./test/labels`) like so: 

```bash

/test/
  images/
    07-18-2025_1dpi/
      img001.jpg
      img002.jpg
    07-19-2025_2dpi/
      img003.jpg
  labels/
    07-18-2025_1dpi/
      img001.txt
      img002.txt
    07-19-2025_dpi/
      img003.txt

```

## Parallelizing Runs
Inference testing, even with all the print-outs and outputs I've included, runs pretty fast; however, you can parallelize it if you have the computational capacity (put the commands in an array, define a max_jobs value, and then loop through execution of commands until max_jobs is reached).

```bash
# Define the maximum number of concurrent jobs
MAX_JOBS=3

# Array of commands
commands=(
"time python inference_testing.py --model_versions 209 --output_dir /output/ --confidence_threshold 0.5 --csv_file_name example.csv --base_dir /test/model_209/images/ --model_path /models/detection_v209_5_class.pt --ground_truth --gt_label_dir /test/model_209/labels/"
"time python inference_testing.py --model_versions 210 --output_dir /output/ --confidence_threshold 0.5 --csv_file_name example.csv --base_dir /test/model_210/images/ --model_path /models/detection_v210_3_class.pt --ground_truth --gt_label_dir /test/model_210/labels/"
"time python inference_testing.py --model_versions 211 --output_dir /output/ --confidence_threshold 0.5 --csv_file_name example.csv --base_dir /test/model_211/images/ --model_path /models/detection_v211_4_class.pt --ground_truth --gt_label_dir /test/model_211/labels/"
)
# Function to wait for a free job slot
wait_for_free_job_slot() {
    while : ; do
        # Count the number of running jobs
        jobs_running=$(jobs -r | wc -l)
        # If the number of running jobs is less than the max, break the loop
        if [[ $jobs_running -lt $MAX_JOBS ]]; then
            break
        fi

        # Wait for a short period before checking again
        sleep 1
    done
}

# Execute each command
for cmd in "${commands[@]}"; do
    # Execute command in background
    eval $cmd &


    # Wait for a free job slot
    wait_for_free_job_slot
done

# Wait for all background jobs to finish
wait
```
