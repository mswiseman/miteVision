#!/usr/bin/env bash

time python inference_testing.py                \
--model_versions 210                            \
--output_dir /output/                           \
--confidence_threshold 0.5                      \
--csv_file_name example.csv                     \ # output csv
--model_path /models/detection_v210_3_class.pt  \
--ground_truth                                  \ 
--base_dir /data/modelv210/test/images/         \
--gt_label_dir /data/modelv210/test/labels/     \
--heatmap_performance
