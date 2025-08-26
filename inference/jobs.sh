#!/usr/bin/env bash

time python inference_testing.py                   \
--model_versions 209                               \
--output_dir ../test/output                        \
--confidence_threshold 0.5                         \
--csv_file_name example.csv                        \ # output csv
--model_path ../models/detection_v209_5_class.pt   \
--ground_truth                                     \ 
--base_dir ../test/images/                         \
--gt_label_dir ../test/labels/                     \
--heatmap_performance
