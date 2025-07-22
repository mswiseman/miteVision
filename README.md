# miteVision 

# Introduction

# Overiview

The code and logs for training the most recent classification and segmentation models can be [here](classification_and_segmentation_training_runs.ipynb).

All data and models are hosted on [Roboflow](https://universe.roboflow.com/gent-lab/). The latest models, their performance metrics, and specifications below:

| Version | Type |  Details                                     | Stage | Images | Instances | Precision | Recall | mAP50 | mAP50-95 | #Classes | Classes                                                  |
|---------|---------|---------------------------------------------|-------|--------|-----------|-----------|--------|--------|-----------|----------|-----------------------------------------------------------|
| [v209](https://universe.roboflow.com/gent-lab/tssm-detection-v2/model/209) | Object Detection  | With synthetic data, patience = 25, no pre-train augmentations | Val | 118    | 1841      | 0.836     | 0.822  | 0.869  | 0.559     | 5        | Adult_female, Immature, Viable_egg                        |
| [v209](https://universe.roboflow.com/gent-lab/tssm-detection-v2/model/209) | Object Detection  | With synthetic data, patience = 25, no pre-train augmentations | Test  | 117    | 2093      | 0.824     | 0.743  | 0.788  | 0.532     | 5        | Adult_female, Adult_male, Dead_mite, Immature, Viable_egg |
| [v210](https://universe.roboflow.com/gent-lab/tssm-detection-v2/model/210) | Object Detection  | With synthetic data, patience = 25, no pre-train augmentations | Val | 118    | 1683      | 0.892     | 0.925  | 0.939  | 0.597     | 3        | Adult_female, Immature, Viable_egg                        |
| [v210](https://universe.roboflow.com/gent-lab/tssm-detection-v2/model/210) | Object Detection  | With synthetic data, patience = 25, no pre-train augmentations | Test  | 117    | 1938      | 0.875     | 0.871  | 0.883  | 0.601     | 3        | Adult_female, Immature, Viable_egg                        |
| [v211](https://universe.roboflow.com/gent-lab/tssm-detection-v2/model/211) | Object Detection  | With synthetic data, patience = 25, no pre-train augmentations | Val | 118    | 1738      | 0.871     | 0.845  | 0.913  | 0.597     | 4        | Adult_female, Adult_male, Immature, Viable_egg            |
| [v211](https://universe.roboflow.com/gent-lab/tssm-detection-v2/model/211) | Object Detection  | With synthetic data, patience = 25, no pre-train augmentations | Test  | 115    | 1949      | 0.903     | 0.835  | 0.871  | 0.576     | 4        | Adult_female, Adult_male, Immature, Viable_egg            |
| [v118](https://universe.roboflow.com/gent-lab/tssm-detection-instance-segmentation/model/118) | Instance Segmentation | With synthetic data, patience = 25, no pre-train augmentations | Val |  40 | 639   |  0.898     |  0.892    |  0.927 |  0.704 |   4    | Adult_female, Adult_male, Immature, Viable_egg            


To enable easier reproduction of our results, the referenced test sets and models in Wiseman et al. 2025 have been hosted separately [here](https://drive.google.com/drive/folders/1j1TDOzc_pnrFiZmaGc7LjP3S_jW6tjHo?usp=sharing). 

# Contact



