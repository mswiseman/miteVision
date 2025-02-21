This directory has the necessary components to create synthetic images.

`generate_segmentation.js`: 
- requires segmented training data separated by class in directories under `./Training`
- requires background images
  - need to list filenames in `Background_Images.filtered.txt`
  - need background images that will be overlaid with segmented objects in `./Background_Images`
- You can specify number of objects and augmentations to those objects within the script
- Randomly places objects on random background images
- Saves new image and object mask
- To convert to YOLO format, need to run `convert_masks_to_yolo.py` after

`convert_masks_to_yolo.py`:
- requires directories where images and masks from `generate_segmentation.js` are located (`./Output` and `./Output/Masks`).
- produces YOLO polygon annotation label

`generate_bounding_boxes.js`: 
- requires segmented training data separated by class in directories under `./Training`
- requires background images
  - need to list filenames in `Background_Images.filtered.txt`
  - need background images that will be overlaid with segmented objects in `./Background_Images`
- You can specify number of objects and augmentations to those objects within the script
- Randomly places objects on random background images
- Saves new image and .xml with bounding box coordinates

`change_background_of_synthetic_image.py`:
- requires images with associated YOLO polygon segmentations `./Background_Swap/old_images` and `./Background_Swap/old_labels` as well as the new backgrounds you would like to place your segmentations on `./Background_Swap/replacement_backgrounds`
- outputs `./Background_Swap/new_labels` and `./Background_Swap/new_images` in YOLO format 
 



  
