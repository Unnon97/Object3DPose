# Object Pose Estimation Project: Source folder

This folder contains scripts used during the project

## Folder description
The folder contents can be described as follows:
1. "object_pose_estimation.py" script is the main script called by the HTTP server app for extracting object pose as per the given image
2. "click_pnp.py" script helps in providing the pixel locations based on the click we do on the image
3. "dataset.py" script is a dataloader created earlier for easier data importing
4. "kmeanssegment.py" script contains the implementation experiment of the idea of using kmeans clustering to find the object surface for pose extraction
5. "meanshiftsegment.py" script contains the implementation experiment of the idea of using mean shift approach for segmenting the object separate from the background. This approach is too slow per image so discarded. 
6. "sam.py" script is the direct implemtation of segmenting using SAM model with model type "vit_h_4b8939". (https://github.com/facebookresearch/segment-anything)
7. "utils_cornerextracion.py" script is used by pose_estimation.py script for finding the corners or edges on the object for feature extraction