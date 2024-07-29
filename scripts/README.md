# Brick Pose Estimation Project: Scripts folder

This folder contains scripts used during the project

## Folder description
The folder contents can be described as follows:
1. "app.py" script creates the HTTP server app that needs to run for obtaining the 3D pose of the brick
2. "request.py" script uses the given RGB, Depth and camera metadata json file and requests the HTTP server app to respond with position and orientation of the brick
3. "sam_based.py" script is again taken directly from the Segment Anything Model and saves the segmented image here. (https://github.com/facebookresearch/segment-anything)